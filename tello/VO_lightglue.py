"""Odometría visual monocular con SuperPoint + LightGlue sobre el vídeo del Tello.

Dependencias (venv propio de tello/):
    uv venv .venv --python 3.11
    uv pip install torch --index-url https://download.pytorch.org/whl/cu126
    uv pip install "git+https://github.com/cvg/LightGlue.git" opencv-python numpy plotly pillow av
    uv pip install transformers   # solo para --denso / --planos

Ejemplo:
    .venv/Scripts/python.exe VO_lightglue.py sesiones/20261017_113325
    .venv/Scripts/python.exe VO_lightglue.py sesiones/20261017_113325 --planos
    .venv/Scripts/python.exe VO_lightglue.py sesiones/20261017_113325 --start 300 --end 900

Flujo por pares de keyframes:
  1. SuperPoint extrae puntos y descriptores; LightGlue los empareja.
  2. Un fotograma se convierte en keyframe cuando el flujo mediano supera
     --min-flow píxeles (o tras --max-gap fotogramas).
  3. Matriz esencial (USAC MAGSAC) + recoverPose -> R, t con |t| = 1.
  4. Fusión con la actitud del Tello (estado.csv, salvo --sin-imu): si la
     rotación de la VO discrepa más de --imu-max-err grados de la de la IMU se
     usa la de la IMU y t se recalcula con R fija; además, en cada keyframe la
     orientación se acerca un --imu-alpha a la de la IMU (filtro complementario).
     El mundo queda nivelado con la gravedad (z hacia arriba en las gráficas).
  5. Triangulación de los inliers. Escala: si se está andando, la línea base del
     paso son los metros que marca el podómetro de la IMU (podometro.py, ajustable
     con --zancada); parado, se propaga la escala relativa comparando la
     profundidad de los puntos compartidos con el paso anterior. Los pasos sin
     traslación visual (rotación pura, pérdidas) también avanzan lo andado.
  6. Si el paralaje es casi nulo se trata como rotación pura (t = 0).
  7. Con --denso, 1 de cada --denso-cada keyframes se densifica con Depth
     Anything V2 escalado a la VO (ver densificar.py) y se fusiona por vóxeles.
  8. Con --planos, además se detectan planos, se asocian entre tomas y los
     puntos se proyectan sobre su plano definitivo (ver planos.py).

Salidas en resultados_vo_lightglue/<sesion>/:
    poses.csv, puntos_3d.ply, puntos_densos.ply (con --denso), informe.json,
    planos.json y puntos_planos.ply (con --planos),
    trayectoria.html (Plotly), reconstruccion.mp4 y reconstruccion.gif (paso a paso).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import plotly.graph_objects as go
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch
from PIL import Image
from plotly.subplots import make_subplots

from densificar import Densificador, fusionar_voxeles
from planos import MapaPlanos, detectar_planos
from podometro import crear_podometro

HERE = Path(__file__).resolve().parent

# Cámara (x derecha, y abajo, z adelante) -> mundo de visualización (x derecha,
# y adelante, z arriba), tomando como referencia la cámara del primer keyframe.
CAM_TO_PLOT = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
UP_WORLD = CAM_TO_PLOT.T @ np.array([0, 0, 1.0])  # "arriba" en coordenadas de mundo


# --------------------------------------------------------------------------
# Entrada
# --------------------------------------------------------------------------


def resolve_inputs(path: Path):
    """Admite una carpeta de sesión (video.mp4 + frames.csv + estado.csv) o un MP4."""
    if path.is_dir():
        video = path / "video.mp4"
        frames_csv = path / "frames.csv"
        estado_csv = path / "estado.csv"
        name = path.name
    else:
        video, frames_csv, estado_csv, name = path, None, None, path.stem
    if not video.is_file():
        raise FileNotFoundError(f"No existe el vídeo: {video}")
    stamps = None
    if frames_csv is not None and frames_csv.is_file():
        with frames_csv.open(newline="", encoding="utf-8-sig") as f:
            stamps = np.array([float(r["t_host"]) for r in csv.DictReader(f)])
    estado = None
    if estado_csv is not None and estado_csv.is_file():
        with estado_csv.open(newline="", encoding="utf-8-sig") as f:
            filas = list(csv.DictReader(f))
        if filas:
            estado = {k: np.array([float(r[k]) for r in filas]) for k in filas[0]}
    return video, stamps, estado, name


def iter_video(video: Path, start: int, end: int | None, step: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    idx = start
    try:
        while end is None or idx < end:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % step == 0:
                yield idx, fps, frame
            idx += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Geometría
# --------------------------------------------------------------------------


def make_K(w, h, fx=None, fy=None, cx=None, cy=None):
    # Mismo K orientativo que VO_2_frames.py: sustituir por una calibración real.
    fx = float(fx if fx is not None else 0.85 * w)
    fy = float(fy if fy is not None else fx)
    cx = float(cx if cx is not None else (w - 1) / 2)
    cy = float(cy if cy is not None else (h - 1) / 2)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def rays(K, uv):
    x = np.linalg.inv(K) @ np.column_stack([uv, np.ones(len(uv))]).T
    return (x / np.linalg.norm(x, axis=0)).T


def triangulate(K, R, t, uv0, uv1):
    """Triangula en el marco del keyframe 0 (|t| = 1). Devuelve X (Nx3) y máscara válida."""
    P0 = K @ np.column_stack([np.eye(3), np.zeros(3)])
    P1 = K @ np.column_stack([R, t])
    Xh = cv2.triangulatePoints(P0, P1, uv0.T.astype(float), uv1.T.astype(float)).T
    X = Xh[:, :3] / Xh[:, 3:4]
    X1 = (R @ X.T).T + t
    valid = (X[:, 2] > 0) & (X1[:, 2] > 0) & (X[:, 2] < 80)
    for P, Xc, uv in ((P0, X, uv0), (P1, X1, uv1)):
        proj = (K @ Xc.T).T
        err = np.linalg.norm(proj[:, :2] / proj[:, 2:3] - uv, axis=1)
        valid &= err < 2.0
    # Paralaje por punto: rayos casi paralelos dan profundidades inestables.
    c1 = -R.T @ t
    a = X / np.linalg.norm(X, axis=1, keepdims=True)
    b = X - c1
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    parallax = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))
    valid &= parallax > 0.5
    return X, valid


def orientation_deg(R_wc):
    """Yaw (horario, como el Tello), pitch (morro arriba +) y roll en el marco de visualización."""
    Rp = CAM_TO_PLOT @ R_wc
    right, down, fwd = Rp[:, 0], Rp[:, 1], Rp[:, 2]
    yaw = np.degrees(np.arctan2(fwd[0], fwd[1]))
    pitch = np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))
    roll = np.degrees(np.arctan2(-right[2], -down[2]))
    return yaw, pitch, roll


# --------------------------------------------------------------------------
# Odometría
# --------------------------------------------------------------------------


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Cuerpo del Tello (x adelante, y derecha, z abajo) <- cámara (x derecha, y abajo, z adelante).
R_BODY_CAM = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])


def imu_attitude_fn(estado, t0_host):
    """t_rel -> R_wc de la cámara según la actitud del Tello.

    Convención comprobada con los datos (2026-10-22): NED, R = Rz(yaw)·Ry(pitch)·Rx(roll),
    todos con signo positivo, desfase con el vídeo despreciable. El marco de mundo
    devuelto es el de la cámara (x derecha, y abajo, z adelante) pero nivelado con la
    gravedad y con el yaw inicial a cero, así que CAM_TO_PLOT lo deja con z hacia arriba.
    """
    te = estado["t_host"] - t0_host
    ang = {k: np.unwrap(np.radians(estado[k])) for k in ("yaw", "pitch", "roll")}
    yaw0 = float(np.interp(0.0, te, ang["yaw"]))
    level = R_BODY_CAM.T @ _rz(-yaw0)

    def att(t):
        y, p, r = (float(np.interp(t, te, ang[k])) for k in ("yaw", "pitch", "roll"))
        return level @ _rz(y) @ _ry(p) @ _rx(r) @ R_BODY_CAM

    return att


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def blend_rotation(R, R_target, alpha):
    """Gira R una fracción alpha del camino hacia R_target (slerp en SO(3))."""
    rv, _ = cv2.Rodrigues(R_target @ R.T)
    dR, _ = cv2.Rodrigues(alpha * rv)
    return dR @ R


def translation_known_rotation(K, R, uv0, uv1, iters=3):
    """Dirección de t (|t| = 1) con R conocida: t · (R·x0 × x1) = 0 para cada pareja.

    Mínimos cuadrados con rechazo iterativo de outliers; el signo se fija por
    quiralidad (más puntos triangulados delante de ambas cámaras).
    """
    x0 = np.linalg.inv(K) @ np.column_stack([uv0, np.ones(len(uv0))]).T
    x1 = np.linalg.inv(K) @ np.column_stack([uv1, np.ones(len(uv1))]).T
    A = np.cross((R @ x0).T, x1.T)
    keep = np.ones(len(A), bool)
    t = None
    for _ in range(iters):
        if keep.sum() < 8:
            break
        t = np.linalg.svd(A[keep])[2][-1]
        res = np.abs(A @ t)
        keep = res < max(3 * np.median(res[keep]), 1e-4)
    if t is None:
        return None
    best = max((t, -t), key=lambda tt: triangulate(K, R, tt, uv0, uv1)[1].sum())
    return best


@dataclass
class Keyframe:
    idx: int
    t_rel: float
    feats: dict
    kpts: np.ndarray
    image: np.ndarray
    T_wc: np.ndarray
    scale: float
    depth_ref: np.ndarray  # profundidad (unidades de mundo) por keypoint; nan si no hay


@dataclass
class Step:
    """Una pose estimada más todo lo necesario para dibujarla después."""

    idx: int
    t_rel: float
    T_wc: np.ndarray
    status: str
    matches: int = 0
    inliers: int = 0
    flow_px: float = 0.0
    parallax_deg: float = 0.0
    scale: float = 1.0
    new_points: int = 0
    jpg: bytes = b""
    uv_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    uv_cur: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    inlier_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    dense_points: int = 0
    rot_imu: bool = False  # la rotación de este paso se tomó de la IMU (salto de la VO)
    avance_pod: float = 0.0  # metros andados en este paso según el podómetro


class LightGlueVO:
    def __init__(self, args, device, imu_att=None, densificador=None, podometro=None):
        self.args = args
        self.device = device
        self.imu_att = imu_att  # t_rel -> R_wc según la IMU (mundo nivelado), o None
        self.densificador = densificador
        self.dense_P: list[np.ndarray] = []
        self.dense_C: list[np.ndarray] = []
        self.dense_owner: list[np.ndarray] = []
        self.dense_z: list[float] = (
            []
        )  # profundidad mediana de cada keyframe densificado
        self.dense_plane: list[np.ndarray] = (
            []
        )  # id de plano global por punto (-1 = ninguno)
        self.mapa_planos = MapaPlanos() if getattr(args, "planos", False) else None
        self._kf_sin_denso = 0
        self.podometro = podometro  # distancia andada [m] -> línea base métrica, o None
        self.dir_w = None  # última dirección de avance en el mundo (para pasos sin t)
        self.extractor = SuperPoint(max_num_keypoints=args.max_kpts).eval().to(device)
        self.matcher = LightGlue(features="superpoint").eval().to(device)
        self.K = None
        self.kf: Keyframe | None = None
        self.steps: list[Step] = []
        self.points: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []

    @torch.inference_mode()
    def extract(self, bgr):
        img = numpy_image_to_torch(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        feats = self.extractor.extract(img.to(self.device), resize=self.args.resize)
        return feats, feats["keypoints"][0].cpu().numpy()

    @torch.inference_mode()
    def match(self, f0, f1):
        out = self.matcher({"image0": f0, "image1": f1})
        return out["matches"][0].cpu().numpy()

    def _encode(self, bgr):
        small = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def _new_keyframe(
        self, idx, t_rel, feats, kpts, image, T_wc, scale, depth_ref=None
    ):
        if depth_ref is None:
            depth_ref = np.full(len(kpts), np.nan)
        self.kf = Keyframe(idx, t_rel, feats, kpts, image, T_wc, scale, depth_ref)

    def process(self, idx, t_rel, bgr):
        a = self.args
        if self.K is None:
            h, w = bgr.shape[:2]
            self.K = make_K(w, h, a.fx, a.fy, a.cx, a.cy)
        feats, kpts = self.extract(bgr)
        if self.kf is None:
            T0 = np.eye(4)
            if self.imu_att is not None:
                T0[:3, :3] = self.imu_att(t_rel)  # mundo nivelado con la gravedad
            self._new_keyframe(idx, t_rel, feats, kpts, bgr, T0, 1.0)
            self.steps.append(Step(idx, t_rel, T0, "inicio", jpg=self._encode(bgr)))
            # Dirección de avance inicial: hacia donde mira la cámara, en horizontal.
            fwd = T0[:3, 2] - (T0[:3, 2] @ UP_WORLD) * UP_WORLD
            self.dir_w = fwd / (np.linalg.norm(fwd) + 1e-12)
            return
        kf, K = self.kf, self.K
        gap = idx - kf.idx
        m = self.match(kf.feats, feats)
        uv0, uv1 = kf.kpts[m[:, 0]], kpts[m[:, 1]]
        flow = float(np.median(np.linalg.norm(uv1 - uv0, axis=1))) if len(m) else 0.0

        if len(m) >= a.min_inliers and flow < a.min_flow and gap < a.max_gap:
            return  # poco movimiento: esperamos a un fotograma con más paralaje

        if len(m) < a.min_inliers:
            if gap < a.max_gap:
                return
            self._lost(idx, t_rel, feats, kpts, bgr, len(m), flow)
            return

        E, mask = cv2.findEssentialMat(
            uv0, uv1, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=1.5
        )
        if E is None or E.shape != (3, 3) or mask is None or mask.sum() < a.min_inliers:
            if gap >= a.max_gap:
                self._lost(idx, t_rel, feats, kpts, bgr, len(m), flow)
            return
        n_in, R, t, mask_pose = cv2.recoverPose(E, uv0, uv1, K, mask=mask.copy())
        inl = mask.ravel().astype(bool)
        if n_in < a.min_inliers:
            if gap >= a.max_gap:
                self._lost(idx, t_rel, feats, kpts, bgr, len(m), flow)
            return
        t = t.ravel()

        # Salto de rotación: si la VO y la IMU discrepan, manda la IMU y t se
        # recalcula con esa rotación fija.
        rot_imu = False
        if self.imu_att is not None:
            R_imu = self.imu_att(t_rel).T @ self.imu_att(kf.t_rel)
            if rot_angle_deg(R @ R_imu.T) > a.imu_max_err:
                t_imu = translation_known_rotation(K, R_imu, uv0[inl], uv1[inl])
                if t_imu is not None:
                    R, t, rot_imu = R_imu, t_imu, True

        # Paralaje tras compensar la rotación: separa traslación de rotación pura.
        r0, r1 = rays(K, uv0[inl]), rays(K, uv1[inl])
        cosang = np.sum(r1 * (R @ r0.T).T, axis=1)
        parallax = float(np.median(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
        if parallax < a.min_parallax and gap < a.max_gap and n_in >= 3 * a.min_inliers:
            return  # aún hay solape de sobra: esperamos a tener más línea base

        idx0, idx1 = m[inl, 0], m[inl, 1]
        depth_new = np.full(len(kpts), np.nan)
        new_pts = 0
        ds = self._avance(kf.t_rel, t_rel)  # metros andados según el podómetro
        if parallax < a.min_parallax:
            status, s = "rotacion", kf.scale
            T_j_kf = np.eye(4)
            T_j_kf[:3, :3] = R
            # Las profundidades conocidas se trasladan al nuevo keyframe girándolas.
            d = kf.depth_ref[idx0]
            ok = ~np.isnan(d)
            r = rays(K, uv0[inl][ok])
            Xkf = r / r[:, 2:3] * d[ok, None]
            depth_new[idx1[ok]] = (R @ Xkf.T).T[:, 2]
        else:
            X, valid = triangulate(K, R, t, uv0[inl], uv1[inl])
            ref = kf.depth_ref[idx0]
            common = valid & ~np.isnan(ref)
            s = kf.scale
            if ds > 0:
                s = ds  # andando: la línea base la da el podómetro, en metros
            elif common.sum() >= 12:
                ratio = np.median(ref[common] / X[common, 2])
                s = float(np.clip(ratio, kf.scale / 4, kf.scale * 4))
            status = "ok"
            T_j_kf = np.eye(4)
            T_j_kf[:3, :3], T_j_kf[:3, 3] = R, s * t
            Xs = s * X[valid]
            depth_new[idx1[valid]] = ((R @ Xs.T).T + s * t)[:, 2]
            Xw = (kf.T_wc[:3, :3] @ Xs.T).T + kf.T_wc[:3, 3]
            px = np.round(uv0[inl][valid]).astype(int)
            px[:, 0] = np.clip(px[:, 0], 0, kf.image.shape[1] - 1)
            px[:, 1] = np.clip(px[:, 1], 0, kf.image.shape[0] - 1)
            self.points.append(Xw)
            self.colors.append(kf.image[px[:, 1], px[:, 0]][:, ::-1])  # BGR -> RGB
            new_pts = len(Xw)

        T_wc = kf.T_wc @ np.linalg.inv(T_j_kf)
        if status == "ok":
            mov = T_wc[:3, 3] - kf.T_wc[:3, 3]
            if ds > 0 and not a.sin_andar_horizontal:
                # Andando con el dron en la mano la altura apenas cambia: la VO solo
                # aporta el rumbo en planta (su componente vertical es la más ruidosa).
                mov_h = mov - (mov @ UP_WORLD) * UP_WORLD
                if np.linalg.norm(mov_h) > 1e-9:
                    mov = mov_h / np.linalg.norm(mov_h) * ds
                    T_wc[:3, 3] = kf.T_wc[:3, 3] + mov
            if np.linalg.norm(mov) > 0:
                self.dir_w = mov / np.linalg.norm(mov)
        elif ds > 0:
            # Rotación pura para la VO pero andando: avanzamos lo que diga el podómetro.
            T_wc[:3, 3] += ds * self.dir_w
        if self.imu_att is not None and a.imu_alpha > 0:
            # Filtro complementario: la VO manda a corto plazo, la IMU evita la deriva.
            T_wc[:3, :3] = blend_rotation(
                T_wc[:3, :3], self.imu_att(t_rel), a.imu_alpha
            )
        self.steps.append(
            Step(
                idx,
                t_rel,
                T_wc,
                status,
                len(m),
                int(inl.sum()),
                flow,
                parallax,
                s,
                new_pts,
                self._encode(bgr),
                uv0,
                uv1,
                inl,
                rot_imu=rot_imu,
                avance_pod=ds,
            )
        )
        self._new_keyframe(idx, t_rel, feats, kpts, bgr, T_wc, s, depth_new)
        self._densificar()

    def _avance(self, t0, t1):
        if self.podometro is None:
            return 0.0
        return self.podometro.distancia(t1) - self.podometro.distancia(t0)

    def _densificar(self):
        """Nube densa del keyframe actual si han pasado --denso-cada keyframes desde la última."""
        if self.densificador is None:
            return
        self._kf_sin_denso += 1
        if self._kf_sin_denso < self.args.denso_cada:
            return
        kf = self.kf
        ok = ~np.isnan(kf.depth_ref)
        X, Cg, _ = self.densificador.rejilla(
            kf.image, self.K, kf.kpts[ok], kf.depth_ref[ok]
        )
        if X is None:
            return  # ajuste poco fiable: se reintenta con el siguiente keyframe
        valid = ~np.isnan(X).any(-1)
        R_wc, t_wc = kf.T_wc[:3, :3], kf.T_wc[:3, 3]
        P = X[valid] @ R_wc.T + t_wc
        z_med = float(np.median(kf.depth_ref[ok]))
        ids = np.full(len(P), -1, int)
        if self.mapa_planos is not None:
            labels, planos_cam = detectar_planos(X)
            ids = self.mapa_planos.integrar(
                P, labels[valid], planos_cam, R_wc, t_wc, z_med, len(self.steps) - 1
            )
        self.dense_P.append(P)
        self.dense_C.append(Cg[valid])
        self.dense_owner.append(np.full(len(P), len(self.steps) - 1))
        self.dense_plane.append(ids)
        self.dense_z.append(z_med)
        self.steps[-1].dense_points = len(P)
        self._kf_sin_denso = 0

    def _lost(self, idx, t_rel, feats, kpts, bgr, n_matches, flow):
        """Sin geometría fiable: se reinicia en este fotograma conservando la posición.

        La orientación se toma de la IMU del Tello si está disponible
        (las pérdidas suelen ser giros rápidos con motion blur).
        """
        T_wc = self.kf.T_wc.copy()
        status = "perdido"
        if self.imu_att is not None:
            T_wc[:3, :3] = self.imu_att(t_rel)
            status = "perdido_imu"
        ds = self._avance(self.kf.t_rel, t_rel)
        if ds > 0:
            T_wc[:3, 3] += ds * self.dir_w  # seguimos andando aunque la imagen no sirva
        self.steps.append(
            Step(
                idx,
                t_rel,
                T_wc,
                status,
                n_matches,
                0,
                flow,
                0.0,
                self.kf.scale,
                jpg=self._encode(bgr),
                avance_pod=ds,
            )
        )
        self._new_keyframe(idx, t_rel, feats, kpts, bgr, T_wc, self.kf.scale)

    def cloud(self):
        if not self.points:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        return np.vstack(self.points), np.vstack(self.colors)

    def sparse_cloud_with_owner(self):
        """Nube dispersa y, por punto, el índice del paso que lo creó."""
        P, C = self.cloud()
        owners = [
            np.full(s.new_points, i) for i, s in enumerate(self.steps) if s.new_points
        ]
        owner = np.concatenate(owners) if owners else np.zeros(0, int)
        return P, C, owner

    def dense_cloud(self, voxel=None):
        """Nube densa fusionada por vóxeles (P, C, owner, plane_id), o None si no hay.

        Con --planos, cada punto se proyecta antes sobre su plano global definitivo.
        """
        if not self.dense_P:
            return None
        if voxel is None:
            # Separación entre muestras a la profundidad típica: z · paso / fx.
            voxel = np.median(self.dense_z) * self.densificador.paso / self.K[0, 0]
        P = np.vstack(self.dense_P)
        ids = np.concatenate(self.dense_plane)
        if self.mapa_planos is not None:
            P = self.mapa_planos.proyectar(P, ids)
        return fusionar_voxeles(
            P,
            voxel,
            np.vstack(self.dense_C),
            np.concatenate(self.dense_owner),
            ids,
        )


# --------------------------------------------------------------------------
# Salidas en disco
# --------------------------------------------------------------------------


def save_tables(out: Path, steps, P, C, dense=None):
    with (out / "poses.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx_frame",
                "t_rel_s",
                "x",
                "y",
                "z",
                "yaw_deg",
                "pitch_deg",
                "roll_deg",
                "estado",
                "matches",
                "inliers",
                "flujo_px",
                "paralaje_deg",
                "escala",
                "puntos_nuevos",
                "rot_imu",
                "avance_podometro_m",
            ]
        )
        for s in steps:
            p = CAM_TO_PLOT @ s.T_wc[:3, 3]
            w.writerow(
                [
                    s.idx,
                    f"{s.t_rel:.3f}",
                    *(f"{v:.5f}" for v in p),
                    *(f"{v:.2f}" for v in orientation_deg(s.T_wc[:3, :3])),
                    s.status,
                    s.matches,
                    s.inliers,
                    f"{s.flow_px:.1f}",
                    f"{s.parallax_deg:.2f}",
                    f"{s.scale:.4f}",
                    s.new_points,
                    int(s.rot_imu),
                    f"{s.avance_pod:.3f}",
                ]
            )
    write_ply(out / "puntos_3d.ply", P, C)
    if dense is not None:
        write_ply(out / "puntos_densos.ply", dense[0], dense[1])


def write_ply(path: Path, P, C):
    """PLY binario (x, y, z en el marco de visualización + color RGB)."""
    Pp = (CAM_TO_PLOT @ P.T).T
    v = np.empty(
        len(Pp),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    v["x"], v["y"], v["z"] = Pp.T
    v["red"], v["green"], v["blue"] = C.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(v.tobytes())


def near_trajectory(P, steps, factor=1.0):
    """Máscara de puntos (marco cámara-0) a menos de factor·extensión del centro de la trayectoria."""
    C = np.array([s.T_wc[:3, 3] for s in steps])
    lo, hi = C.min(axis=0), C.max(axis=0)
    radius = max(float(np.linalg.norm(hi - lo)), 1e-2) * factor
    return np.linalg.norm(P - (lo + hi) / 2, axis=1) < radius


def frustum_lines(T_wc, size, aspect=0.75):
    """Segmentos (pares de puntos, ya en el marco de visualización) de una pirámide de cámara."""
    C = T_wc[:3, 3]
    R = T_wc[:3, :3]
    corners = np.array(
        [[-1, -aspect, 1.4], [1, -aspect, 1.4], [1, aspect, 1.4], [-1, aspect, 1.4]]
    )
    W = (R @ (corners * size).T).T + C
    segs = [(C, W[i]) for i in range(4)] + [(W[i], W[(i + 1) % 4]) for i in range(4)]
    # Marca la parte de arriba de la imagen para ver el roll.
    segs.append((W[0], (W[0] + W[1]) / 2 + R @ np.array([0, -0.35 * size, 0])))
    segs.append(((W[0] + W[1]) / 2 + R @ np.array([0, -0.35 * size, 0]), W[1]))
    return [(CAM_TO_PLOT @ a, CAM_TO_PLOT @ b) for a, b in segs]


# Paleta de referencia (dataviz): azul, naranja, aqua; rampa azul para el tiempo.
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
TEXT2, GRID, SURFACE = "#52514e", "#e6e5e0", "#fcfcfb"
TIME_SCALE = [[0.0, "#9ec5f4"], [0.5, "#2a78d6"], [1.0, "#0d366b"]]
# Tres clases con color (los tres primeros huecos de la paleta) y el resto en gris.
PLANE_COLORS = {
    "horizontal": BLUE,
    "pared A": ORANGE,
    "pared B": AQUA,
    "otro": "#8a8984",
}


def plane_summary(mapa, dense, min_hull_points=200):
    """Tipo, normal, puntos y contorno (casco convexo en el plano) de cada plano del mapa."""
    P, ids = dense[0], dense[3]
    up_world = CAM_TO_PLOT.T @ np.array([0, 0, 1.0])
    tipos = mapa.clasificar(up_world)
    out = []
    for g, (pl, tipo) in enumerate(zip(mapa.planos, tipos)):
        Pg = P[ids == g]
        hull = None
        if len(Pg) >= min_hull_points:
            n = pl.n
            a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
            e1 = np.cross(n, a)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(n, e1)
            c = Pg.mean(axis=0)
            c -= (n @ c - pl.d) * n
            uv = np.column_stack([(Pg - c) @ e1, (Pg - c) @ e2]).astype(np.float32)
            h = cv2.convexHull(uv)[:, 0, :]
            hull = (
                CAM_TO_PLOT @ (c + np.outer(h[:, 0], e1) + np.outer(h[:, 1], e2)).T
            ).T
        out.append(
            {
                "id": g,
                "tipo": tipo,
                "normal": (CAM_TO_PLOT @ pl.n).round(4).tolist(),
                "d": round(pl.d, 5),
                "puntos": int(len(Pg)),
                "tomas": pl.tomas,
                "primer_paso": pl.primer_paso,
                "contorno": hull,
            }
        )
    return out


def save_planes(out: Path, planes, dense):
    """planos.json y puntos_planos.ply (nube densa coloreada por tipo de plano)."""
    serial = [
        {
            k: (v.round(5).tolist() if isinstance(v, np.ndarray) else v)
            for k, v in pl.items()
        }
        for pl in planes
    ]
    (out / "planos.json").write_text(
        json.dumps(serial, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    P, ids = dense[0], dense[3]
    rgb = {
        k: [int(c[i : i + 2], 16) for i in (1, 3, 5)] for k, c in PLANE_COLORS.items()
    }
    C = np.full((len(P), 3), 60, np.uint8)  # puntos sin plano en gris oscuro
    for pl in planes:
        C[ids == pl["id"]] = rgb[pl["tipo"]]
    write_ply(out / "puntos_planos.ply", P, C)


def plane_meshes(planes):
    """Un Mesh3d por tipo de plano, con cada contorno triangulado en abanico."""
    traces = []
    for tipo, color in PLANE_COLORS.items():
        xs, ys, zs, ii, jj, kk = [], [], [], [], [], []
        for pl in planes:
            h = pl["contorno"]
            if pl["tipo"] != tipo or h is None or len(h) < 3:
                continue
            base = len(xs)
            xs += h[:, 0].tolist()
            ys += h[:, 1].tolist()
            zs += h[:, 2].tolist()
            for q in range(1, len(h) - 1):
                ii.append(base)
                jj.append(base + q)
                kk.append(base + q + 1)
        if ii:
            traces.append(
                go.Mesh3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    i=ii,
                    j=jj,
                    k=kk,
                    color=color,
                    opacity=0.35,
                    name=f"Planos: {tipo}",
                    showlegend=True,
                    hoverinfo="name",
                    flatshading=True,
                )
            )
    return traces


def plotly_report(
    out: Path,
    steps,
    P,
    C,
    estado,
    t0_host,
    name,
    dense=None,
    planes=None,
    metric=False,
    max_points=40000,
    max_dense=150000,
):
    pos = np.array([CAM_TO_PLOT @ s.T_wc[:3, 3] for s in steps])
    ang = np.array([orientation_deg(s.T_wc[:3, :3]) for s in steps])
    t = np.array([s.t_rel for s in steps])
    steplen = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    fsize = (
        max(float(np.median(steplen[steplen > 0])) * 1.5, 1e-3)
        if (steplen > 0).any()
        else 0.05
    )

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"type": "scene", "colspan": 2}, None], [{}, {}], [{}, {}]],
        row_heights=[0.56, 0.22, 0.22],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        subplot_titles=(
            "Trayectoria y nube de puntos "
            + ("(metros, escala del podómetro)" if metric else "(escala arbitraria)"),
            "Posición del dron",
            "Yaw: VO vs IMU del Tello",
            "Pitch y roll (VO)",
            "Correspondencias por keyframe",
        ),
    )

    if len(P):
        keep = near_trajectory(P, steps)
        P, C = P[keep], C[keep]
    if dense is not None and len(dense[0]):
        Pd, Cd = dense[0], dense[1]
        keep = near_trajectory(Pd, steps)
        Pd, Cd = Pd[keep], Cd[keep]
        sel = np.linspace(0, len(Pd) - 1, min(max_dense, len(Pd))).astype(int)
        Pp = (CAM_TO_PLOT @ Pd[sel].T).T
        fig.add_trace(
            go.Scatter3d(
                x=Pp[:, 0],
                y=Pp[:, 1],
                z=Pp[:, 2],
                mode="markers",
                name="Nube densa (Depth Anything)",
                marker=dict(
                    size=1.4,
                    color=[f"rgb({r},{g},{b})" for r, g, b in Cd[sel]],
                    opacity=0.8,
                ),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    if len(P):
        sel = np.linspace(0, len(P) - 1, min(max_points, len(P))).astype(int)
        Pp = (CAM_TO_PLOT @ P[sel].T).T
        fig.add_trace(
            go.Scatter3d(
                x=Pp[:, 0],
                y=Pp[:, 1],
                z=Pp[:, 2],
                mode="markers",
                name="Puntos 3D (LightGlue)",
                marker=dict(
                    size=1.6,
                    color=[f"rgb({r},{g},{b})" for r, g, b in C[sel]],
                    opacity=0.7,
                ),
                hoverinfo="skip",
                # Con nube densa, la dispersa queda oculta (se activa en la leyenda).
                visible="legendonly" if dense is not None else True,
            ),
            row=1,
            col=1,
        )

    if planes:
        for tr in plane_meshes(planes):
            fig.add_trace(tr, row=1, col=1)

    xs, ys, zs = [], [], []
    every = max(1, len(steps) // 120)
    for s in steps[::every]:
        for a, b in frustum_lines(s.T_wc, fsize):
            xs += [a[0], b[0], None]
            ys += [a[1], b[1], None]
            zs += [a[2], b[2], None]
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name="Cámaras",
            line=dict(color="#8a8984", width=2),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    hover = [
        f"frame {s.idx}<br>t = {s.t_rel:.2f} s<br>{s.status}<br>inliers {s.inliers}/{s.matches}"
        for s in steps
    ]
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0],
            y=pos[:, 1],
            z=pos[:, 2],
            mode="lines+markers",
            name="Trayectoria",
            line=dict(color=t, colorscale=TIME_SCALE, width=5),
            marker=dict(
                size=2.5,
                color=t,
                colorscale=TIME_SCALE,
                colorbar=dict(title="t [s]", len=0.45, y=0.78, thickness=14),
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    lost = np.array([s.status.startswith("perdido") for s in steps])
    if lost.any():
        fig.add_trace(
            go.Scatter3d(
                x=pos[lost, 0],
                y=pos[lost, 1],
                z=pos[lost, 2],
                mode="markers",
                name="Tracking perdido",
                marker=dict(size=5, color=RED, symbol="x"),
                text=[h for h, l in zip(hover, lost) if l],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    for i, label, color in ((0, "Inicio", AQUA), (-1, "Fin", ORANGE)):
        fig.add_trace(
            go.Scatter3d(
                x=[pos[i, 0]],
                y=[pos[i, 1]],
                z=[pos[i, 2]],
                mode="markers+text",
                name=label,
                text=[label],
                textposition="top center",
                showlegend=False,
                marker=dict(size=7, color=color, line=dict(color=SURFACE, width=2)),
            ),
            row=1,
            col=1,
        )

    for k, (label, color) in enumerate(
        (("x (derecha)", BLUE), ("y (adelante)", ORANGE), ("z (arriba)", AQUA))
    ):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=pos[:, k],
                name=label,
                line=dict(color=color, width=2),
                legendgroup="pos",
            ),
            row=2,
            col=1,
        )

    yaw_vo = np.degrees(np.unwrap(np.radians(ang[:, 0])))
    fig.add_trace(
        go.Scatter(
            x=t,
            y=yaw_vo - yaw_vo[0],
            name="Yaw VO",
            line=dict(color=BLUE, width=2),
            legendgroup="yaw",
        ),
        row=2,
        col=2,
    )
    if estado is not None and "yaw" in estado:
        te = estado["t_host"] - t0_host
        win = (te >= t[0]) & (te <= t[-1])
        if win.any():
            yi = np.degrees(np.unwrap(np.radians(estado["yaw"][win])))
            y0 = np.interp(t[0], te[win], yi)
            fig.add_trace(
                go.Scatter(
                    x=te[win],
                    y=yi - y0,
                    name="Yaw IMU Tello",
                    line=dict(color=ORANGE, width=2),
                    legendgroup="yaw",
                ),
                row=2,
                col=2,
            )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=ang[:, 1],
            name="Pitch VO",
            line=dict(color=BLUE, width=2),
            legendgroup="pr",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=ang[:, 2],
            name="Roll VO",
            line=dict(color=ORANGE, width=2),
            legendgroup="pr",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=[s.matches for s in steps],
            name="Matches LightGlue",
            line=dict(color=BLUE, width=2),
            legendgroup="m",
        ),
        row=3,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=[s.inliers for s in steps],
            name="Inliers esencial",
            line=dict(color=ORANGE, width=2),
            legendgroup="m",
        ),
        row=3,
        col=2,
    )

    axis = dict(
        gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TEXT2)
    )
    for r, c, ylab in (
        (2, 1, "m" if metric else "unidades VO"),
        (2, 2, "Δ yaw [°]"),
        (3, 1, "grados"),
        (3, 2, "nº puntos"),
    ):
        fig.update_xaxes(axis, title_text="t [s]", row=r, col=c)
        fig.update_yaxes(axis, title_text=ylab, row=r, col=c)
    fig.update_scenes(
        aspectmode="data",
        bgcolor=SURFACE,
        xaxis=dict(title="x", gridcolor=GRID, backgroundcolor=SURFACE),
        yaxis=dict(title="y", gridcolor=GRID, backgroundcolor=SURFACE),
        zaxis=dict(title="z", gridcolor=GRID, backgroundcolor=SURFACE),
        camera=dict(eye=dict(x=-1.3, y=-1.3, z=1.0)),
    )
    fig.update_layout(
        title=dict(
            text=f"VO SuperPoint + LightGlue · sesión {name} · {len(steps)} keyframes",
            x=0.01,
            font=dict(size=20),
        ),
        height=1700,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="Inter, Segoe UI, sans-serif", color="#0b0b0b"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=0.0),
        margin=dict(l=60, r=30, t=110, b=50),
    )
    fig.write_html(out / "trayectoria.html", include_plotlyjs="cdn")


# --------------------------------------------------------------------------
# Animación paso a paso (render propio con OpenCV, estilo visor de reconstrucción)
# --------------------------------------------------------------------------


def look_at(eye, target):
    f = target - eye
    f /= np.linalg.norm(f) + 1e-12
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, -u, f])  # filas: x derecha, y abajo, z adelante


class Viewer3D:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.f = 0.9 * w
        self.target = None
        self.dist = None
        self.az = np.radians(-120)

    def project(self, P, R, eye):
        Pc = (P - eye) @ R.T
        z = Pc[:, 2]
        ok = z > 1e-6
        uv = np.full((len(P), 2), -1.0)
        uv[ok] = self.f * Pc[ok, :2] / z[ok, None] + [self.w / 2, self.h / 2]
        return uv, z, ok

    def render(self, P, Cl, traj, frustums, cur_frustum):
        img = np.full((self.h, self.w, 3), (22, 20, 18), np.uint8)
        lo, hi = traj.min(axis=0), traj.max(axis=0)
        extent = max(float(np.linalg.norm(hi - lo)), 1e-2)
        goal = 0.6 * (lo + hi) / 2 + 0.4 * traj[-1]
        goal_d = 1.3 * extent
        if self.target is None:
            self.target, self.dist = goal, max(goal_d, 0.5)
        self.target = 0.85 * self.target + 0.15 * goal
        self.dist = 0.92 * self.dist + 0.08 * max(goal_d, 0.5)
        self.az += np.radians(0.35)
        el = np.radians(32)
        eye = self.target + self.dist * np.array(
            [np.cos(el) * np.cos(self.az), np.cos(el) * np.sin(self.az), np.sin(el)]
        )
        R = look_at(eye, self.target)

        if len(P):
            uv, z, ok = self.project(P, R, eye)
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            ok &= (u >= 0) & (u < self.w - 1) & (v >= 0) & (v < self.h - 1)
            order = np.argsort(-z[ok])  # de lejos a cerca: lo cercano pisa
            uu, vv, cc = u[ok][order], v[ok][order], Cl[ok][order]
            # Puntos lejanos más tenues para dar profundidad y no tapar la trayectoria.
            zz = z[ok][order]
            fade = np.clip(1.4 - zz / (2.0 * self.dist), 0.35, 1.0)[:, None]
            img[vv, uu] = (cc * fade).astype(np.uint8)
            near = zz < self.dist
            img[vv[near], uu[near] + 1] = (cc[near] * fade[near]).astype(np.uint8)

        def seg(a, b, color, thick):
            uv, _, ok = self.project(np.array([a, b]), R, eye)
            if ok.all():
                p, q = tuple(np.round(uv[0]).astype(int)), tuple(
                    np.round(uv[1]).astype(int)
                )
                cv2.line(img, p, q, color, thick, cv2.LINE_AA)

        for fr in frustums:
            for a, b in fr:
                seg(a, b, (175, 172, 165), 1)
        n = len(traj)
        for i in range(1, n):
            k = i / max(n - 1, 1)
            color = tuple(
                int(c)
                for c in (
                    np.array([250, 214, 180]) * (1 - k) + np.array([229, 135, 57]) * k
                )
            )
            seg(
                traj[i - 1], traj[i], (10, 10, 10), 5
            )  # halo oscuro para separarla de la nube
            seg(traj[i - 1], traj[i], color, 3)
        for a, b in cur_frustum:
            seg(a, b, (52, 104, 235), 2)  # naranja (BGR)
        return img


def draw_matches(img, step, scale):
    out = img.copy()
    if len(step.uv_prev):
        p0 = step.uv_prev * scale
        p1 = step.uv_cur * scale
        for a, b, good in zip(p0, p1, step.inlier_mask):
            if not good:
                cv2.line(
                    out,
                    tuple(np.round(a).astype(int)),
                    tuple(np.round(b).astype(int)),
                    (60, 60, 150),
                    1,
                    cv2.LINE_AA,
                )
        for a, b, good in zip(p0, p1, step.inlier_mask):
            if good:
                a, b = tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int))
                cv2.line(out, a, b, (122, 175, 27), 1, cv2.LINE_AA)
                cv2.circle(out, b, 2, (122, 220, 60), -1, cv2.LINE_AA)
    return out


def put(img, text, org, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thick + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )


def render_animation(
    out: Path,
    vo: LightGlueVO,
    cloud,
    fps_out,
    gif_frames,
    gif_width,
    max_points=600000,
):
    """`cloud` = (P, C, owner): puntos en orden de creación y el paso que creó cada uno."""
    steps = vo.steps
    pw, ph, header = 640, 480, 44
    W, H = 2 * pw, ph + header
    viewer = Viewer3D(pw, ph)
    pos = np.array([CAM_TO_PLOT @ s.T_wc[:3, 3] for s in steps])
    steplen = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    fsize = (
        max(float(np.median(steplen[steplen > 0])) * 2.0, 1e-3)
        if (steplen > 0).any()
        else 0.05
    )

    P, C, owner = cloud
    if len(P):
        keep = near_trajectory(P, steps)
        if keep.sum() > max_points:  # submuestreo uniforme conservando el orden
            idx = np.flatnonzero(keep)
            keep[:] = False
            keep[idx[np.linspace(0, len(idx) - 1, max_points).astype(int)]] = True
        P, C, owner = P[keep], C[keep], owner[keep]
    Pp = (CAM_TO_PLOT @ P.T).T if len(P) else P
    Cbgr = (
        np.clip(C[:, ::-1].astype(float) * 1.15 + 15, 0, 255).astype(np.uint8)
        if len(C)
        else C
    )

    # H.264 vía PyAV: el mp4v de OpenCV no se reproduce en navegadores ni en VS Code.
    container = av.open(str(out / "reconstruccion.mp4"), mode="w")
    stream = container.add_stream(
        "libx264", rate=Fraction(fps_out).limit_denominator(1000)
    )
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": "23", "preset": "medium"}
    gif_pick = set(
        np.linspace(0, len(steps) - 1, min(gif_frames, len(steps))).astype(int).tolist()
    )
    gif = []
    every = max(1, len(steps) // 60)
    status_color = {
        "ok": (122, 220, 60),
        "rotacion": (0, 190, 240),
        "perdido": (72, 73, 227),
        "perdido_imu": (72, 73, 227),
        "inicio": (235, 235, 235),
    }

    for i, s in enumerate(steps):
        frame = cv2.imdecode(np.frombuffer(s.jpg, np.uint8), cv2.IMREAD_COLOR)
        left = draw_matches(frame, s, pw / vo.kf.image.shape[1])
        n_pts = int(np.searchsorted(owner, i, side="right"))
        frs = [frustum_lines(steps[j].T_wc, fsize) for j in range(0, i, every)]
        right = viewer.render(
            Pp[:n_pts],
            Cbgr[:n_pts],
            pos[: i + 1],
            frs,
            frustum_lines(s.T_wc, fsize * 1.4),
        )
        canvas = np.full((H, W, 3), (18, 16, 14), np.uint8)
        canvas[header:, :pw] = left
        canvas[header:, pw:] = right
        cv2.line(canvas, (pw, header), (pw, H), (60, 60, 60), 1)
        put(canvas, f"frame {s.idx}   t = {s.t_rel:.2f} s", (14, 29), 0.62)
        put(canvas, f"inliers {s.inliers}/{s.matches}", (pw - 190, 29), 0.62)
        put(
            canvas,
            s.status.upper(),
            (pw + 14, 29),
            0.62,
            status_color.get(s.status, (235, 235, 235)),
            2,
        )
        put(
            canvas,
            f"keyframe {i + 1}/{len(steps)}   puntos {n_pts}",
            (W - 360, 29),
            0.62,
        )
        put(canvas, "SuperPoint + LightGlue", (14, H - 14), 0.5)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            container.mux(packet)
        if i in gif_pick:
            g = cv2.resize(
                rgb, (gif_width, int(H * gif_width / W)), interpolation=cv2.INTER_AREA
            )
            gif.append(Image.fromarray(g))
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    if gif:
        # Paleta común a todos los fotogramas: evita parpadeos y restos entre frames.
        sample = [gif[k] for k in np.linspace(0, len(gif) - 1, 8).astype(int)]
        mosaic = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for k, g in enumerate(sample):
            mosaic.paste(g, (0, k * g.height))
        pal = mosaic.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
        gif = [g.quantize(palette=pal, dither=Image.Dither.NONE) for g in gif]
        gif[0].save(
            out / "reconstruccion.gif",
            save_all=True,
            append_images=gif[1:],
            duration=int(1000 / fps_out),
            loop=0,
            disposal=1,
        )


# --------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "entrada", type=Path, help="Carpeta de sesión (con video.mp4) o un MP4"
    )
    p.add_argument("--start", type=int, default=0, help="Primer fotograma")
    p.add_argument("--end", type=int, default=None, help="Último fotograma (exclusivo)")
    p.add_argument(
        "--step", type=int, default=1, help="Procesar 1 de cada N fotogramas"
    )
    p.add_argument("--max-kpts", type=int, default=2048)
    p.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Lado largo para SuperPoint (None = nativo)",
    )
    p.add_argument(
        "--min-flow",
        type=float,
        default=10.0,
        help="Flujo mediano [px] para probar keyframe",
    )
    p.add_argument(
        "--max-gap", type=int, default=20, help="Máx. fotogramas entre keyframes"
    )
    p.add_argument("--min-inliers", type=int, default=40)
    p.add_argument(
        "--min-parallax", type=float, default=1.0, help="[°] por debajo = rotación pura"
    )
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps-anim", type=float, default=12.0)
    p.add_argument("--gif-frames", type=int, default=160)
    p.add_argument("--gif-width", type=int, default=760)
    p.add_argument("--cpu", action="store_true", help="Forzar CPU")
    p.add_argument(
        "--sin-imu",
        action="store_true",
        help="No fusionar la actitud de estado.csv (solo VO)",
    )
    p.add_argument(
        "--imu-max-err",
        type=float,
        default=4.0,
        help="[°] discrepancia VO-IMU por paso a partir de la cual manda la IMU",
    )
    p.add_argument(
        "--zancada",
        type=float,
        default=0.65,
        help="Metros por paso del podómetro (calíbralo andando una distancia conocida)",
    )
    p.add_argument(
        "--sin-andar-horizontal",
        action="store_true",
        help="Permitir cambios de altura al andar (p. ej. escaleras); por defecto,"
        " mientras se anda la traslación se restringe al plano horizontal",
    )
    p.add_argument(
        "--sin-podometro",
        action="store_true",
        help="No usar el podómetro de la IMU para la escala (escala arbitraria)",
    )
    p.add_argument(
        "--imu-alpha",
        type=float,
        default=0.05,
        help="Fracción por keyframe con la que la orientación se acerca a la IMU",
    )
    p.add_argument(
        "--denso",
        action="store_true",
        help="Densificar keyframes con Depth Anything V2 alineado a la escala de la VO",
    )
    p.add_argument(
        "--denso-cada", type=int, default=4, help="Densificar 1 de cada N keyframes"
    )
    p.add_argument(
        "--denso-paso", type=int, default=8, help="Muestrear 1 píxel cada X en u y v"
    )
    p.add_argument(
        "--planos",
        action="store_true",
        help="Detectar planos, asociarlos entre tomas y ajustar a ellos la nube densa"
        " (implica --denso)",
    )
    p.add_argument(
        "--voxel",
        type=float,
        default=None,
        help="Tamaño de vóxel de la fusión (por defecto: separación entre muestras)",
    )
    args = p.parse_args()
    args.denso = args.denso or args.planos

    video, stamps, estado, name = resolve_inputs(args.entrada)
    out = args.out or HERE / "resultados_vo_lightglue" / name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Dispositivo: {device}  vídeo: {video}")

    t0_host = stamps[0] if stamps is not None else 0.0
    imu_att = None
    if estado is not None and stamps is not None and not args.sin_imu:
        imu_att = imu_attitude_fn(estado, t0_host)
    podometro = None
    if imu_att is not None and not args.sin_podometro and "agz" in estado:
        podometro = crear_podometro(estado, t0_host, args.zancada)
        print(
            f"Podómetro: {len(podometro.t_pasos)} pasos -> {podometro.total:.1f} m "
            f"(zancada {args.zancada} m)"
        )
    densificador = Densificador(device, args.denso_paso) if args.denso else None
    vo = LightGlueVO(args, device, imu_att, densificador, podometro)
    tic = time.time()
    n = 0
    for idx, fps, frame in iter_video(video, args.start, args.end, args.step):
        if stamps is not None and idx < len(stamps):
            t_rel = float(stamps[idx] - t0_host)
        else:
            t_rel = idx / fps
        vo.process(idx, t_rel, frame)
        n += 1
        if n % 200 == 0:
            print(
                f"  fotograma {idx}: {len(vo.steps)} keyframes, {time.time() - tic:.0f} s"
            )
    if len(vo.steps) < 2:
        raise SystemExit("No se ha podido estimar ninguna pose: revisa el intervalo")

    P, C = vo.cloud()
    dense = vo.dense_cloud(args.voxel)
    estados = [s.status for s in vo.steps]
    print(
        f"Fotogramas procesados: {n}  keyframes: {len(vo.steps)}  puntos 3D: {len(P)}  "
        f"({time.time() - tic:.0f} s)"
    )
    print("Estados:", {k: estados.count(k) for k in sorted(set(estados))})
    if imu_att is not None:
        print(
            f"Saltos de rotación corregidos con la IMU: {sum(s.rot_imu for s in vo.steps)}"
        )
    pos = np.array([s.T_wc[:3, 3] for s in vo.steps])
    recorrido = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    print(
        f"Recorrido de la trayectoria: {recorrido:.1f} "
        + ("m" if podometro is not None else "unidades VO")
    )
    if dense is not None:
        n_kf_dense = sum(1 for s in vo.steps if s.dense_points)
        n_raw = sum(len(d) for d in vo.dense_P)
        print(
            f"Nube densa: {n_kf_dense} keyframes, {n_raw} muestras -> "
            f"{len(dense[0])} tras fusionar por vóxeles"
        )

    planes = None
    if vo.mapa_planos is not None and dense is not None:
        planes = plane_summary(vo.mapa_planos, dense)
        save_planes(out, planes, dense)
        tipos = [pl["tipo"] for pl in planes]
        print(
            f"Planos: {len(planes)} en el mapa "
            f"({sum(pl['tomas'] > 1 for pl in planes)} vistos en varias tomas); "
            + ", ".join(f"{k}: {tipos.count(k)}" for k in PLANE_COLORS if k in tipos)
        )

    save_tables(out, vo.steps, P, C, dense)
    plotly_report(
        out, vo.steps, P, C, estado, t0_host, name, dense, planes, podometro is not None
    )
    print("Renderizando animación paso a paso...")
    cloud = dense[:3] if dense is not None else vo.sparse_cloud_with_owner()
    render_animation(out, vo, cloud, args.fps_anim, args.gif_frames, args.gif_width)
    info = {
        "sesion": name,
        "video": str(video),
        "dispositivo": str(device),
        "K": vo.K.tolist(),
        "fotogramas_procesados": n,
        "keyframes": len(vo.steps),
        "puntos_3d": int(len(P)),
        "puntos_densos": int(len(dense[0])) if dense is not None else 0,
        "planos": len(planes) if planes is not None else 0,
        "saltos_rotacion_imu": int(sum(s.rot_imu for s in vo.steps)),
        "estados": {k: estados.count(k) for k in set(estados)},
        "parametros": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "podometro": (
            {"pasos": len(podometro.t_pasos), "zancada_m": args.zancada,
             "distancia_m": round(podometro.total, 2)}
            if podometro is not None else None
        ),
        "recorrido": round(recorrido, 3),
        "aviso": (
            "Escala en metros vía podómetro: depende de --zancada (error típico 10-20 %)."
            if podometro is not None
            else "Escala métrica no observable (monocular); orientación afectada por posible EIS."
        ),
    }
    (out / "informe.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Resultados en {out}")


if __name__ == "__main__":
    main()
