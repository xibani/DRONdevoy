"""Odometría visual monocular con SuperPoint + LightGlue sobre el vídeo del Tello.

Dependencias (venv propio de tello/):
    uv venv .venv --python 3.11
    uv pip install torch --index-url https://download.pytorch.org/whl/cu126
    uv pip install "git+https://github.com/cvg/LightGlue.git" opencv-python numpy plotly pillow

Ejemplo:
    .venv/Scripts/python.exe VO_lightglue.py sesiones/20261017_113325
    .venv/Scripts/python.exe VO_lightglue.py sesiones/20261017_113325 --start 300 --end 900

Flujo por pares de keyframes:
  1. SuperPoint extrae puntos y descriptores; LightGlue los empareja.
  2. Un fotograma se convierte en keyframe cuando el flujo mediano supera
     --min-flow píxeles (o tras --max-gap fotogramas).
  3. Matriz esencial (USAC MAGSAC) + recoverPose -> R, t con |t| = 1.
  4. Triangulación de los inliers. La escala relativa entre pasos se obtiene
     comparando la profundidad de los puntos compartidos con el paso anterior,
     de modo que la trayectoria es coherente (aunque sigue sin escala métrica).
  5. Si el paralaje es casi nulo se trata como rotación pura (t = 0).

Salidas en resultados_vo_lightglue/<sesion>/:
    poses.csv, puntos_3d.ply, informe.json,
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

HERE = Path(__file__).resolve().parent

# Cámara (x derecha, y abajo, z adelante) -> mundo de visualización (x derecha,
# y adelante, z arriba), tomando como referencia la cámara del primer keyframe.
CAM_TO_PLOT = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


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


def yaw_rotation_cam(dyaw_deg):
    """Giro de yaw (horario visto desde arriba) expresado en el marco cámara-0."""
    a = np.radians(-dyaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    return CAM_TO_PLOT.T @ Rz @ CAM_TO_PLOT


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


class LightGlueVO:
    def __init__(self, args, device, imu_yaw=None):
        self.args = args
        self.device = device
        self.imu_yaw = imu_yaw  # t_rel -> yaw IMU desenrollado [°], o None
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
            self._new_keyframe(idx, t_rel, feats, kpts, bgr, np.eye(4), 1.0)
            self.steps.append(
                Step(idx, t_rel, np.eye(4), "inicio", jpg=self._encode(bgr))
            )
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

        # Paralaje tras compensar la rotación: separa traslación de rotación pura.
        r0, r1 = rays(K, uv0[inl]), rays(K, uv1[inl])
        cosang = np.sum(r1 * (R @ r0.T).T, axis=1)
        parallax = float(np.median(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
        if parallax < a.min_parallax and gap < a.max_gap and n_in >= 3 * a.min_inliers:
            return  # aún hay solape de sobra: esperamos a tener más línea base

        idx0, idx1 = m[inl, 0], m[inl, 1]
        depth_new = np.full(len(kpts), np.nan)
        new_pts = 0
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
            if common.sum() >= 12:
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
            )
        )
        self._new_keyframe(idx, t_rel, feats, kpts, bgr, T_wc, s, depth_new)

    def _lost(self, idx, t_rel, feats, kpts, bgr, n_matches, flow):
        """Sin geometría fiable: se reinicia en este fotograma conservando la posición.

        La orientación salta con el yaw de la IMU del Tello si está disponible
        (las pérdidas suelen ser giros rápidos con motion blur).
        """
        T_wc = self.kf.T_wc.copy()
        status = "perdido"
        if self.imu_yaw is not None:
            dyaw = self.imu_yaw(t_rel) - self.imu_yaw(self.kf.t_rel)
            T_wc[:3, :3] = yaw_rotation_cam(dyaw) @ T_wc[:3, :3]
            status = "perdido_imu"
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
            )
        )
        self._new_keyframe(idx, t_rel, feats, kpts, bgr, T_wc, self.kf.scale)

    def cloud(self):
        if not self.points:
            return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
        return np.vstack(self.points), np.vstack(self.colors)


# --------------------------------------------------------------------------
# Salidas en disco
# --------------------------------------------------------------------------


def save_tables(out: Path, steps, P, C):
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
                ]
            )
    Pp = (CAM_TO_PLOT @ P.T).T
    with (out / "puntos_3d.ply").open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(Pp)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(Pp, C):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


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


def plotly_report(out: Path, steps, P, C, estado, t0_host, name, max_points=40000):
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
            "Trayectoria y nube de puntos (escala arbitraria)",
            "Posición del dron",
            "Yaw: VO vs IMU del Tello",
            "Pitch y roll (VO)",
            "Correspondencias por keyframe",
        ),
    )

    if len(P):
        keep = near_trajectory(P, steps)
        P, C = P[keep], C[keep]
    if len(P):
        sel = np.linspace(0, len(P) - 1, min(max_points, len(P))).astype(int)
        Pp = (CAM_TO_PLOT @ P[sel].T).T
        fig.add_trace(
            go.Scatter3d(
                x=Pp[:, 0],
                y=Pp[:, 1],
                z=Pp[:, 2],
                mode="markers",
                name="Puntos 3D",
                marker=dict(
                    size=1.6,
                    color=[f"rgb({r},{g},{b})" for r, g, b in C[sel]],
                    opacity=0.7,
                ),
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

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
        (2, 1, "unidades VO"),
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


def render_animation(out: Path, vo: LightGlueVO, fps_out, gif_frames, gif_width):
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

    # Puntos acumulados en orden de creación, con su keyframe de origen.
    owners = []
    for i, s in enumerate(steps):
        if s.new_points:
            owners.append(np.full(s.new_points, i))
    owner = np.concatenate(owners) if owners else np.zeros(0, int)
    P, C = vo.cloud()
    if len(P):
        keep = near_trajectory(P, steps)
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
        help="No usar el yaw de estado.csv para cubrir las pérdidas de tracking",
    )
    args = p.parse_args()

    video, stamps, estado, name = resolve_inputs(args.entrada)
    out = args.out or HERE / "resultados_vo_lightglue" / name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Dispositivo: {device}  vídeo: {video}")

    t0_host = stamps[0] if stamps is not None else 0.0
    imu_yaw = None
    if estado is not None and stamps is not None and not args.sin_imu:
        te = estado["t_host"] - t0_host
        ye = np.degrees(np.unwrap(np.radians(estado["yaw"])))
        imu_yaw = lambda t: float(np.interp(t, te, ye))  # noqa: E731
    vo = LightGlueVO(args, device, imu_yaw)
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
    estados = [s.status for s in vo.steps]
    print(
        f"Fotogramas procesados: {n}  keyframes: {len(vo.steps)}  puntos 3D: {len(P)}  "
        f"({time.time() - tic:.0f} s)"
    )
    print("Estados:", {k: estados.count(k) for k in sorted(set(estados))})

    save_tables(out, vo.steps, P, C)
    plotly_report(out, vo.steps, P, C, estado, t0_host, name)
    print("Renderizando animación paso a paso...")
    render_animation(out, vo, args.fps_anim, args.gif_frames, args.gif_width)
    info = {
        "sesion": name,
        "video": str(video),
        "dispositivo": str(device),
        "K": vo.K.tolist(),
        "fotogramas_procesados": n,
        "keyframes": len(vo.steps),
        "puntos_3d": int(len(P)),
        "estados": {k: estados.count(k) for k in set(estados)},
        "parametros": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "aviso": "Escala métrica no observable (monocular); orientación afectada por posible EIS.",
    }
    (out / "informe.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Resultados en {out}")


if __name__ == "__main__":
    main()
