"""
anexo_utils.py — utilidades comunes del curso de VIO sobre EuRoC.

Convenios (ver anexo_convenios.md):
  * T_A_B lleva puntos de B a A:  p_A = T_A_B @ p_B
  * Cuaterniones Hamilton, orden interno (x, y, z, w) como scipy
  * Mundo z-arriba, g_w = (0, 0, -9.81)
  * Todos los tiempos en segundos float64 con el offset de la secuencia restado

Uso:
    from anexo_utils import EurocSequence, Exp, Log, skew, save_tum
    seq = EurocSequence("~/datasets/euroc/MH_01_easy/mav0")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

G_MAG = 9.81
G_W = np.array([0.0, 0.0, -G_MAG])


# ---------------------------------------------------------------------------
# SO(3) / SE(3)
# ---------------------------------------------------------------------------
def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def Exp(phi: np.ndarray) -> np.ndarray:
    """Mapa exponencial de SO(3): vector de rotación (3,) -> R (3,3)."""
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + skew(phi)
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def Log(R: np.ndarray) -> np.ndarray:
    """Logaritmo de SO(3): R (3,3) -> vector de rotación (3,)."""
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-10:
        return 0.5 * w
    if abs(theta - np.pi) < 1e-5:          # caso degenerado theta ~ pi
        return theta * Rot.from_matrix(R).as_rotvec() / max(
            np.linalg.norm(Rot.from_matrix(R).as_rotvec()), 1e-12)
    return theta / (2.0 * np.sin(theta)) * w


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    """Jacobiano derecho de SO(3). Necesario en preintegración y en algunos ESKF."""
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * skew(phi)
    K = skew(phi / theta)
    return (np.eye(3)
            - (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / theta * (K @ K))


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def quat_wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    """(w,x,y,z) -> matriz de rotación. Acepta (4,) o (N,4)."""
    q = np.atleast_2d(np.asarray(q_wxyz, dtype=float))
    R = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    return R[0] if np.ndim(q_wxyz) == 1 else R


# ---------------------------------------------------------------------------
# Cámara
# ---------------------------------------------------------------------------
@dataclass
class PinholeCamera:
    fu: float
    fv: float
    cu: float
    cv_: float
    dist: np.ndarray          # (k1, k2, p1, p2)
    width: int
    height: int
    T_body_cam: np.ndarray    # T_BS del YAML: p_body = T_body_cam @ p_cam
    rate_hz: float = 20.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fu, 0.0, self.cu],
                         [0.0, self.fv, self.cv_],
                         [0.0, 0.0, 1.0]])

    @property
    def T_cam_imu(self) -> np.ndarray:
        """Lleva puntos del frame IMU/body al frame cámara."""
        return inv_T(self.T_body_cam)

    def undistort_maps(self):
        """Mapas para rectificar la imagen manteniendo los MISMOS intrínsecos K."""
        return cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.K, (self.width, self.height), cv2.CV_32FC1)

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """(N,2) píxeles distorsionados -> (N,2) píxeles ideales (mismo K)."""
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.undistortPoints(p, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def normalize(self, pts: np.ndarray) -> np.ndarray:
        """(N,2) píxeles distorsionados -> (N,2) coords normalizadas (x/z, y/z)."""
        p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.dist).reshape(-1, 2)


@dataclass
class ImuParams:
    rate_hz: float
    sigma_g: float      # gyroscope_noise_density      [rad/s/sqrt(Hz)]
    sigma_a: float      # accelerometer_noise_density   [m/s^2/sqrt(Hz)]
    sigma_bg: float     # gyroscope_random_walk         [rad/s^2/sqrt(Hz)]
    sigma_ba: float     # accelerometer_random_walk     [m/s^3/sqrt(Hz)]

    def discrete(self, dt: float):
        """Devuelve (sigma_g_d, sigma_a_d, sigma_bg_d, sigma_ba_d) discretos."""
        return (self.sigma_g / np.sqrt(dt),
                self.sigma_a / np.sqrt(dt),
                self.sigma_bg * np.sqrt(dt),
                self.sigma_ba * np.sqrt(dt))


# ---------------------------------------------------------------------------
# Secuencia EuRoC
# ---------------------------------------------------------------------------
class EurocSequence:
    """Carga una secuencia EuRoC en formato ASL.

    Atributos principales
    ---------------------
    imu : DataFrame  con columnas t, wx, wy, wz, ax, ay, az
    cam : DataFrame  con columnas t, filename, path
    gt  : DataFrame  con columnas t, px..pz, qw..qz, vx..vz, bgx..bgz, bax..baz
    cam0, cam1 : PinholeCamera
    imu_params : ImuParams
    t0_ns : offset restado a todos los tiempos
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        if (self.root / "mav0").exists():
            self.root = self.root / "mav0"

        imu = pd.read_csv(self.root / "imu0/data.csv")
        imu.columns = ["t_ns", "wx", "wy", "wz", "ax", "ay", "az"]

        cam = pd.read_csv(self.root / "cam0/data.csv")
        cam.columns = ["t_ns", "filename"]

        gt = pd.read_csv(self.root / "state_groundtruth_estimate0/data.csv")
        gt.columns = ["t_ns", "px", "py", "pz", "qw", "qx", "qy", "qz",
                      "vx", "vy", "vz", "bgx", "bgy", "bgz", "bax", "bay", "baz"]

        # Offset común: crítico para no perder resolución en float64
        self.t0_ns = int(min(imu.t_ns.iloc[0], cam.t_ns.iloc[0], gt.t_ns.iloc[0]))
        for df in (imu, cam, gt):
            df["t"] = (df.t_ns.astype(np.int64) - self.t0_ns).astype(np.float64) * 1e-9

        cam["path"] = cam.filename.apply(lambda f: str(self.root / "cam0/data" / f.strip()))

        self.imu, self.cam, self.gt = imu, cam, gt
        self.t_imu = imu.t.values
        self.t_cam = cam.t.values
        self.t_gt = gt.t.values

        self.cam0 = self._load_cam("cam0")
        self.cam1 = self._load_cam("cam1") if (self.root / "cam1/sensor.yaml").exists() else None
        self.imu_params = self._load_imu_params()

        self._gt_slerp = None

    # -- carga de YAML -------------------------------------------------------
    def _load_cam(self, name: str) -> PinholeCamera:
        with open(self.root / name / "sensor.yaml") as f:
            c = yaml.safe_load(f)
        fu, fv, cu, cvv = c["intrinsics"]
        return PinholeCamera(
            fu=fu, fv=fv, cu=cu, cv_=cvv,
            dist=np.array(c["distortion_coefficients"], dtype=float),
            width=int(c["resolution"][0]), height=int(c["resolution"][1]),
            T_body_cam=np.array(c["T_BS"]["data"], dtype=float).reshape(4, 4),
            rate_hz=float(c.get("rate_hz", 20)),
        )

    def _load_imu_params(self) -> ImuParams:
        with open(self.root / "imu0/sensor.yaml") as f:
            c = yaml.safe_load(f)
        return ImuParams(
            rate_hz=float(c.get("rate_hz", 200)),
            sigma_g=float(c["gyroscope_noise_density"]),
            sigma_a=float(c["accelerometer_noise_density"]),
            sigma_bg=float(c["gyroscope_random_walk"]),
            sigma_ba=float(c["accelerometer_random_walk"]),
        )

    # -- imágenes ------------------------------------------------------------
    def image(self, k: int, undistort: bool = False) -> np.ndarray:
        img = cv2.imread(self.cam.path.iloc[k], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.cam.path.iloc[k])
        if undistort:
            if not hasattr(self, "_maps"):
                self._maps = self.cam0.undistort_maps()
            img = cv2.remap(img, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
        return img

    # -- IMU entre dos instantes --------------------------------------------
    def imu_between(self, t0: float, t1: float, interp_edges: bool = True) -> np.ndarray:
        """Devuelve (M,7): [t, wx,wy,wz, ax,ay,az] con t0 <= t <= t1.

        Si interp_edges, inserta muestras interpoladas exactamente en t0 y t1,
        de modo que sum(dt) == t1 - t0 exactamente.
        """
        data = self.imu[["t", "wx", "wy", "wz", "ax", "ay", "az"]].values
        i0, i1 = np.searchsorted(self.t_imu, [t0, t1], side="left")
        chunk = data[i0:i1]
        if not interp_edges:
            return chunk

        def interp_at(t):
            j = np.clip(np.searchsorted(self.t_imu, t), 1, len(self.t_imu) - 1)
            ta, tb = self.t_imu[j - 1], self.t_imu[j]
            w = 0.0 if tb == ta else (t - ta) / (tb - ta)
            row = (1 - w) * data[j - 1] + w * data[j]
            row[0] = t
            return row

        rows = [interp_at(t0)]
        rows += [r for r in chunk if t0 < r[0] < t1]
        rows.append(interp_at(t1))
        return np.array(rows)

    # -- ground truth --------------------------------------------------------
    def gt_at(self, t: float | np.ndarray):
        """Interpola el GT: devuelve (p (3,), R (3,3), v (3,), bg (3,), ba (3,))
        para un t escalar. Rotación con SLERP, el resto lineal."""
        if self._gt_slerp is None:
            q = self.gt[["qx", "qy", "qz", "qw"]].values
            self._gt_slerp = Slerp(self.t_gt, Rot.from_quat(q))
        t = float(np.clip(t, self.t_gt[0], self.t_gt[-1]))
        p = np.array([np.interp(t, self.t_gt, self.gt[c].values) for c in ("px", "py", "pz")])
        v = np.array([np.interp(t, self.t_gt, self.gt[c].values) for c in ("vx", "vy", "vz")])
        bg = np.array([np.interp(t, self.t_gt, self.gt[c].values) for c in ("bgx", "bgy", "bgz")])
        ba = np.array([np.interp(t, self.t_gt, self.gt[c].values) for c in ("bax", "bay", "baz")])
        return p, self._gt_slerp(t).as_matrix(), v, bg, ba

    def gt_T_ws(self, t: float) -> np.ndarray:
        p, R, *_ = self.gt_at(t)
        return make_T(R, p)

    def __repr__(self):
        return (f"<EurocSequence {self.root.parent.name} | "
                f"{len(self.cam)} imgs, {len(self.imu)} imu, "
                f"{self.t_cam[-1]-self.t_cam[0]:.1f} s>")


# ---------------------------------------------------------------------------
# Exportación / evaluación
# ---------------------------------------------------------------------------
def save_tum(path: str | Path, times: np.ndarray, poses: np.ndarray,
             t_offset_ns: int = 0) -> Path:
    """Guarda trayectoria en formato TUM: t tx ty tz qx qy qz qw.

    poses: (N,4,4) matrices T_w_body, o (N,7) [tx..tz,qx..qw].
    """
    path = Path(path)
    poses = np.asarray(poses)
    if poses.ndim == 3:
        R = poses[:, :3, :3]
        t = poses[:, :3, 3]
        q = Rot.from_matrix(R).as_quat()          # (x,y,z,w)
        rows = np.hstack([t, q])
    else:
        rows = poses
    ts = np.asarray(times, dtype=float) + t_offset_ns * 1e-9
    with open(path, "w") as f:
        for ti, r in zip(ts, rows):
            f.write(f"{ti:.9f} " + " ".join(f"{x:.9f}" for x in r) + "\n")
    return path


def euroc_gt_to_tum(seq: EurocSequence, path: str | Path,
                    t_lo: float = None, t_hi: float = None) -> Path:
    """Exporta el GT de EuRoC a formato TUM con la MISMA base de tiempo
    que uses en tus estimaciones."""
    m = np.ones(len(seq.gt), dtype=bool)
    if t_lo is not None:
        m &= seq.t_gt >= t_lo
    if t_hi is not None:
        m &= seq.t_gt <= t_hi
    g = seq.gt[m]
    rows = np.hstack([g[["px", "py", "pz"]].values,
                      g[["qx", "qy", "qz", "qw"]].values])
    return save_tum(path, g.t.values, rows)


def align_umeyama(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    """Alinea Y sobre X (ambas (N,3)). Devuelve (s, R, t) tal que
    X ~= s * R @ Y.T + t. Útil para depurar antes de pasar a `evo`."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    C = Xc.T @ Yc / len(X)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (np.trace(np.diag(D) @ S) / (Yc ** 2).sum(1).mean()) if with_scale else 1.0
    t = mx - s * R @ my
    return s, R, t


def ate_rmse(X: np.ndarray, Y: np.ndarray, with_scale: bool = True) -> float:
    """ATE-RMSE rápido (solo traslación) tras alineamiento Umeyama."""
    s, R, t = align_umeyama(X, Y, with_scale)
    Y_al = (s * (R @ Y.T)).T + t
    return float(np.sqrt(((X - Y_al) ** 2).sum(1).mean()))
