"""
atalaya.geometria — SO(3)/SE(3), alineamiento y exportación.

Convenios (idénticos a anexo_convenios.md del curso):
  * T_A_B lleva puntos de B a A:  p_A = T_A_B @ p_B
  * Cuaterniones Hamilton; orden interno (x,y,z,w) como scipy; en disco (w,x,y,z)
  * Mundo z-arriba, G_W = (0, 0, -9.81)
  * Perturbación derecha/local: R = R̄ · Exp(δθ)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

G_MAG = 9.81
G_W = np.array([0.0, 0.0, -G_MAG])


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
    if abs(theta - np.pi) < 1e-5:
        rv = Rot.from_matrix(R).as_rotvec()
        return theta * rv / max(np.linalg.norm(rv), 1e-12)
    return theta / (2.0 * np.sin(theta)) * w


def right_jacobian(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).ravel()
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) - 0.5 * skew(phi)
    K = skew(phi / theta)
    return (np.eye(3)
            - (1 - np.cos(theta)) / theta * K
            + (theta - np.sin(theta)) / theta * (K @ K))


def jac_izq_inv(phi: np.ndarray) -> np.ndarray:
    """Jl^{-1}(φ). Jl(φ) = Jr(-φ)."""
    return np.linalg.inv(right_jacobian(-np.asarray(phi, dtype=float)))


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
    q = np.atleast_2d(np.asarray(q_wxyz, dtype=float))
    R = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    return R[0] if np.ndim(q_wxyz) == 1 else R


def R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = Rot.from_matrix(np.asarray(R)).as_quat()  # (x,y,z,w)
    q = np.atleast_2d(q)
    out = q[:, [3, 0, 1, 2]]
    return out[0] if np.asarray(R).ndim == 2 else out


# ---------------------------------------------------------------------------
# Evaluación / exportación
# ---------------------------------------------------------------------------
def align_umeyama(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    """Alinea Y sobre X (ambas (N,3)): X ~= s * R @ Y.T + t."""
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
    """ATE-RMSE de traslación tras alineamiento Umeyama.
    with_scale=True -> Sim(3);  False -> SE(3)."""
    s, R, t = align_umeyama(X, Y, with_scale)
    Y_al = (s * (R @ Y.T)).T + t
    return float(np.sqrt(((X - Y_al) ** 2).sum(1).mean()))


def save_tum(path, times: np.ndarray, poses: np.ndarray, t_offset_ns: int = 0) -> Path:
    """Trayectoria en formato TUM: t tx ty tz qx qy qz qw.
    poses: (N,4,4) T_w_body, o (N,7) [tx..tz, qx..qw]."""
    path = Path(path)
    poses = np.asarray(poses)
    if poses.ndim == 3:
        t = poses[:, :3, 3]
        q = Rot.from_matrix(poses[:, :3, :3]).as_quat()
        rows = np.hstack([t, q])
    else:
        rows = poses
    ts = np.asarray(times, dtype=float) + t_offset_ns * 1e-9
    with open(path, "w") as f:
        for ti, r in zip(ts, rows):
            f.write(f"{ti:.9f} " + " ".join(f"{x:.9f}" for x in r) + "\n")
    return path
