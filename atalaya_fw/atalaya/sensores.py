"""
atalaya.sensores — cámara pinhole y parámetros de ruido del IMU.

La cámara sigue el enfoque (B) de la Fase 2 del curso: se rastrea sobre la
imagen CRUDA y solo se indistorsionan los keypoints. Nunca se rectifica la
imagen completa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometria import inv_T


@dataclass
class CamaraPinhole:
    fu: float
    fv: float
    cu: float
    cv_: float
    dist: np.ndarray                 # radtan: (k1,k2,p1,p2[,k3]); fisheye: (k1..k4)
    width: int
    height: int
    T_body_cam: np.ndarray           # p_body = T_body_cam @ p_cam  (== T_BS de EuRoC)
    modelo: str = "radtan"           # "radtan" | "fisheye"
    rate_hz: float = 20.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fu, 0.0, self.cu],
                         [0.0, self.fv, self.cv_],
                         [0.0, 0.0, 1.0]])

    @property
    def f_media(self) -> float:
        return 0.5 * (self.fu + self.fv)

    @property
    def T_cam_imu(self) -> np.ndarray:
        """Lleva puntos del frame IMU/body al frame cámara."""
        return inv_T(self.T_body_cam)

    def normalizar(self, uv: np.ndarray) -> np.ndarray:
        """(N,2) píxeles distorsionados -> (N,2) coords normalizadas ideales."""
        p = np.ascontiguousarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        if self.modelo == "fisheye":
            d = np.asarray(self.dist, dtype=np.float64).ravel()[:4].reshape(4, 1)
            out = cv2.fisheye.undistortPoints(p.astype(np.float32), self.K, d)
        else:
            out = cv2.undistortPoints(p, self.K, np.asarray(self.dist, dtype=np.float64))
        return out.reshape(-1, 2)


@dataclass
class ParamsImu:
    rate_hz: float = 200.0
    sigma_g: float = 1.6968e-4       # gyroscope_noise_density   [rad/s/sqrt(Hz)]
    sigma_a: float = 2.0e-3          # accelerometer_noise_density [m/s^2/sqrt(Hz)]
    sigma_bg: float = 1.9393e-5      # gyroscope_random_walk     [rad/s^2/sqrt(Hz)]
    sigma_ba: float = 3.0e-3         # accelerometer_random_walk [m/s^3/sqrt(Hz)]

    def como_tupla(self):
        return (self.sigma_g, self.sigma_a, self.sigma_bg, self.sigma_ba)


def camara_desde_config(c: dict) -> CamaraPinhole:
    """Construye la cámara desde el bloque `camara:` del YAML."""
    fu, fv, cu, cvv = [float(x) for x in c["intrinsecos"]]
    T = c.get("T_body_cam")
    T = np.eye(4) if T is None else np.array(T, dtype=float).reshape(4, 4)
    return CamaraPinhole(
        fu=fu, fv=fv, cu=cu, cv_=cvv,
        dist=np.array(c.get("distorsion", [0, 0, 0, 0]), dtype=float),
        width=int(c["resolucion"][0]), height=int(c["resolucion"][1]),
        T_body_cam=T,
        modelo=str(c.get("modelo", "radtan")),
        rate_hz=float(c.get("rate_hz", 20.0)),
    )
