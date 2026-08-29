"""
atalaya.datasets.base — el contrato que cumple TODO adaptador.

Una `Secuencia` normaliza cualquier origen de datos (EuRoC, ArduPilot, CSV
genérico) a:

  * t_imu (N,), gyro (N,3) [rad/s], accel (N,3) [m/s²], en el frame body
  * cam: t_cam (M,) y rutas (M,) de imágenes en disco
  * gt opcional: t_gt, p (K,3), R (K,3,3), v (K,3) — SOLO para diagnóstico
  * camara: CamaraPinhole; imu_params: ParamsImu

Convenios: mundo z-ARRIBA (g = (0,0,-9.81)); tiempos en segundos float64 con el
offset de la secuencia restado (t0_ns guarda ese offset). El bug silencioso más
común del curso es no restar el offset: aquí lo resta el constructor, siempre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from ..sensores import CamaraPinhole, ParamsImu


@dataclass
class Secuencia:
    nombre: str
    t_imu: np.ndarray
    gyro: np.ndarray
    accel: np.ndarray
    t_cam: np.ndarray
    rutas: list
    camara: CamaraPinhole
    imu_params: ParamsImu
    t0_ns: int = 0
    # GT opcional (mundo z-arriba, R_ws body->mundo)
    t_gt: Optional[np.ndarray] = None
    gt_p: Optional[np.ndarray] = None
    gt_R: Optional[np.ndarray] = None
    gt_v: Optional[np.ndarray] = None
    _slerp: object = field(default=None, repr=False)
    _IMU: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self._IMU = np.column_stack([self.t_imu, self.gyro, self.accel])

    # -- imágenes -----------------------------------------------------------
    def leer_imagen(self, k: int) -> np.ndarray:
        img = cv2.imread(str(self.rutas[k]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.rutas[k])
        return img

    # -- IMU entre dos instantes -------------------------------------------
    def imu_entre(self, ta: float, tb: float) -> np.ndarray:
        """(M,7) [t, w(3), a(3)] con muestras interpoladas exactamente en
        ta y tb, de modo que sum(dt) == tb - ta EXACTAMENTE. Sin esto, cada
        intervalo de cámara pierde hasta un dt de IMU y el filtro acumula un
        sesgo de velocidad que parece bias del acelerómetro (Fase 5)."""
        t = self.t_imu
        i0, i1 = np.searchsorted(t, [ta, tb], side="left")

        def en(x):
            j = int(np.clip(np.searchsorted(t, x), 1, len(t) - 1))
            t0, t1 = t[j - 1], t[j]
            w = 0.0 if t1 == t0 else (x - t0) / (t1 - t0)
            fila = (1 - w) * self._IMU[j - 1] + w * self._IMU[j]
            fila[0] = x
            return fila

        filas = [en(ta)]
        filas.extend(f for f in self._IMU[i0:i1] if ta < f[0] < tb)
        filas.append(en(tb))
        return np.array(filas)

    # -- ground truth -------------------------------------------------------
    @property
    def tiene_gt(self) -> bool:
        return self.t_gt is not None and len(self.t_gt) > 1

    def gt_en(self, t):
        """(p, R, v) interpolados; formas (N,3), (N,3,3), (N,3)."""
        assert self.tiene_gt, "esta secuencia no trae ground truth"
        if self._slerp is None:
            self._slerp = Slerp(self.t_gt, Rot.from_matrix(self.gt_R))
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        t = np.clip(t, self.t_gt[0], self.t_gt[-1])

        def interp(M):
            return np.column_stack([np.interp(t, self.t_gt, M[:, j]) for j in range(3)])

        v = interp(self.gt_v) if self.gt_v is not None else np.zeros((len(t), 3))
        return interp(self.gt_p), self._slerp(t).as_matrix(), v

    # -- detección de tramo estático ---------------------------------------
    def tramo_estatico(self, t_max: float = 10.0, ventana: float = 0.5,
                       umbral_gyro: float = 0.02, umbral_acc: float = 0.15):
        """Busca el tramo quieto inicial: devuelve (t_ini, t_fin) o None.

        Criterio: std del giro < umbral_gyro [rad/s] y std de ||a|| <
        umbral_acc [m/s²] en ventanas deslizantes desde el arranque."""
        m = self.t_imu <= self.t_imu[0] + t_max
        t, g, a = self.t_imu[m], self.gyro[m], self.accel[m]
        na = np.linalg.norm(a, axis=1)
        dt = np.median(np.diff(t))
        n_v = max(int(round(ventana / dt)), 4)
        fin = None
        for i in range(0, len(t) - n_v, n_v // 2):
            sl = slice(i, i + n_v)
            if g[sl].std(0).max() < umbral_gyro and na[sl].std() < umbral_acc:
                fin = t[sl.stop - 1]
            else:
                break
        if fin is None or fin - t[0] < ventana:
            return None
        return float(t[0]), float(fin)

    def __repr__(self):
        gt = f", GT {len(self.t_gt)}" if self.tiene_gt else ", sin GT"
        return (f"<Secuencia {self.nombre} | {len(self.t_cam)} imgs, "
                f"{len(self.t_imu)} imu ({1.0/np.median(np.diff(self.t_imu)):.0f} Hz)"
                f"{gt} | {self.t_cam[-1]-self.t_cam[0]:.1f} s>")
