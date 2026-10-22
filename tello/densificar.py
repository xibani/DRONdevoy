"""Densificación de keyframes con Depth Anything V2 alineado a la escala de la VO.

Depth Anything V2 (modelos "relative") devuelve una disparidad relativa d,
proporcional a la inversa de la profundidad: d ≈ a / z. Los puntos triangulados
por la VO en ese keyframe dan pares (d, z) con los que se estima a; con ella
cada píxel muestreado cada --denso-paso píxeles recibe una profundidad en las
mismas unidades (arbitrarias) que la trayectoria.

Solo se ajusta la escala (no el desplazamiento b de un ajuste afín): las
profundidades de la VO son muy ruidosas en paredes poco texturadas y con poco
paralaje, y un ajuste de dos parámetros llega a invertir la pendiente. La
mediana de d·z, en cambio, aguanta ese ruido; la forma la pone la red.

Dependencias extra: uv pip install transformers
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODELO = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass
class Ajuste:
    a: float
    err_rel: float  # mediana de |z_red - z_vo| / z_vo
    n: int  # puntos de la VO usados en el ajuste


def ajustar_escala(d, z, min_puntos=30):
    """Escala robusta d ≈ a / z: a = mediana(d·z) sobre disparidades positivas."""
    ok = (d > 1e-3) & (z > 0)
    if ok.sum() < min_puntos:
        return None
    a = float(np.median(d[ok] * z[ok]))
    err = float(np.median(np.abs(a / d[ok] - z[ok]) / z[ok]))
    return Ajuste(a, err, int(ok.sum()))


class Densificador:
    def __init__(self, device, paso=8, modelo=MODELO, max_err_rel=0.3, max_salto=0.04):
        self.device = device
        self.paso = paso
        self.max_err_rel = max_err_rel
        self.max_salto = max_salto  # salto de log(z) por píxel que se considera borde
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(modelo)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(modelo, dtype=dtype)
            .to(device)
            .eval()
        )
        self.dtype = dtype

    @torch.inference_mode()
    def disparidad(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        pred = self.model(pixel_values=inputs["pixel_values"].to(self.dtype)).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred[:, None].float(), size=bgr.shape[:2], mode="bicubic", align_corners=False
        )
        return pred[0, 0].cpu().numpy()

    def profundidad(self, bgr, uv, z):
        """Mapa de profundidad denso (HxW, nan donde no es fiable) y el ajuste usado."""
        d = self.disparidad(bgr)
        h, w = d.shape
        px = np.clip(np.round(uv).astype(int), 0, [w - 1, h - 1])
        ajuste = ajustar_escala(d[px[:, 1], px[:, 0]], z)
        if ajuste is None or ajuste.err_rel > self.max_err_rel:
            return None, ajuste
        zmap = np.full_like(d, np.nan)
        zmax = 2.5 * np.percentile(z, 95)
        zmin = 0.3 * np.percentile(z, 5)
        ok = d > ajuste.a / zmax
        zmap[ok] = ajuste.a / d[ok]
        zmap[(zmap < zmin) | (zmap > zmax)] = np.nan
        # Bordes de objetos: la red suaviza los saltos y genera "píxeles volantes".
        logz = np.log(np.where(np.isnan(zmap), zmax, zmap)).astype(np.float32)
        gx = cv2.Sobel(logz, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(logz, cv2.CV_32F, 0, 1, ksize=3) / 8
        zmap[np.hypot(gx, gy) > self.max_salto] = np.nan
        return zmap, ajuste

    def rejilla(self, bgr, K, uv, z):
        """Rejilla (gh, gw, 3) de puntos en marco cámara muestreando cada `paso` píxeles.

        Devuelve (X, C, ajuste): X con nan donde no hay profundidad fiable y C los
        colores RGB de esos píxeles. X es None si el ajuste de escala no es fiable.
        """
        zmap, ajuste = self.profundidad(bgr, uv, z)
        if zmap is None:
            return None, None, ajuste
        h, w = zmap.shape
        p = self.paso
        vv, uu = np.mgrid[p // 2 : h : p, p // 2 : w : p]
        zz = zmap[vv, uu]
        rays = np.linalg.inv(K) @ np.stack([uu, vv, np.ones_like(uu)]).reshape(3, -1)
        X = (rays * zz.ravel()).T.reshape(*uu.shape, 3)
        return X, bgr[vv, uu][..., ::-1].copy(), ajuste


def fusionar_voxeles(P, voxel, *por_punto):
    """Un punto por vóxel: se queda el primero en aparecer (los arrays van en orden temporal).

    `por_punto` son arrays alineados con P (colores, paso de origen...) que se filtran igual.
    """
    if len(P) == 0:
        return (P, *por_punto)
    keys = np.floor(P / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    return (P[first], *(a[first] for a in por_punto))
