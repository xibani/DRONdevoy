"""Planos como referencia entre tomas (fase 2 de la densificación).

1. En cada keyframe densificado se buscan planos sobre la rejilla de puntos
   (marco cámara) con RANSAC secuencial que exige también normales coherentes;
   cada plano se queda con su mayor componente conectada en la rejilla.
2. Los planos se pasan al mundo y se asocian con los del mapa global si tienen
   normal parecida, distancia parecida y se solapan espacialmente; si no, se
   crean nuevos. Cada plano del mapa acumula los momentos de sus puntos, así
   que su ajuste mejora con cada toma.
3. Al final, los puntos de cada plano se proyectan sobre su plano definitivo:
   la misma pared vista desde varias tomas deja de ser un conjunto de láminas.

Con el mundo nivelado por la IMU (z hacia arriba tras CAM_TO_PLOT) los planos se
clasifican en horizontales y en paredes de las dos direcciones dominantes
(hipótesis "Manhattan world": las paredes de una casa son casi siempre
perpendiculares entre sí).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Detección en un keyframe
# --------------------------------------------------------------------------


def normales_rejilla(X):
    """Normales por diferencias centrales en la rejilla (gh, gw, 3); nan en bordes y huecos."""
    N = np.full_like(X, np.nan)
    dx = X[1:-1, 2:] - X[1:-1, :-2]
    dy = X[2:, 1:-1] - X[:-2, 1:-1]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    # Orientadas hacia la cámara (que está en el origen).
    flip = np.sum(n * X[1:-1, 1:-1], axis=-1) > 0
    n[flip] *= -1
    N[1:-1, 1:-1] = n
    return N


def _plano_por_svd(P):
    c = P.mean(axis=0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    return n, float(n @ c)


def detectar_planos(
    X, rel_dist=0.03, max_ang=20.0, min_frac=0.06, max_planos=6, iters=250, seed=0
):
    """RANSAC secuencial sobre la rejilla X (gh, gw, 3) en marco cámara.

    Devuelve etiquetas (gh, gw) con -1 = sin plano y la lista [(n, d)] con n·X = d.
    El umbral de distancia es relativo a la profundidad (el error de la red crece con z).
    """
    gh, gw, _ = X.shape
    N = normales_rejilla(X)
    valid = ~np.isnan(X).any(-1) & ~np.isnan(N).any(-1)
    labels = np.full((gh, gw), -1, int)
    planos = []
    flatX, flatN = X.reshape(-1, 3), N.reshape(-1, 3)
    thr = rel_dist * np.abs(flatX[:, 2])
    cos_max = np.cos(np.radians(max_ang))
    min_pts = int(min_frac * valid.sum())
    rng = np.random.default_rng(seed)
    libre = valid.ravel().copy()
    for k in range(max_planos):
        cand = np.flatnonzero(libre)
        if len(cand) < max(min_pts, 30):
            break
        best, best_n = None, 0
        for _ in range(iters):
            p = flatX[rng.choice(cand, 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            if np.linalg.norm(n) < 1e-12:
                continue
            n /= np.linalg.norm(n)
            d = n @ p[0]
            inl = libre & (np.abs(flatX @ n - d) < thr) & (np.abs(flatN @ n) > cos_max)
            c = int(inl.sum())
            if c > best_n:
                best, best_n = inl, c
        if best is None or best_n < min_pts:
            break
        # Mayor componente conectada: evita "planos" de puntos sueltos coplanares.
        m = best.reshape(gh, gw).astype(np.uint8)
        n_cc, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n_cc <= 1:
            break
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        inl = (cc == biggest).ravel()
        if inl.sum() < min_pts:
            libre &= ~best  # descartamos esos puntos como semilla y seguimos
            continue
        n, d = _plano_por_svd(flatX[inl])
        if d < 0:
            n, d = -n, -d
        planos.append((n, d))
        labels.ravel()[inl] = len(planos) - 1
        libre &= ~inl
    return labels, planos


# --------------------------------------------------------------------------
# Mapa global de planos
# --------------------------------------------------------------------------


@dataclass
class Plano:
    n: np.ndarray  # normal unitaria (mundo)
    d: float  # n·X = d
    s1: np.ndarray  # Σ x
    s2: np.ndarray  # Σ x xᵀ
    count: int
    keys: set = field(default_factory=set)  # vóxeles gruesos ocupados
    primer_paso: int = 0
    tomas: int = 1

    def reajustar(self):
        c = self.s1 / self.count
        cov = self.s2 / self.count - np.outer(c, c)
        n = np.linalg.eigh(cov)[1][:, 0]
        if n @ self.n < 0:
            n = -n
        self.n, self.d = n, float(n @ c)


def _claves(P, voxel):
    return set(map(tuple, np.floor(P / voxel).astype(np.int64)))


_VECINOS = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]


def _solape(keys_nuevas, keys_plano):
    """Fracción de vóxeles nuevos con algún vecino (26-conexo) ocupado por el plano."""
    hits = sum(
        any((a + i, b + j, c + k) in keys_plano for i, j, k in _VECINOS)
        for a, b, c in keys_nuevas
    )
    return hits / len(keys_nuevas)


class MapaPlanos:
    """Mapa global de planos.

    Asociación (calibrada con la sesión 20261017_113325): normal a menos de
    `max_ang`, distancia mediana de los puntos nuevos al plano existente menor
    que `dist_rel`·z y solape espacial ≥ `min_solape`. La distancia se mide punto
    a plano y no comparando los términos d: con normales que difieren unos grados,
    d cambia mucho aunque los planos coincidan donde están los puntos.
    """

    def __init__(self, max_ang=12.0, dist_rel=0.06, min_solape=0.3, voxel_rel=0.1):
        self.planos: list[Plano] = []
        self.cos_max = np.cos(np.radians(max_ang))
        self.dist_rel = dist_rel
        self.min_solape = min_solape
        self.voxel_rel = voxel_rel
        self.voxel = None  # fijo desde el primer keyframe: las claves deben ser comparables

    def integrar(self, P, labels, planos_cam, R_wc, t_wc, z_med, paso_idx):
        """Asocia los planos de un keyframe al mapa. P: puntos mundo; labels: plano local o -1.

        Devuelve, por punto, el id del plano global (-1 si no pertenece a ninguno).
        z_med fija las tolerancias: la escala de la VO es arbitraria.
        """
        dist_thr = self.dist_rel * z_med
        if self.voxel is None:
            self.voxel = self.voxel_rel * z_med
        ids = np.full(len(P), -1, int)
        for j, (n_c, d_c) in enumerate(planos_cam):
            sel = labels == j
            if not sel.any():
                continue
            n = R_wc @ n_c
            d = d_c + n @ t_wc
            Pj = P[sel]
            keys = _claves(Pj, self.voxel)
            match = None
            best = 0.0
            for g, pl in enumerate(self.planos):
                if abs(n @ pl.n) < self.cos_max:
                    continue
                if np.median(np.abs(Pj @ pl.n - pl.d)) > dist_thr:
                    continue
                solape = _solape(keys, pl.keys)
                if solape >= self.min_solape and solape > best:
                    match, best = g, solape
            if match is None:
                self.planos.append(
                    Plano(n, d, Pj.sum(0), Pj.T @ Pj, len(Pj), keys, paso_idx)
                )
                match = len(self.planos) - 1
            else:
                pl = self.planos[match]
                pl.s1 += Pj.sum(0)
                pl.s2 += Pj.T @ Pj
                pl.count += len(Pj)
                pl.keys |= keys
                pl.tomas += 1
                pl.reajustar()
            ids[sel] = match
        return ids

    def proyectar(self, P, ids):
        """Proyecta cada punto sobre su plano global definitivo."""
        Q = P.copy()
        for g, pl in enumerate(self.planos):
            m = ids == g
            if m.any():
                Q[m] -= np.outer(Q[m] @ pl.n - pl.d, pl.n)
        return Q

    def clasificar(self, up, min_puntos=0):
        """Tipo por plano: 'horizontal', 'pared A', 'pared B' u 'otro'.

        `up` es el vector vertical en coordenadas de mundo. Las dos familias de
        paredes salen de la dirección dominante del acimut de sus normales (mod 90°).
        """
        tipos = []
        ref = np.cross(up, [1.0, 0, 0])
        if np.linalg.norm(ref) < 1e-6:
            ref = np.cross(up, [0, 1.0, 0])
        e1 = ref / np.linalg.norm(ref)
        e2 = np.cross(up, e1)
        az, peso = [], []
        for pl in self.planos:
            vert = abs(pl.n @ up)
            if vert > np.cos(np.radians(20)):
                tipos.append("horizontal")
            elif vert < np.sin(np.radians(20)):
                tipos.append("pared")
                az.append(np.arctan2(pl.n @ e2, pl.n @ e1))
                peso.append(pl.count)
            else:
                tipos.append("otro")
        if az:
            # Media circular de 4·acimut: invariante a giros de 90° entre paredes.
            az, peso = np.array(az), np.array(peso, float)
            theta0 = np.angle(np.sum(peso * np.exp(4j * az))) / 4
            k = 0
            for i, tp in enumerate(tipos):
                if tp != "pared":
                    continue
                rel = (az[k] - theta0) % np.pi
                k += 1
                tipos[i] = "pared A" if min(rel, np.pi - rel) < np.pi / 4 else "pared B"
        return tipos
