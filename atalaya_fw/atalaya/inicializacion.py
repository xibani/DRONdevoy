"""
atalaya.inicializacion — estado inicial del filtro.

Dos modos:

* "estatica" (defecto para datos propios): detecta el tramo quieto inicial,
  estima b_g como la media del giro, y la actitud alineando el acelerómetro
  medio con la gravedad (roll/pitch observables; yaw = 0, no observable).
  p = 0, v = 0.
* "gt": toma (p, v, R) del GT en t0. No es trampa: en VIO monocular la
  posición y el yaw NO son observables, fijarlos solo elige el marco de
  referencia para poder evaluar en SE(3) sin alinear.

En AMBOS modos, b_a se inicializa a CERO. La Fase 3.9 demostró que el b_a
estimado en reposo es degenerado: sus componentes perpendiculares a la
gravedad son cero por construcción, no por medida.
"""
from __future__ import annotations

import numpy as np

from .geometria import Exp, G_MAG


def actitud_desde_gravedad(a_medio: np.ndarray) -> np.ndarray:
    """R_ws con roll/pitch tales que R_ws @ â = e3 (yaw = 0).

    Modelo A.5: en reposo a_body = R_ws^T·(0,0,g), luego el eje z del mundo
    visto en body es â = a_medio/||a_medio||."""
    a_hat = np.asarray(a_medio, float) / max(np.linalg.norm(a_medio), 1e-12)
    e3 = np.array([0.0, 0.0, 1.0])
    v = np.cross(a_hat, e3)
    s = np.linalg.norm(v)
    c = float(a_hat @ e3)
    if s < 1e-12:
        return np.eye(3) if c > 0 else Exp(np.array([np.pi, 0, 0]))
    return Exp(v / s * np.arctan2(s, c))


def estado_inicial(seq, t0: float, tipo: str = "estatica",
                   t_estatico_max: float = 10.0, verbose=True):
    """Devuelve (p0, v0, R0, bg0, ba0) y un dict de diagnóstico."""
    diag = {}
    if tipo == "gt":
        assert seq.tiene_gt, "inicializacion 'gt' pero la secuencia no trae GT"
        p, R, v = seq.gt_en(t0)
        return (p[0], v[0], R[0], np.zeros(3), np.zeros(3)), diag

    tramo = seq.tramo_estatico(t_max=t_estatico_max)
    if tramo is None:
        raise RuntimeError(
            "no encuentro tramo estático al inicio (¿el vuelo arranca ya en "
            "movimiento?). Deja el vehículo quieto unos segundos al principio, "
            "o usa inicializacion.tipo='gt' si tienes referencia.")
    ta, tb = tramo
    m = (seq.t_imu >= ta) & (seq.t_imu <= tb)
    bg0 = seq.gyro[m].mean(0)
    a_medio = seq.accel[m].mean(0)
    R0 = actitud_desde_gravedad(a_medio)

    diag = dict(t_estatico=(ta, tb), bg0=bg0, norma_a=float(np.linalg.norm(a_medio)))
    if verbose:
        res = R0 @ a_medio - np.array([0, 0, np.linalg.norm(a_medio)])
        print(f"tramo estático [{ta:.2f}, {tb:.2f}] s "
              f"({(tb-ta):.1f} s, {m.sum()} muestras)")
        print(f"  b_g = {bg0}  (|b_g| = {np.linalg.norm(bg0):.2e} rad/s)")
        print(f"  ||a|| = {np.linalg.norm(a_medio):.3f} m/s² "
              f"(esperado ≈ {G_MAG}); residuo de alineación = "
              f"{np.linalg.norm(res):.2e}")
        print("  b_a = 0 (el tramo estático es degenerado para b_a: Fase 3.9)")
    return (np.zeros(3), np.zeros(3), R0, bg0, np.zeros(3)), diag
