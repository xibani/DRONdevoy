"""
atalaya.time_offset — offset temporal cámara↔IMU por correlación de giro.

En un setup Pi + autopiloto (ATALAYA) cámara e IMU NO están hardware-
sincronizadas: el offset típico es de decenas de ms (anexo A.6) y se
manifiesta como "ATE razonable pero RPE horrible".

Método: la velocidad angular es observable por las dos vías sin escala ni
extrínseca de traslación:
  * visual: ||Log(R_rel)|| / dt entre frames consecutivos (matriz esencial)
  * inercial: ||w - b_g|| interpolada
Se barre el offset y se maximiza la correlación normalizada; el máximo se
refina con una parábola sobre tres puntos. Convención del resultado:

    t_frame_corregido = t_frame + offset

es decir, el offset devuelto es lo que hay que poner en
`dataset.camara.offset_temporal_s` del YAML.
"""
from __future__ import annotations

import numpy as np

from .frontend import ConfigFrontend, RastreadorKLT
from .geometria import Log


def _rotacion_kabsch(n0, n1):
    """Rotación R (c1<-c0) que mejor alinea los rayos normalizados.

    A diferencia de la matriz esencial, NO degenera con baseline pequeño
    (frames consecutivos): la traslación contamina un poco la magnitud, pero
    no desplaza el pico de correlación, que es lo único que importa aquí."""
    d0 = np.hstack([n0, np.ones((len(n0), 1))])
    d1 = np.hstack([n1, np.ones((len(n1), 1))])
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(d1.T @ d0)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return U @ S @ Vt


def velocidad_angular_visual(seq, k0, k1, salto=1, cfg=None, verbose=True):
    """(t_medios, omega_vis) con la velocidad angular entre frames k y k+salto."""
    c = cfg or ConfigFrontend()
    tr = RastreadorKLT(c)
    cam = seq.camara
    ts, om = [], []
    ids_a, _, pts_a = tr.rastrear(seq.leer_imagen(k0), k0)
    k_a = k0
    for k in range(k0 + 1, k1):
        ids_b, _, pts_b = tr.rastrear(seq.leer_imagen(k), k)
        if k - k_a < salto:
            continue
        comunes, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
        if len(comunes) >= 15:
            R = _rotacion_kabsch(cam.normalizar(pts_a[ia]),
                                 cam.normalizar(pts_b[ib]))
            dt = seq.t_cam[k] - seq.t_cam[k_a]
            if dt > 0:
                ts.append(0.5 * (seq.t_cam[k] + seq.t_cam[k_a]))
                # R_wc1 = R_wc0·Exp(ω dt)  =>  R_c1c0 = Exp(-ω dt)
                om.append(-Log(R) / dt)              # ω en frame CÁMARA (3,)
        ids_a, pts_a, k_a = ids_b, pts_b, k
    ts, om = np.array(ts), np.array(om).reshape(-1, 3)
    if verbose and len(ts):
        print(f"velocidad angular visual: {len(ts)} intervalos, "
              f"|ω| mediana {np.median(np.linalg.norm(om, axis=1)):.3f} rad/s")
    return ts, om


def estimar_offset(seq, k0=0, k1=None, bg=None, rango_s=0.5, paso_ms=2.0,
                   verbose=True):
    """Devuelve (offset_s, correlacion_max, barrido).

    Correlación 3D por ejes en el frame de cámara (el giro del IMU se lleva
    con la extrínseca de rotación). Necesita rotación de verdad en la
    ventana: si el vehículo va recto y nivelado, la correlación es plana y
    el resultado no es fiable (se avisa)."""
    k1 = k1 if k1 is not None else len(seq.t_cam)
    t_v, om_v = velocidad_angular_visual(seq, k0, k1, verbose=verbose)
    if len(t_v) < 20:
        raise RuntimeError("muy pocos intervalos visuales válidos para correlar")

    bg = np.zeros(3) if bg is None else np.asarray(bg, float)
    R_cb = seq.camara.T_cam_imu[:3, :3]
    om_imu_cam = (seq.gyro - bg) @ R_cb.T            # (N,3) en frame cámara

    # Paso-alto (sustraer media móvil de ~1 s): la contaminación por paralaje
    # de traslación es lenta (sigue a la velocidad) e inclina la curva de
    # correlación; el lag vive en el contenido de frecuencia alta del giro.
    dt_v = float(np.median(np.diff(t_v)))
    n_hp = max(int(round(1.0 / max(dt_v, 1e-3))) | 1, 5)
    ker = np.ones(n_hp) / n_hp

    def paso_alto(M):
        return M - np.column_stack([np.convolve(M[:, j], ker, mode="same")
                                    for j in range(3)])

    om_v = paso_alto(om_v)

    offsets = np.arange(-rango_s, rango_s + 1e-9, paso_ms * 1e-3)
    # Correlación de Pearson POR EJE y luego media: si un eje viene
    # contaminado por paralaje de traslación (varianza inflada), normalizarlo
    # por eje impide que domine el pico por pura magnitud.
    om_vc = om_v - om_v.mean(0)
    nv = np.linalg.norm(om_vc, axis=0)
    corr = np.full(len(offsets), np.nan)
    for i, off in enumerate(offsets):
        om_i = np.column_stack([np.interp(t_v + off, seq.t_imu, om_imu_cam[:, j])
                                for j in range(3)])
        om_i = paso_alto(om_i)
        om_ic = om_i - om_i.mean(0)
        ni = np.linalg.norm(om_ic, axis=0)
        den = nv * ni
        val = np.where(den > 0, (om_vc * om_ic).sum(0) / np.maximum(den, 1e-12),
                       np.nan)
        corr[i] = float(np.nanmean(val))

    j = int(np.nanargmax(corr))
    off = offsets[j]
    if 0 < j < len(offsets) - 1:                    # refinado parabólico
        y0, y1, y2 = corr[j - 1], corr[j], corr[j + 2 - 1]
        den = (y0 - 2 * y1 + y2)
        if abs(den) > 1e-12:
            off = offsets[j] + 0.5 * (y0 - y2) / den * (paso_ms * 1e-3)

    contraste = float(np.nanmax(corr) - np.nanmedian(corr))
    if verbose:
        print(f"offset estimado = {off*1e3:+.1f} ms   "
              f"(correlación {np.nanmax(corr):.3f}, contraste {contraste:.3f})")
        if contraste < 0.05:
            print("  AVISO: correlación casi plana — poca rotación en la "
                  "ventana. Usa un tramo con giros claros.")
        print("  -> pon este valor en dataset.camara.offset_temporal_s")
    return float(off), float(np.nanmax(corr)), np.column_stack([offsets, corr])
