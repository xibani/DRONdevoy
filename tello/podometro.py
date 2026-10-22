"""Podómetro con la IMU del Tello para dar escala métrica a la VO llevando el dron en la mano.

Con el dron en la mano las velocidades de estado.csv (vgx/vgy/vgz) valen 0, y la
aceleración no se puede comparar con la de la VO: el bamboleo de cada paso
(~1.7 Hz) aparece claramente en el acelerómetro pero no en la trayectoria
visual (lo elimina en buena parte el estabilizador electrónico del vídeo). Lo
que sí es fiable es contar pasos:

  1. Aceleración vertical en el mundo = actitud · acelerómetro − gravedad.
  2. Filtro paso banda 1–3 Hz (cadencias de marcha).
  3. "Andando" donde la energía local (RMS en ventanas de ~1.5 s) supera un umbral.
  4. Cada ciclo (cruce por cero ascendente con histéresis) dentro de un tramo
     andando cuenta como un paso; distancia = pasos · zancada.

D(t) (metros acumulados) es continua y lineal entre pasos, así que la línea
base entre dos keyframes es D(t_j) − D(t_i).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 20.0  # Hz de la rejilla uniforme sobre la que se filtra (estado.csv llega a ~10 Hz)


def _rot(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def aceleracion_vertical(estado, t0_host):
    """(t, a_up) con a_up en m/s², hacia arriba y sin gravedad (convención NED del Tello)."""
    t = estado["t_host"] - t0_host
    yaw, pitch, roll = (np.unwrap(np.radians(estado[k])) for k in ("yaw", "pitch", "roll"))
    f_b = np.stack([estado["agx"], estado["agy"], estado["agz"]], 1) * 9.81e-3  # mg -> m/s²
    a_down = np.array(
        [(_rot(y, p, r) @ f)[2] for y, p, r, f in zip(yaw, pitch, roll, f_b)]
    )
    # Fuerza específica en reposo = −g en z (abajo); aceleración real = f + g.
    return t, -(a_down + 9.81)


def _paso_banda(x, lo, hi, fs):
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


@dataclass
class Podometro:
    t_pasos: np.ndarray  # instante de cada paso [s, t_rel]
    zancada: float  # m por paso
    t_grid: np.ndarray
    andando: np.ndarray  # máscara sobre t_grid
    a_banda: np.ndarray  # aceleración vertical filtrada (para gráficas)

    def distancia(self, t):
        """Metros acumulados hasta t (continua, lineal entre pasos)."""
        if len(self.t_pasos) == 0:
            return 0.0
        k = np.arange(1, len(self.t_pasos) + 1) * self.zancada
        return float(np.interp(t, self.t_pasos, k, left=0.0, right=k[-1]))

    @property
    def total(self):
        return len(self.t_pasos) * self.zancada


def crear_podometro(
    estado, t0_host, zancada=0.7, banda=(1.0, 3.0), rms_min=0.2, histeresis=0.08,
    ventana=1.5, dt_min=0.3, dt_max=1.0,
):
    t, a_up = aceleracion_vertical(estado, t0_host)
    tg = np.arange(t[0], t[-1], 1 / FS)
    ab = _paso_banda(np.interp(tg, t, a_up), *banda, FS)
    n = max(1, int(ventana * FS))
    rms = np.sqrt(np.convolve(ab**2, np.ones(n) / n, mode="same"))
    andando = rms > rms_min

    pasos = []
    armado = False  # histéresis: hay que bajar de −h antes de contar el siguiente cruce
    for i in range(1, len(ab)):
        if ab[i] < -histeresis:
            armado = True
        elif armado and ab[i] > histeresis and andando[i]:
            if not pasos or dt_min < tg[i] - pasos[-1]:
                pasos.append(tg[i])
            armado = False
    pasos = np.array(pasos)
    if len(pasos) > 1:
        # Fuera de cadencia de marcha (pasos aislados o encadenados demasiado rápido): fuera.
        dt = np.diff(pasos)
        ok = np.zeros(len(pasos), bool)
        ok[1:] |= (dt > dt_min) & (dt < dt_max)
        ok[:-1] |= (dt > dt_min) & (dt < dt_max)
        pasos = pasos[ok]
    return Podometro(pasos, zancada, tg, andando, ab)
