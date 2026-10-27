#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vio/integrador_giro.py — Orientación integrando solo el giroscopio.

Modelo de medida del giroscopio (el mismo que usará el MSCKF):
    ω_m = ω + b_g + n_g
  ω_m: lo que mide el sensor, en ejes de la IMU [rad/s]
  ω:   velocidad angular real del cuerpo respecto al mundo, en ejes de la IMU
  b_g: sesgo (offset que varía lentamente); n_g: ruido blanco

Cinemática de la orientación (ω en ejes del cuerpo -> se multiplica por la derecha):
    dR_w_i/dt = R_w_i · [ω]×
Discretizada suponiendo ω constante en [t_k, t_k+1]:
    R_w_i(t_k+1) = R_w_i(t_k) · Exp(ω̄ · Δt)
Con ω̄ = media de las dos muestras que delimitan el intervalo (regla del punto
medio): error de truncamiento O(Δt³) por paso en vez del O(Δt²) de usar solo
la muestra inicial. Para ω constante ambas son exactas.

Se integran DOS orientaciones en paralelo para ver el efecto del sesgo:
  R_corr:  usa ω_m - b̂_g   (b̂_g estimado en el tramo inicial en reposo)
  R_bruto: usa ω_m tal cual
La diferencia de yaw entre ambas crece linealmente: ≈ b_g,z · t.

Inicialización (tramo en reposo de t_reposo_s):
  b̂_g     = media de ω_m                     (en reposo ω = 0)
  R_w_i(0) = R_desde_gravedad(media de a_m)   (roll y pitch reales, yaw = 0)
Si en ese tramo σ(ω_m) es alta o ‖ā‖ se aleja de g, el dispositivo se movió:
se descarta y se vuelve a empezar.
"""

import numpy as np

from so3 import exp_so3, ortonormalizar, R_desde_gravedad, euler_zyx

FASE_INICIALIZANDO = "inicializando"
FASE_INTEGRANDO = "integrando"


class IntegradorGiro:

    def __init__(self, cfg):
        """cfg: el dict completo de config/hito_giro.yaml."""
        self.cfg_ini = cfg["inicializacion"]
        self.cfg_int = cfg["integracion"]
        self.reiniciar()

    def reiniciar(self):
        self.fase = FASE_INICIALIZANDO
        self.buf_t = []
        self.buf_w = []
        self.buf_a = []
        self.motivo_reinicio = None

        self.b_g = np.zeros(3)
        self.R_corr = np.eye(3)
        self.R_bruto = np.eye(3)
        self.t_prev = None
        self.w_prev = None
        self.n_pasos = 0
        self.n_huecos = 0

    # -----------------------------------------------------------------
    def _intentar_inicializar(self):
        """Comprueba el tramo en reposo y, si es válido, fija b̂_g y R_w_i(0)."""
        w = np.array(self.buf_w)
        a = np.array(self.buf_a)

        sigma_gir_dps = np.degrees(w.std(axis=0)).max()
        a_media = a.mean(axis=0)
        error_norma = abs(np.linalg.norm(a_media) - self.cfg_ini["g"])

        if sigma_gir_dps > self.cfg_ini["sigma_gir_max_dps"]:
            motivo = f"movimiento en la inicialización (σ gir = {sigma_gir_dps:.2f} °/s)"
            self.reiniciar()
            self.motivo_reinicio = motivo
            return
        if error_norma > self.cfg_ini["error_norma_acc_max"]:
            motivo = f"‖ā‖ lejos de g (error {error_norma:.2f} m/s²)"
            self.reiniciar()
            self.motivo_reinicio = motivo
            return

        self.b_g = w.mean(axis=0)
        R0 = R_desde_gravedad(a_media)
        self.R_corr = R0.copy()
        self.R_bruto = R0.copy()
        self.t_prev = self.buf_t[-1]
        self.w_prev = w[-1]
        self.fase = FASE_INTEGRANDO
        self.motivo_reinicio = None
        self.buf_t, self.buf_w, self.buf_a = [], [], []

    # -----------------------------------------------------------------
    def procesar(self, t_ns, w_m, a_m):
        """Procesa una muestra de IMU. Devuelve la fase actual."""
        w_m = np.asarray(w_m, dtype=np.float64)
        a_m = np.asarray(a_m, dtype=np.float64)
        self.a_ultima = a_m

        # ---------- Fase 1: acumular el tramo en reposo ----------
        if self.fase == FASE_INICIALIZANDO:
            self.buf_t.append(t_ns)
            self.buf_w.append(w_m)
            self.buf_a.append(a_m)
            if (t_ns - self.buf_t[0]) / 1e9 >= self.cfg_ini["t_reposo_s"]:
                self._intentar_inicializar()
            return self.fase

        # ---------- Fase 2: integrar ----------
        dt = (t_ns - self.t_prev) / 1e9
        if dt <= 0:
            return self.fase                      # marca repetida o desordenada: ignorar
        if dt > self.cfg_int["dt_max_s"]:
            self.n_huecos += 1                    # se integra igual, suponiendo ω constante

        w_medio = 0.5 * (self.w_prev + w_m)       # regla del punto medio

        # R_w_i ← R_w_i · Exp((ω̄ - b̂_g)·Δt)
        self.R_corr = self.R_corr @ exp_so3((w_medio - self.b_g) * dt)
        # Sin corregir el sesgo, para comparar
        self.R_bruto = self.R_bruto @ exp_so3(w_medio * dt)

        self.n_pasos += 1
        if self.n_pasos % self.cfg_int["reortonormalizar_cada"] == 0:
            self.R_corr = ortonormalizar(self.R_corr)
            self.R_bruto = ortonormalizar(self.R_bruto)

        self.t_prev = t_ns
        self.w_prev = w_m
        return self.fase

    # -----------------------------------------------------------------
    def progreso_inicializacion(self):
        """Fracción [0, 1] del tramo en reposo ya acumulado."""
        if self.fase != FASE_INICIALIZANDO or not self.buf_t:
            return 1.0 if self.fase == FASE_INTEGRANDO else 0.0
        return min(1.0, (self.buf_t[-1] - self.buf_t[0]) / 1e9 / self.cfg_ini["t_reposo_s"])

    def roll_pitch_acelerometro(self):
        """Roll y pitch instantáneos que dice la gravedad (válidos solo sin aceleración).

        Sirven de referencia: el giroscopio deriva en los tres ejes, pero roll y
        pitch se pueden corregir con la gravedad; el yaw no tiene referencia.
        """
        _, pitch, roll = euler_zyx(R_desde_gravedad(self.a_ultima))
        return roll, pitch
