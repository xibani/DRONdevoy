#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vio/fuentes.py — Fuentes de datos intercambiables para el estimador.

El estimador no sabe de dónde vienen los datos: recibe una secuencia de
eventos ordenados por tiempo. Hay dos fuentes con la misma interfaz
(generadores de EventoIMU):

  fuente_grabacion_imu(raiz)   lee mav0/imu0/data.csv de una grabación EuRoC
                               ([PC] o [Raspberry Pi])
  fuente_vivo_imu(cfg_imu)     lee el MPU-6050 en tiempo real ([Raspberry Pi])

Cuando entre la cámara, se añadirá EventoImagen y una mezcla ordenada por t_ns.
"""

import os
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class EventoIMU:
    t_ns: int            # CLOCK_MONOTONIC [ns]
    w: np.ndarray        # velocidad angular medida, ejes IMU [rad/s]
    a: np.ndarray        # fuerza específica medida, ejes IMU [m/s²]


def leer_imu_euroc(ruta_csv):
    """Lee un data.csv de IMU EuRoC. Timestamps como int (sin pasar por float)."""
    t, w, a = [], [], []
    with open(ruta_csv) as f:
        for linea in f:
            if linea.startswith("#") or not linea.strip():
                continue
            c = linea.strip().split(",")
            t.append(int(c[0]))
            w.append([float(c[1]), float(c[2]), float(c[3])])
            a.append([float(c[4]), float(c[5]), float(c[6])])
    return (np.array(t, dtype=np.int64),
            np.array(w, dtype=np.float64),
            np.array(a, dtype=np.float64))


def fuente_grabacion_imu(raiz, tiempo_real=False, velocidad=1.0):
    """Reproduce la IMU de una grabación EuRoC.

    tiempo_real=False: entrega los eventos tan rápido como se consuman
                       (procesado offline, tests).
    tiempo_real=True:  espera entre eventos lo que indican sus timestamps
                       (dividido por 'velocidad'), para simular la Pi en vivo.
    """
    t, w, a = leer_imu_euroc(os.path.join(raiz, "mav0", "imu0", "data.csv"))

    t_inicio_reloj = time.monotonic()
    for k in range(len(t)):
        if tiempo_real:
            # Instante (en el reloj del PC) en que "debería" llegar esta muestra
            objetivo = t_inicio_reloj + (t[k] - t[0]) / 1e9 / velocidad
            espera = objetivo - time.monotonic()
            if espera > 0:
                time.sleep(espera)
        yield EventoIMU(int(t[k]), w[k], a[k])


def fuente_vivo_imu(cfg_imu):
    """Lee el MPU-6050 indefinidamente. Solo en la Raspberry Pi."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captura"))
    from mpu6050 import LectorMPU6050

    lector = LectorMPU6050(cfg_imu)
    try:
        while True:
            t_ns, w, a, _temp = lector.leer()
            yield EventoIMU(t_ns, np.array(w), np.array(a))
    finally:
        lector.cerrar()
