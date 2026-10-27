#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vio/protocolo.py — Mensajes Pi -> PC por UDP.

Cada mensaje es un datagrama UDP con un objeto JSON en UTF-8. JSON es más
lento y voluminoso que un formato binario, pero se puede leer a simple
vista al depurar (p. ej. con Wireshark) y a 30 Hz el coste es despreciable.

UDP no garantiza entrega ni orden: si se pierde un paquete, el siguiente
lo sustituye (solo importa la pose más reciente). El número de secuencia
'seq' permite contar pérdidas en el PC.

Campos del mensaje de orientación:
  tipo         "orientacion"
  seq          contador creciente
  t_ns         timestamp de la última muestra integrada (CLOCK_MONOTONIC de la Pi)
  fase         "inicializando" | "integrando"
  progreso     [0, 1] de la inicialización
  aviso        texto o null (p. ej. motivo de reinicio de la inicialización)
  R_corr       R_w_i con sesgo corregido, 9 floats por filas
  R_bruto      R_w_i sin corregir el sesgo, 9 floats por filas
  b_g          sesgo estimado del giroscopio [rad/s]
  rp_acc       [roll, pitch] de la gravedad instantánea [rad]
"""

import json

import numpy as np

TAM_MAX_DATAGRAMA = 65507   # máximo de UDP sobre IPv4


def codificar(mensaje):
    """dict (con arrays numpy) -> bytes."""
    plano = {}
    for clave, valor in mensaje.items():
        if isinstance(valor, np.ndarray):
            plano[clave] = valor.reshape(-1).tolist()
        elif isinstance(valor, (np.floating, np.integer)):
            plano[clave] = valor.item()
        else:
            plano[clave] = valor
    datos = json.dumps(plano).encode("utf-8")
    if len(datos) > TAM_MAX_DATAGRAMA:
        raise ValueError(f"Mensaje de {len(datos)} bytes: demasiado grande para UDP")
    return datos


def decodificar(datos):
    """bytes -> dict; las matrices R_* vuelven como arrays 3×3."""
    mensaje = json.loads(datos.decode("utf-8"))
    for clave in ("R_corr", "R_bruto"):
        if clave in mensaje:
            mensaje[clave] = np.array(mensaje[clave], dtype=np.float64).reshape(3, 3)
    for clave in ("b_g", "rp_acc"):
        if clave in mensaje and mensaje[clave] is not None:
            mensaje[clave] = np.array(mensaje[clave], dtype=np.float64)
    return mensaje
