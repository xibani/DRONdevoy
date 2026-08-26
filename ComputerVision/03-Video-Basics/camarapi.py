#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CLIENTE OPENCV  ->  se ejecuta EN TU PC (el del curso)
================================================================================

Lee el stream MJPEG que publica la Raspberry Pi y lo trata exactamente igual
que una webcam local. Sustituye a  cv2.VideoCapture(0)  del curso.

MODOS DE DEMO (autovalidacion):
    python camara_pi_cliente.py test        -> comprueba conexion y muestra info
    python camara_pi_cliente.py gris        -> equivalente al notebook 00
    python camara_pi_cliente.py grabar      -> equivalente al notebook 00 (parte 2)
    python camara_pi_cliente.py rectangulo  -> equivalente al notebook 02
    python camara_pi_cliente.py interactivo -> equivalente al notebook 02 (raton)

Pulsa 'q' sobre la ventana para salir.
================================================================================
"""

import sys
import time
from threading import Thread, Lock
import queue

import cv2

# ------------------------------------------------------------------------------
# 1) CONFIGURACION: cambia esta IP por la de TU Raspberry Pi
#    (en la Pi:  hostname -I  )
# ------------------------------------------------------------------------------
# IP_RASPBERRY = "192.168.1.112"
IP_RASPBERRY = "192.168.0.21"
PUERTO = 8000
URL_STREAM = f"http://{IP_RASPBERRY}:{PUERTO}/stream.mjpg"


# ------------------------------------------------------------------------------
# 2) LECTOR EN HILO
#
#    Por que hace falta: en un stream de red, si no lees tan rapido como llegan
#    los frames, se acumulan en el buffer y ves la imagen con 2-3 s de retraso.
#    Este hilo lee SIEMPRE y descarta lo viejo; el programa principal se queda
#    solo con el ultimo frame disponible. Latencia baja y constante.
# ------------------------------------------------------------------------------
class CamaraPi:

    def __init__(self, url=URL_STREAM):
        self.url = URL_STREAM
        self.cap = cv2.VideoCapture(url)

        # Buffer mínimo (no todos los backends lo respetan, por eso el hilo)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir el stream: {url}\n"
                "Comprueba que: (a) el servidor esta corriendo en la Pi, "
                "(b) la IP es correcta, (c) PC y Pi estan en la misma red."
            )

        self.frame = None
        self.parado = False
        self.lock = Lock()

        # Arrancanmos el hilo lector
        self.hilo = Thread(target=self._bucle_lectura, daemon=True)
        self.hilo.start()

        # Esperar al primer frame (max 5s) para poder saber la resolución
        t0 = time.time()
        while self.frame is None and time.time() - t0 < 5.0:
            time.sleep(0.05)
        if self.frame is None:
            raise RuntimeError("Stream abierto pero no llegan frames.")

    def _bucle_lectura(self):
        """Se ejecuta en segundo plano: lee sin parar y guarda el último frame."""
        while not self.parado:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        """Devuelve (ret, frame) igual que cap.read() del curso."""
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    @property
    def width(self):
        return self.frame.shape[1]

    @property
    def height(self):
        return self.frame.shape[0]

    def release(self):
        self.parado = True
        self.hilo.join(timeout=1.0)
        self.cap.release()
