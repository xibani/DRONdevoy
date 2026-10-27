#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apps/pi_orientacion.py — Hito 1: orientación por giroscopio en la Pi, enviada al PC.

Se ejecuta en: [Raspberry Pi]  (también en el PC con --grabacion, para probar)

Flujo:
  fuente (MPU-6050 en vivo, o una grabación EuRoC)
     -> IntegradorGiro (200 Hz: inicialización en reposo + integración en SO(3))
     -> mensaje JSON por UDP al PC (a frecuencia_envio_hz)

Uso:
  [Raspberry Pi]  python3 apps/pi_orientacion.py --destino 192.168.1.100
  [Raspberry Pi]  python3 apps/pi_orientacion.py --grabacion datos/20261026_183000
  [WSL/Ubuntu] o [Windows PowerShell] (prueba local sin la Pi):
                  python apps/pi_orientacion.py --grabacion <carpeta> --destino 127.0.0.1
"""

import argparse
import os
import socket
import sys
import time

import numpy as np
import yaml

RAIZ_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ_REPO, "vio"))

from fuentes import fuente_grabacion_imu, fuente_vivo_imu   # noqa: E402
from integrador_giro import IntegradorGiro, FASE_INTEGRANDO   # noqa: E402
from so3 import euler_zyx                                      # noqa: E402
import protocolo                                               # noqa: E402


def cargar_yaml(ruta):
    with open(ruta) as f:
        return yaml.safe_load(f)


def construir_mensaje(integ, t_ns, seq):
    roll_acc, pitch_acc = integ.roll_pitch_acelerometro()
    return {
        "tipo": "orientacion",
        "seq": seq,
        "t_ns": int(t_ns),
        "fase": integ.fase,
        "progreso": integ.progreso_inicializacion(),
        "aviso": integ.motivo_reinicio,
        "R_corr": integ.R_corr,
        "R_bruto": integ.R_bruto,
        "b_g": integ.b_g,
        "rp_acc": np.array([roll_acc, pitch_acc]),
    }


def main():
    parser = argparse.ArgumentParser(description="Orientación por giroscopio -> PC")
    parser.add_argument("--config", default=os.path.join(RAIZ_REPO, "config", "hito_giro.yaml"))
    parser.add_argument("--config-captura", default=os.path.join(RAIZ_REPO, "config", "captura.yaml"))
    parser.add_argument("--grabacion", default=None,
                        help="carpeta EuRoC a reproducir en lugar de leer la IMU en vivo")
    parser.add_argument("--rapido", action="store_true",
                        help="con --grabacion: no esperar entre muestras")
    parser.add_argument("--destino", default=None, help="IP del PC (sobrescribe red.ip_destino)")
    parser.add_argument("--puerto", type=int, default=None)
    args = parser.parse_args()

    cfg = cargar_yaml(args.config)
    cfg_imu = cargar_yaml(args.config_captura)["imu"]
    ip = args.destino or cfg["red"]["ip_destino"]
    puerto = args.puerto or cfg["red"]["puerto"]
    periodo_envio_ns = int(1e9 / cfg["red"]["frecuencia_envio_hz"])

    # --- Fuente de datos: misma interfaz en vivo y en reproducción ---
    if args.grabacion:
        fuente = fuente_grabacion_imu(args.grabacion, tiempo_real=not args.rapido)
        print(f"Fuente: grabación {args.grabacion}")
    else:
        fuente = fuente_vivo_imu(cfg_imu)
        print("Fuente: MPU-6050 en vivo")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"Enviando a {ip}:{puerto} | MANTÉN EL DISPOSITIVO QUIETO "
          f"{cfg['inicializacion']['t_reposo_s']:.0f} s")

    integ = IntegradorGiro(cfg)
    seq = 0
    t_ultimo_envio = None
    t_aviso = time.monotonic()
    n_muestras = 0
    fase_anterior = integ.fase
    aviso_anterior = None

    try:
        for ev in fuente:
            integ.procesar(ev.t_ns, ev.w, ev.a)
            n_muestras += 1

            # Mensajes de consola cuando cambia algo importante
            if integ.motivo_reinicio and integ.motivo_reinicio != aviso_anterior:
                print(f"  Reinicio de la inicialización: {integ.motivo_reinicio}. Quieto otra vez.")
            aviso_anterior = integ.motivo_reinicio
            if integ.fase != fase_anterior and integ.fase == FASE_INTEGRANDO:
                b = np.degrees(integ.b_g)
                _, pitch0, roll0 = euler_zyx(integ.R_corr)
                print(f"  Inicializado. b_g = [{b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f}] °/s | "
                      f"roll {np.degrees(roll0):+.1f}° pitch {np.degrees(pitch0):+.1f}°. Ya puedes mover.")
            fase_anterior = integ.fase

            # Envío limitado a frecuencia_envio_hz (por tiempo de los datos, no del reloj)
            if t_ultimo_envio is None or ev.t_ns - t_ultimo_envio >= periodo_envio_ns:
                datos = protocolo.codificar(construir_mensaje(integ, ev.t_ns, seq))
                try:
                    sock.sendto(datos, (ip, puerto))
                except OSError as e:
                    print(f"  Error de red: {e}")
                seq += 1
                t_ultimo_envio = ev.t_ns

            # Estado por consola cada segundo
            ahora = time.monotonic()
            if ahora - t_aviso >= 1.0 and integ.fase == FASE_INTEGRANDO:
                yc, _, _ = euler_zyx(integ.R_corr)
                yb, _, _ = euler_zyx(integ.R_bruto)
                print(f"  muestras {n_muestras} | enviados {seq} | yaw corr {np.degrees(yc):+7.1f}° | "
                      f"yaw bruto {np.degrees(yb):+7.1f}° | huecos {integ.n_huecos}")
                t_aviso = ahora
    except KeyboardInterrupt:
        print("\nParado.")
    finally:
        sock.close()

    print(f"Fin. {n_muestras} muestras, {seq} mensajes enviados.")


if __name__ == "__main__":
    main()
