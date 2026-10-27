#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
captura/grabar.py — Fase 1: grabación sincronizada IMU + cámara en formato EuRoC/ASL.

Se ejecuta en: [Raspberry Pi]

Hardware:
  - IMU:    MPU-6050 por I2C (bus 1, 0x68)
  - Cámara: OV5647 (Raspberry Pi Camera v1), obturador rodante

Base de tiempos
---------------
Todo va en ns (int64) con CLOCK_MONOTONIC:
  - Cámara: SensorTimestamp de libcamera = marca del buffer V4L2 (unicam),
    que el kernel toma con CLOCK_MONOTONIC en el inicio de trama (SOF).
    Con obturador rodante, el SOF NO coincide con el instante "medio" de
    exposición de la imagen; ese desfase constante lo absorbe el t_d de
    Kalibr (convención: t_cam_corregido = t_cam + t_d).
  - IMU: time.clock_gettime_ns(CLOCK_MONOTONIC) en cuanto el bit DATA_RDY
    de INT_STATUS indica muestra nueva. El retardo del DLPF (~4.8 ms)
    también acaba dentro de t_d.

Salida (formato EuRoC/ASL)
--------------------------
  <salida>/<AAAAMMDD_HHMMSS>/
    meta.json                      configuración y contadores
    mav0/cam0/data.csv             #timestamp [ns],filename
    mav0/cam0/data/<t_ns>.png      escala de grises 8 bits, sin pérdidas
    mav0/cam0/diagnostico.csv      t_sensor_ns,t_llegada_ns,exposicion_us,ganancia
    mav0/imu0/data.csv             #timestamp [ns],w_RS_S_x..z [rad s^-1],a_RS_S_x..z [m s^-2]
    mav0/imu0/diagnostico.csv      t_ns,temp_c

Uso
---
  [Raspberry Pi]  python3 captura/grabar.py --duracion 70
  [Raspberry Pi]  python3 captura/grabar.py --fps 30
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from datetime import datetime

import yaml

RUTA_CONFIG_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "config", "captura.yaml")

# Driver de la IMU: compartido con la lectura en vivo de vio/fuentes.py
from mpu6050 import LectorMPU6050, ahora_ns

CABECERA_IMU_EUROC = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
                      "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
                      "a_RS_S_z [m s^-2]\n")
CABECERA_CAM_EUROC = "#timestamp [ns],filename\n"


def cargar_config(ruta):
    with open(ruta) as f:
        return yaml.safe_load(f)


# =====================================================================
# IMU
# =====================================================================
def proceso_imu(carpeta_imu, cfg_imu, evento_parar, evento_lista, contador, errores):
    """Proceso dedicado a la IMU (otro proceso = otro intérprete, sin GIL compartido).

    La lectura (sondeo de DATA_RDY + marca CLOCK_MONOTONIC) está en
    captura/mpu6050.py. Error de la marca: latencia de una lectura I2C de
    1 byte (~0.1 ms a 400 kHz) más la planificación del SO; se mide en
    verificar.py (desviación típica de Δt).
    """
    lector = LectorMPU6050(cfg_imu)

    f_datos = open(os.path.join(carpeta_imu, "data.csv"), "w")
    f_diag = open(os.path.join(carpeta_imu, "diagnostico.csv"), "w")
    f_datos.write(CABECERA_IMU_EUROC)
    f_diag.write("t_ns,temp_c\n")
    evento_lista.set()

    n = 0
    try:
        while not evento_parar.is_set():
            t_ns, w, a, temp = lector.leer()

            # Formato EuRoC: giroscopio primero, luego acelerómetro
            f_datos.write(f"{t_ns},{w[0]:.7f},{w[1]:.7f},{w[2]:.7f},"
                          f"{a[0]:.6f},{a[1]:.6f},{a[2]:.6f}\n")
            f_diag.write(f"{t_ns},{temp:.2f}\n")
            n += 1
            contador.value = n
            errores.value = lector.errores
    except KeyboardInterrupt:
        pass   # Ctrl+C llega a todo el grupo de procesos: salir limpio
    finally:
        f_datos.close()
        f_diag.close()
        lector.cerrar()


# =====================================================================
# Cámara
# =====================================================================
def abrir_camara(cfg_cam):
    """Configura el modo binned 640x480 a fps fijo y salida YUV420.

    El plano Y de YUV420 ya es la luminancia: se usa directamente como
    imagen en grises sin conversión de color.
    """
    from picamera2 import Picamera2

    ancho, alto = cfg_cam["ancho"], cfg_cam["alto"]
    picam2 = Picamera2()
    periodo_us = int(round(1e6 / cfg_cam["fps"]))
    controles = {"FrameDurationLimits": (periodo_us, periodo_us)}
    principal = {"size": (ancho, alto), "format": "YUV420"}

    try:
        # Picamera2 >= 0.3.13: elección explícita del modo del sensor
        config = picam2.create_video_configuration(
            main=principal,
            sensor={"output_size": (ancho, alto), "bit_depth": cfg_cam["profundidad_bits"]},
            buffer_count=cfg_cam["buffers"],
            controls=controles)
    except TypeError:
        # Versiones antiguas: el modo se fuerza a través del flujo raw
        config = picam2.create_video_configuration(
            main=principal,
            raw={"size": (ancho, alto)},
            buffer_count=cfg_cam["buffers"],
            controls=controles)

    picam2.configure(config)
    return picam2


def fijar_exposicion(picam2, cfg_cam):
    """AE/AWB convergen y se congelan; exposición acotada, ganancia compensa.

    Exposición constante = constancia fotométrica entre frames, que es la
    hipótesis de KLT (y del error fotométrico de la Ec. 4 de SVO).
    """
    time.sleep(cfg_cam["t_ajuste_ae_s"])
    md = picam2.capture_metadata()
    exp_us = md["ExposureTime"]
    ganancia = md["AnalogueGain"]

    if exp_us > cfg_cam["exposicion_max_us"]:
        ganancia = ganancia * exp_us / cfg_cam["exposicion_max_us"]   # mismo brillo
        exp_us = cfg_cam["exposicion_max_us"]
    ganancia = min(ganancia, cfg_cam["ganancia_max"])

    picam2.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": int(exp_us),
        "AnalogueGain": float(ganancia),
    })
    time.sleep(cfg_cam["t_aplicar_controles_s"])
    md = picam2.capture_metadata()
    return md["ExposureTime"], md["AnalogueGain"]


def hilo_escritor(cola, compresion_png):
    """Escribe las PNG en segundo plano (cv2.imwrite libera el GIL)."""
    import cv2
    while True:
        elemento = cola.get()
        if elemento is None:
            cola.task_done()
            break
        ruta, imagen = elemento
        cv2.imwrite(ruta, imagen, [cv2.IMWRITE_PNG_COMPRESSION, compresion_png])
        cola.task_done()


# =====================================================================
# Programa principal
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Grabación IMU + cámara (formato EuRoC)")
    parser.add_argument("--config", default=RUTA_CONFIG_DEFECTO)
    parser.add_argument("--duracion", type=float, default=0.0,
                        help="segundos de imagen (0 = hasta Ctrl+C)")
    parser.add_argument("--fps", type=float, default=None, help="sobrescribe camara.fps")
    parser.add_argument("--salida", default=None, help="sobrescribe grabacion.carpeta_salida")
    args = parser.parse_args()

    cfg = cargar_config(args.config)
    cfg_imu, cfg_cam, cfg_grab = cfg["imu"], cfg["camara"], cfg["grabacion"]
    if args.fps is not None:
        cfg_cam["fps"] = args.fps
    if args.salida is not None:
        cfg_grab["carpeta_salida"] = args.salida

    # --- Estructura EuRoC ---
    raiz = os.path.join(cfg_grab["carpeta_salida"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    carpeta_cam = os.path.join(raiz, "mav0", "cam0")
    carpeta_img = os.path.join(carpeta_cam, "data")
    carpeta_imu = os.path.join(raiz, "mav0", "imu0")
    os.makedirs(carpeta_img)
    os.makedirs(carpeta_imu)
    print(f"Guardando en: {raiz}")

    # --- 1. IMU (antes que la cámara: graba ya el tramo inicial en reposo) ---
    evento_parar = mp.Event()
    evento_lista = mp.Event()
    contador_imu = mp.Value("q", 0)
    errores_imu = mp.Value("q", 0)
    p_imu = mp.Process(target=proceso_imu,
                       args=(carpeta_imu, cfg_imu, evento_parar, evento_lista,
                             contador_imu, errores_imu),
                       daemon=True)
    p_imu.start()
    if not evento_lista.wait(timeout=cfg_imu["timeout_arranque_s"]):
        print("ERROR: la IMU no arranca. ¿I2C activado? ¿WHO_AM_I correcto?")
        evento_parar.set()
        p_imu.terminate()
        return
    print("IMU grabando. MANTÉN EL DISPOSITIVO QUIETO.")

    picam2 = None
    cola = None
    hilo = None
    f_cam = None
    f_diag = None
    n_img = 0
    descartadas = 0
    exp_us, ganancia = None, None

    try:
        # --- 2. Cámara ---
        picam2 = abrir_camara(cfg_cam)
        picam2.start()
        print(f"Ajustando exposición ({cfg_cam['t_ajuste_ae_s'] + cfg_cam['t_aplicar_controles_s']:.1f} s)...")
        exp_us, ganancia = fijar_exposicion(picam2, cfg_cam)
        print(f"Exposición fija: {exp_us} us | ganancia analógica: {ganancia:.2f}")

        # --- 3. Escritor de imágenes ---
        cola = queue.Queue(maxsize=cfg_cam["cola_max_imagenes"])
        hilo = threading.Thread(target=hilo_escritor,
                                args=(cola, cfg_cam["compresion_png"]), daemon=True)
        hilo.start()

        # --- 4. Bucle de captura ---
        f_cam = open(os.path.join(carpeta_cam, "data.csv"), "w")
        f_diag = open(os.path.join(carpeta_cam, "diagnostico.csv"), "w")
        f_cam.write(CABECERA_CAM_EUROC)
        f_diag.write("t_sensor_ns,t_llegada_ns,exposicion_us,ganancia\n")

        ancho, alto = cfg_cam["ancho"], cfg_cam["alto"]
        print(">>> GRABANDO. Quieto 5 s más y después muévete con suavidad. Ctrl+C para parar.")
        t_inicio = time.monotonic()
        t_aviso = t_inicio

        while True:
            if args.duracion > 0 and time.monotonic() - t_inicio >= args.duracion:
                break

            # capture_request: buffer + metadatos del MISMO frame
            peticion = picam2.capture_request()
            t_llegada = ahora_ns()
            yuv = peticion.make_array("main")
            md = peticion.get_metadata()
            peticion.release()

            gris = yuv[:alto, :ancho].copy()      # plano Y
            t_sensor = md["SensorTimestamp"]
            nombre = f"{t_sensor}.png"

            try:
                cola.put_nowait((os.path.join(carpeta_img, nombre), gris))
            except queue.Full:
                descartadas += 1
                continue

            f_cam.write(f"{t_sensor},{nombre}\n")
            f_diag.write(f"{t_sensor},{t_llegada},{md['ExposureTime']},"
                         f"{md['AnalogueGain']:.3f}\n")
            n_img += 1

            ahora = time.monotonic()
            if ahora - t_aviso >= cfg_grab["aviso_cada_s"]:
                print(f"  t={ahora - t_inicio:5.1f} s | img {n_img} | IMU {contador_imu.value} | "
                      f"cola {cola.qsize()} | descartadas {descartadas} | "
                      f"errores I2C {errores_imu.value}")
                t_aviso = ahora

    except KeyboardInterrupt:
        print("\nParando...")

    finally:
        # --- 5. Cierre ordenado ---
        evento_parar.set()
        p_imu.join(timeout=3.0)
        if picam2 is not None:
            picam2.stop()
            picam2.close()
        if cola is not None:
            print("Vaciando la cola de imágenes...")
            cola.put(None)
            cola.join()
        if hilo is not None:
            hilo.join()
        for f in (f_cam, f_diag):
            if f is not None:
                f.close()

        meta = {
            "fecha": datetime.now().isoformat(),
            "reloj": "CLOCK_MONOTONIC, ns",
            "formato": "EuRoC/ASL",
            "imu": dict(cfg_imu, muestras=contador_imu.value, errores_i2c=errores_imu.value),
            "camara": dict(cfg_cam, exposicion_us_fijada=exp_us,
                           ganancia_fijada=ganancia, imagenes=n_img,
                           descartadas_por_disco=descartadas,
                           timestamp="SensorTimestamp (libcamera, SOF)"),
        }
        with open(os.path.join(raiz, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"Hecho. IMU {contador_imu.value} | imágenes {n_img} | "
              f"descartadas {descartadas} | errores I2C {errores_imu.value}")
        print(f"Verifica con:  python3 captura/verificar.py {raiz}")


if __name__ == "__main__":
    main()
