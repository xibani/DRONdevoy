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
import math
import multiprocessing as mp
import os
import queue
import threading
import time
from datetime import datetime

import yaml

RUTA_CONFIG_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "config", "captura.yaml")

# ---------------------------------------------------------------------
# Registros del MPU-6050 (mapa de registros, rev. 4.2). Son definiciones
# del hardware, no parámetros: por eso viven aquí y no en el YAML.
# ---------------------------------------------------------------------
REG_SMPLRT_DIV   = 0x19
REG_CONFIG       = 0x1A
REG_GYRO_CONFIG  = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_INT_PIN_CFG  = 0x37
REG_INT_ENABLE   = 0x38
REG_INT_STATUS   = 0x3A   # bit 0 = DATA_RDY_INT
REG_ACCEL_XOUT_H = 0x3B   # inicio del bloque acel(6) + temp(2) + gir(6)
REG_PWR_MGMT_1   = 0x6B
REG_WHO_AM_I     = 0x75

PWR_RESET        = 0x80   # DEVICE_RESET
PWR_RELOJ_GIR_X  = 0x01   # despierto, reloj = PLL del giroscopio X
INT_LATCH_ANYRD  = 0x20   # LATCH_INT_EN: el flag se mantiene hasta leer INT_STATUS
INT_DATA_RDY     = 0x01   # DATA_RDY_EN

T_RESET_S        = 0.1    # espera tras reset/despertar (hoja de datos)
T_ESTABILIZAR_S  = 0.05

G = 9.80665               # gravedad estándar, para convertir g -> m/s^2

CABECERA_IMU_EUROC = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
                      "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
                      "a_RS_S_z [m s^-2]\n")
CABECERA_CAM_EUROC = "#timestamp [ns],filename\n"


# =====================================================================
# Utilidades
# =====================================================================
def ahora_ns():
    """Instante actual en ns con CLOCK_MONOTONIC (mismo reloj que la cámara)."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def cargar_config(ruta):
    with open(ruta) as f:
        return yaml.safe_load(f)


def a_entero_con_signo(byte_alto, byte_bajo):
    """Une dos bytes big-endian en un int16 (complemento a dos)."""
    valor = (byte_alto << 8) | byte_bajo
    if valor >= 32768:
        valor = valor - 65536
    return valor


def convertir_bloque_imu(d, cfg_imu):
    """Convierte el bloque de 14 bytes del MPU-6050 a unidades SI.

    Orden del bloque: AX AY AZ TEMP GX GY GZ (cada uno 2 bytes, big-endian).
    Devuelve (w [rad/s], a [m/s^2], temp [°C]).
    """
    ax_c = a_entero_con_signo(d[0], d[1])
    ay_c = a_entero_con_signo(d[2], d[3])
    az_c = a_entero_con_signo(d[4], d[5])
    tp_c = a_entero_con_signo(d[6], d[7])
    gx_c = a_entero_con_signo(d[8], d[9])
    gy_c = a_entero_con_signo(d[10], d[11])
    gz_c = a_entero_con_signo(d[12], d[13])

    esc_acc = cfg_imu["escala_acc_lsb_por_g"]
    esc_gir = cfg_imu["escala_gir_lsb_por_dps"]

    a = (ax_c / esc_acc * G, ay_c / esc_acc * G, az_c / esc_acc * G)
    w = (math.radians(gx_c / esc_gir),
         math.radians(gy_c / esc_gir),
         math.radians(gz_c / esc_gir))
    temp = tp_c / 340.0 + 36.53   # fórmula de la hoja de datos del MPU-6050
    return w, a, temp


# =====================================================================
# IMU
# =====================================================================
def configurar_imu(bus, cfg_imu):
    dir_imu = cfg_imu["direccion"]

    quien = bus.read_byte_data(dir_imu, REG_WHO_AM_I)
    if quien != cfg_imu["who_am_i_esperado"]:
        raise RuntimeError(f"WHO_AM_I = {hex(quien)}, se esperaba "
                           f"{hex(cfg_imu['who_am_i_esperado'])}")

    # Reset completo -> estado conocido
    bus.write_byte_data(dir_imu, REG_PWR_MGMT_1, PWR_RESET)
    time.sleep(T_RESET_S)

    # Despertar con el PLL del giroscopio X como reloj (más estable que el
    # oscilador interno de 8 MHz: menos deriva de la frecuencia de muestreo)
    bus.write_byte_data(dir_imu, REG_PWR_MGMT_1, PWR_RELOJ_GIR_X)
    time.sleep(T_RESET_S)

    # DLPF y divisor de muestreo: f_salida = 1 kHz / (1 + SMPLRT_DIV)
    bus.write_byte_data(dir_imu, REG_CONFIG, cfg_imu["dlpf_cfg"])
    bus.write_byte_data(dir_imu, REG_SMPLRT_DIV, cfg_imu["smplrt_div"])

    # Rangos
    bus.write_byte_data(dir_imu, REG_GYRO_CONFIG, cfg_imu["gyro_config"])
    bus.write_byte_data(dir_imu, REG_ACCEL_CONFIG, cfg_imu["accel_config"])

    # DATA_RDY con latch: el flag sigue activo hasta que leemos INT_STATUS,
    # así no se pierde aunque el sondeo llegue tarde
    bus.write_byte_data(dir_imu, REG_INT_PIN_CFG, INT_LATCH_ANYRD)
    bus.write_byte_data(dir_imu, REG_INT_ENABLE, INT_DATA_RDY)
    time.sleep(T_ESTABILIZAR_S)


def proceso_imu(carpeta_imu, cfg_imu, evento_parar, evento_lista, contador, errores):
    """Proceso dedicado a la IMU (otro proceso = otro intérprete, sin GIL compartido).

    Sondeo de DATA_RDY:
      1. leer INT_STATUS hasta ver el bit 0 a 1 (lectura que además lo borra)
      2. tomar la marca de tiempo
      3. leer el bloque de 14 bytes
    Error de la marca: latencia de una lectura I2C de 1 byte (~0.1 ms a 400 kHz)
    más la planificación del SO. Se mide en verificar.py (desv. típica de Δt).
    """
    from smbus2 import SMBus

    try:
        os.sched_setaffinity(0, {cfg_imu["nucleo_cpu"]})
    except Exception:
        pass

    dir_imu = cfg_imu["direccion"]
    bus = SMBus(cfg_imu["bus_i2c"])
    configurar_imu(bus, cfg_imu)

    f_datos = open(os.path.join(carpeta_imu, "data.csv"), "w")
    f_diag = open(os.path.join(carpeta_imu, "diagnostico.csv"), "w")
    f_datos.write(CABECERA_IMU_EUROC)
    f_diag.write("t_ns,temp_c\n")

    bus.read_byte_data(dir_imu, REG_INT_STATUS)   # limpiar un flag pendiente
    evento_lista.set()

    n = 0
    try:
        while not evento_parar.is_set():
            # --- 1. ¿muestra nueva? ---
            try:
                estado = bus.read_byte_data(dir_imu, REG_INT_STATUS)
            except OSError:
                errores.value += 1
                continue
            if (estado & INT_DATA_RDY) == 0:
                continue

            # --- 2. marca de tiempo ---
            t_ns = ahora_ns()

            # --- 3. bloque de datos ---
            try:
                d = bus.read_i2c_block_data(dir_imu, REG_ACCEL_XOUT_H, 14)
            except OSError:
                errores.value += 1
                continue

            w, a, temp = convertir_bloque_imu(d, cfg_imu)

            # Formato EuRoC: giroscopio primero, luego acelerómetro
            f_datos.write(f"{t_ns},{w[0]:.7f},{w[1]:.7f},{w[2]:.7f},"
                          f"{a[0]:.6f},{a[1]:.6f},{a[2]:.6f}\n")
            f_diag.write(f"{t_ns},{temp:.2f}\n")
            n += 1
            contador.value = n
    except KeyboardInterrupt:
        pass   # Ctrl+C llega a todo el grupo de procesos: salir limpio
    finally:
        f_datos.close()
        f_diag.close()
        bus.close()


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
