#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
captura/mpu6050.py — Driver mínimo del MPU-6050 por I2C.

Se usa en: [Raspberry Pi]  (lo importan captura/grabar.py y vio/fuentes.py)

Contiene:
  - configurar_imu(bus, cfg_imu):  deja el chip a la frecuencia/rangos del YAML
  - convertir_bloque_imu(d, cfg):  14 bytes crudos -> (w [rad/s], a [m/s^2], temp [°C])
  - LectorMPU6050:                 lectura bloqueante muestra a muestra con marca CLOCK_MONOTONIC

Las funciones de conversión no dependen de smbus2, así que se pueden probar
en el PC sin hardware (tests/test_captura.py).
"""

import math
import os
import time

# ---------------------------------------------------------------------
# Registros (mapa de registros del MPU-6050, rev. 4.2): definiciones de
# hardware, no parámetros.
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
ERRORES_SEGUIDOS_MAX = 100   # tantos fallos I2C seguidos = IMU desconectada

G = 9.80665


def ahora_ns():
    """Instante actual en ns con CLOCK_MONOTONIC (mismo reloj que la cámara)."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def a_entero_con_signo(byte_alto, byte_bajo):
    """Une dos bytes big-endian en un int16 (complemento a dos)."""
    valor = (byte_alto << 8) | byte_bajo
    if valor >= 32768:
        valor = valor - 65536
    return valor


def convertir_bloque_imu(d, cfg_imu):
    """Convierte el bloque de 14 bytes del MPU-6050 a unidades SI.

    Orden del bloque: AX AY AZ TEMP GX GY GZ (2 bytes cada uno, big-endian).
    Devuelve (w [rad/s], a [m/s^2], temp [°C]) como tuplas de floats.
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
    temp = tp_c / 340.0 + 36.53   # fórmula de la hoja de datos
    return w, a, temp


def configurar_imu(bus, cfg_imu):
    """Reset y configuración: DLPF, frecuencia de salida, rangos y DATA_RDY."""
    dir_imu = cfg_imu["direccion"]

    quien = bus.read_byte_data(dir_imu, REG_WHO_AM_I)
    if quien != cfg_imu["who_am_i_esperado"]:
        raise RuntimeError(f"WHO_AM_I = {hex(quien)}, se esperaba "
                           f"{hex(cfg_imu['who_am_i_esperado'])}")

    bus.write_byte_data(dir_imu, REG_PWR_MGMT_1, PWR_RESET)
    time.sleep(T_RESET_S)
    bus.write_byte_data(dir_imu, REG_PWR_MGMT_1, PWR_RELOJ_GIR_X)
    time.sleep(T_RESET_S)

    # f_salida = 1 kHz / (1 + SMPLRT_DIV)   (con DLPF activo)
    bus.write_byte_data(dir_imu, REG_CONFIG, cfg_imu["dlpf_cfg"])
    bus.write_byte_data(dir_imu, REG_SMPLRT_DIV, cfg_imu["smplrt_div"])

    bus.write_byte_data(dir_imu, REG_GYRO_CONFIG, cfg_imu["gyro_config"])
    bus.write_byte_data(dir_imu, REG_ACCEL_CONFIG, cfg_imu["accel_config"])

    bus.write_byte_data(dir_imu, REG_INT_PIN_CFG, INT_LATCH_ANYRD)
    bus.write_byte_data(dir_imu, REG_INT_ENABLE, INT_DATA_RDY)
    time.sleep(T_ESTABILIZAR_S)


class LectorMPU6050:
    """Lectura bloqueante: cada llamada a leer() devuelve la siguiente muestra.

    Sondeo de DATA_RDY:
      1. leer INT_STATUS hasta ver el bit 0 a 1 (esa lectura también lo borra)
      2. tomar la marca de tiempo CLOCK_MONOTONIC
      3. leer el bloque de 14 bytes
    """

    def __init__(self, cfg_imu):
        from smbus2 import SMBus   # solo existe en la Pi

        try:
            os.sched_setaffinity(0, {cfg_imu["nucleo_cpu"]})
        except Exception:
            pass

        self.cfg = cfg_imu
        self.dir = cfg_imu["direccion"]
        self.bus = SMBus(cfg_imu["bus_i2c"])
        configurar_imu(self.bus, cfg_imu)
        self.bus.read_byte_data(self.dir, REG_INT_STATUS)   # limpiar flag pendiente
        self.errores = 0

    def leer(self):
        """Espera a la siguiente muestra. Devuelve (t_ns, w, a, temp)."""
        seguidos = 0
        while True:
            # --- 1. ¿muestra nueva? ---
            try:
                estado = self.bus.read_byte_data(self.dir, REG_INT_STATUS)
            except OSError:
                self.errores += 1
                seguidos += 1
                if seguidos > ERRORES_SEGUIDOS_MAX:
                    raise RuntimeError("Demasiados errores I2C seguidos: ¿IMU desconectada?")
                continue
            if (estado & INT_DATA_RDY) == 0:
                continue

            # --- 2. marca de tiempo ---
            t_ns = ahora_ns()

            # --- 3. bloque de datos ---
            try:
                d = self.bus.read_i2c_block_data(self.dir, REG_ACCEL_XOUT_H, 14)
            except OSError:
                self.errores += 1
                seguidos += 1
                continue

            w, a, temp = convertir_bloque_imu(d, self.cfg)
            return t_ns, w, a, temp

    def cerrar(self):
        self.bus.close()
