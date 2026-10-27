#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_captura.py — Tests sintéticos de la Fase 1.

Se ejecuta en: [Windows PowerShell] / [WSL/Ubuntu] / [Raspberry Pi]
  python tests/test_captura.py          (o:  python -m pytest tests/test_captura.py)

No necesita cámara ni IMU: genera grabaciones falsas en formato EuRoC con
defectos conocidos (muestra de IMU perdida, frame perdido, relojes distintos)
y comprueba que verificar.py los detecta con los valores exactos.
También valida la conversión de bytes del MPU-6050 de grabar.py.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml

RAIZ_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ_REPO, "captura"))

import grabar      # noqa: E402  (picamera2 y smbus2 se importan dentro de las funciones)
import verificar   # noqa: E402

G = 9.80665
with open(os.path.join(RAIZ_REPO, "config", "captura.yaml")) as _f:
    CFG = yaml.safe_load(_f)


# =====================================================================
# Generador de grabaciones sintéticas
# =====================================================================
def crear_grabacion(carpeta, duracion_imu_s=66.0, retraso_cam_s=3.0, duracion_cam_s=61.0,
                    indices_imu_perdidos=(), indices_cam_perdidos=(),
                    offset_reloj_llegada_ns=0, semilla=0):
    """Escribe una grabación EuRoC falsa y devuelve los parámetros "verdaderos"."""
    rng = np.random.default_rng(semilla)
    f_imu = CFG["imu"]["frecuencia_hz"]
    f_cam = CFG["camara"]["fps"]
    t0 = 1_000_000_000_000   # ~16 min tras el arranque, en ns

    # ---- IMU: 200 Hz con jitter gaussiano de 0.1 ms ----
    n_imu = int(duracion_imu_s * f_imu)
    t_imu = t0 + np.round(np.arange(n_imu) * 1e9 / f_imu
                          + rng.normal(0.0, 1e5, n_imu)).astype(np.int64)
    sesgo_gir = np.radians([0.5, -0.3, 0.1])           # rad/s
    w = sesgo_gir + rng.normal(0.0, np.radians(0.05), (n_imu, 3))
    a = np.array([0.0, 0.0, G]) + rng.normal(0.0, 0.02, (n_imu, 3))   # +z hacia arriba
    conservar_imu = np.setdiff1d(np.arange(n_imu), indices_imu_perdidos)

    # ---- Cámara: 20 fps exactos (el sensor es muy regular) ----
    n_cam = int(duracion_cam_s * f_cam)
    t_cam = t0 + int(retraso_cam_s * 1e9) + np.round(np.arange(n_cam) * 1e9 / f_cam).astype(np.int64)
    t_lleg = t_cam + 30_000_000 + offset_reloj_llegada_ns   # 30 ms de latencia
    conservar_cam = np.setdiff1d(np.arange(n_cam), indices_cam_perdidos)

    # ---- Escritura en formato EuRoC ----
    carpeta_imu = os.path.join(carpeta, "mav0", "imu0")
    carpeta_cam = os.path.join(carpeta, "mav0", "cam0")
    carpeta_img = os.path.join(carpeta_cam, "data")
    os.makedirs(carpeta_imu)
    os.makedirs(carpeta_img)

    with open(os.path.join(carpeta_imu, "data.csv"), "w") as f:
        f.write(grabar.CABECERA_IMU_EUROC)
        for i in conservar_imu:
            f.write(f"{t_imu[i]},{w[i, 0]:.7f},{w[i, 1]:.7f},{w[i, 2]:.7f},"
                    f"{a[i, 0]:.6f},{a[i, 1]:.6f},{a[i, 2]:.6f}\n")
    with open(os.path.join(carpeta_imu, "diagnostico.csv"), "w") as f:
        f.write("t_ns,temp_c\n")
        for i in conservar_imu:
            f.write(f"{t_imu[i]},30.00\n")

    with open(os.path.join(carpeta_cam, "data.csv"), "w") as fd, \
         open(os.path.join(carpeta_cam, "diagnostico.csv"), "w") as fg:
        fd.write(grabar.CABECERA_CAM_EUROC)
        fg.write("t_sensor_ns,t_llegada_ns,exposicion_us,ganancia\n")
        for i in conservar_cam:
            nombre = f"{t_cam[i]}.png"
            fd.write(f"{t_cam[i]},{nombre}\n")
            fg.write(f"{t_cam[i]},{t_lleg[i]},8000,2.000\n")
            open(os.path.join(carpeta_img, nombre), "wb").close()   # PNG vacía: solo se comprueba que existe

    meta = {"imu": {"frecuencia_hz": f_imu}, "camara": {"fps": f_cam, "descartadas_por_disco": 0}}
    with open(os.path.join(carpeta, "meta.json"), "w") as f:
        json.dump(meta, f)

    return {"sesgo_gir_dps": np.degrees(sesgo_gir)}


def analizar(carpeta):
    """Ejecuta el mismo análisis que verificar.main() y devuelve los resultados."""
    cv = CFG["verificacion"]
    datos = verificar.cargar_grabacion(carpeta)
    r_imu = verificar.analizar_intervalos(datos["t_imu"], CFG["imu"]["frecuencia_hz"], cv["factor_hueco"])
    r_cam = verificar.analizar_intervalos(datos["t_cam"], CFG["camara"]["fps"], cv["factor_hueco"])
    r_rel = verificar.analizar_reloj(datos["t_cam"], datos["t_llegada"], datos["t_imu"], cv["latencia_max_ms"])
    r_rep = verificar.analizar_reposo(datos["t_imu"], datos["w"], datos["a"], cv["t_reposo_s"], cv["g"])
    crit = verificar.evaluar_criterio(r_imu, r_cam, r_rel, 0, cv["criterio"])
    return r_imu, r_cam, r_rel, r_rep, crit


# =====================================================================
# Tests
# =====================================================================
def test_conversion_bytes_mpu6050():
    """int16 big-endian en complemento a dos y escalado a SI."""
    assert grabar.a_entero_con_signo(0x7F, 0xFF) == 32767
    assert grabar.a_entero_con_signo(0x80, 0x00) == -32768
    assert grabar.a_entero_con_signo(0xFF, 0xFF) == -1
    assert grabar.a_entero_con_signo(0x00, 0x00) == 0

    # +1 g en Z (8192 cuentas = 0x2000) y +65.5 cuentas ≈ +1 °/s en X (0x0041 = 65)
    bloque = [0x00, 0x00,  0x00, 0x00,  0x20, 0x00,   # AX AY AZ
              0x00, 0x00,                             # TEMP = 0 -> 36.53 °C
              0x00, 0x41,  0x00, 0x00,  0xFF, 0xBF]   # GX=65, GY=0, GZ=-65
    w, a, temp = grabar.convertir_bloque_imu(bloque, CFG["imu"])
    assert abs(a[2] - G) < 1e-9 and a[0] == 0.0 and a[1] == 0.0
    assert abs(w[0] - np.radians(65 / 65.5)) < 1e-12
    assert abs(w[2] + np.radians(65 / 65.5)) < 1e-12
    assert abs(temp - 36.53) < 1e-9


def test_grabacion_limpia_supera_f1():
    tmp = tempfile.mkdtemp()
    try:
        verdad = crear_grabacion(tmp)
        r_imu, r_cam, r_rel, r_rep, crit = analizar(tmp)

        assert r_imu["perdidas"] == 0 and r_cam["perdidas"] == 0
        assert abs(r_imu["frecuencia_hz"] - 200.0) < 0.01
        assert abs(r_cam["frecuencia_hz"] - 20.0) < 1e-6
        # σ por marca = 0.1 ms -> σ de Δt = 0.1·√2 ≈ 0.141 ms (diferencia de dos ruidos independientes)
        assert abs(r_imu["jitter_ms"] - 0.1 * np.sqrt(2)) < 0.01
        assert r_rel["reloj_ok"] and r_rel["imu_cubre_cam"]
        assert abs(r_rel["latencia_ms"]["media"] - 30.0) < 1e-6
        assert r_rep["eje_arriba"] == "+z"
        assert abs(r_rep["error_norma"]) < 0.01
        assert np.allclose(r_rep["sesgo_gir_dps"], verdad["sesgo_gir_dps"], atol=0.01)
        assert all(ok for _, ok in crit), crit
    finally:
        shutil.rmtree(tmp)


def test_detecta_perdidas():
    tmp = tempfile.mkdtemp()
    try:
        # 1 muestra suelta + 2 consecutivas de IMU; 1 frame de cámara
        crear_grabacion(tmp, indices_imu_perdidos=[3000, 6000, 6001], indices_cam_perdidos=[500])
        r_imu, r_cam, _, _, crit = analizar(tmp)
        assert r_imu["perdidas"] == 3, r_imu["perdidas"]
        assert r_imu["huecos"] == 2
        assert r_cam["perdidas"] == 1
        assert abs(r_imu["jitter_ms"] - 0.1 * np.sqrt(2)) < 0.01   # los huecos no inflan el jitter
        estado = dict(crit)
        assert not estado["frames perdidos ≤ 0"]
    finally:
        shutil.rmtree(tmp)


def test_detecta_relojes_distintos():
    tmp = tempfile.mkdtemp()
    try:
        # Simula marcar la llegada con CLOCK_REALTIME: offset de horas
        crear_grabacion(tmp, offset_reloj_llegada_ns=3600 * 10**9)
        _, _, r_rel, _, crit = analizar(tmp)
        assert not r_rel["reloj_ok"]
        assert not dict(crit)["reloj común verificado"]
    finally:
        shutil.rmtree(tmp)


def test_detecta_imu_que_no_cubre_camara():
    tmp = tempfile.mkdtemp()
    try:
        # La cámara arranca antes que la IMU
        crear_grabacion(tmp, retraso_cam_s=-1.0)
        _, _, r_rel, _, _ = analizar(tmp)
        assert not r_rel["imu_cubre_cam"]
    finally:
        shutil.rmtree(tmp)


def test_timestamp_ns_sin_perdida_de_precision():
    """Los ns no deben pasar por float64 al cargar."""
    tmp = tempfile.mkdtemp()
    try:
        crear_grabacion(tmp, duracion_imu_s=10.0, retraso_cam_s=1.0, duracion_cam_s=5.0)
        datos = verificar.cargar_grabacion(tmp)
        assert datos["t_imu"].dtype == np.int64 and datos["t_cam"].dtype == np.int64
        with open(os.path.join(tmp, "mav0", "imu0", "data.csv")) as f:
            f.readline()
            primero = int(f.readline().split(",")[0])
        assert datos["t_imu"][0] == primero
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} tests superados")
