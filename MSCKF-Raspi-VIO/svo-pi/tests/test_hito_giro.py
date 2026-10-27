#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_hito_giro.py — Tests del hito 1 (integrador de giroscopio, fuentes, red).

Se ejecuta en: [Windows PowerShell] / [WSL/Ubuntu] / [Raspberry Pi]
  python tests/test_hito_giro.py

Datos sintéticos con verdad conocida:
  - placa inclinada (roll 10°, pitch -5°) en reposo
  - giro de 90° alrededor del eje z de la IMU con perfil suave
    ω_z(τ) = A·sin²(π τ / T),  A = π/T  ->  ∫ω = A·T/2 = π/2
  - giroscopio con sesgo b_g y ruido; acelerómetro = R_w_iᵀ·(0,0,g) + ruido
"""

import os
import shutil
import socket
import sys
import tempfile
import time

import numpy as np
import yaml

RAIZ_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ_REPO, "vio"))
sys.path.insert(0, os.path.join(RAIZ_REPO, "apps"))

import so3                                     # noqa: E402
import protocolo                               # noqa: E402
from fuentes import fuente_grabacion_imu       # noqa: E402
from integrador_giro import IntegradorGiro, FASE_INTEGRANDO, FASE_INICIALIZANDO   # noqa: E402

with open(os.path.join(RAIZ_REPO, "config", "hito_giro.yaml")) as _f:
    CFG = yaml.safe_load(_f)

G = 9.80665
F_IMU = 200.0
ROLL0, PITCH0 = np.radians(10.0), np.radians(-5.0)
R0 = so3.rot_y(PITCH0) @ so3.rot_x(ROLL0)              # yaw 0
SESGO = np.radians([0.8, -0.5, 0.6])                    # rad/s


def angulo_giro(tau, T):
    """θ(τ) = ∫₀^τ A sin²(π s/T) ds con A = π/T."""
    A = np.pi / T
    tau = np.clip(tau, 0.0, T)
    return A * (tau / 2.0 - T * np.sin(2 * np.pi * tau / T) / (4 * np.pi))


def generar_imu(t_reposo=2.5, t_inicio_giro=None, T_giro=1.0, t_total=5.0,
                sigma_w_dps=0.05, sigma_a=0.02, semilla=0):
    """Devuelve (t_ns, w, a, R_final_verdad)."""
    if t_inicio_giro is None:
        t_inicio_giro = t_reposo + 0.5
    rng = np.random.default_rng(semilla)
    n = int(t_total * F_IMU)
    t_s = np.arange(n) / F_IMU
    t_ns = (10**12 + np.round(t_s * 1e9)).astype(np.int64)

    w = np.zeros((n, 3))
    a = np.zeros((n, 3))
    A = np.pi / T_giro
    for k in range(n):
        tau = t_s[k] - t_inicio_giro
        en_giro = 0.0 <= tau <= T_giro
        wz = A * np.sin(np.pi * tau / T_giro) ** 2 if en_giro else 0.0
        R = R0 @ so3.rot_z(angulo_giro(tau, T_giro))
        w[k] = np.array([0.0, 0.0, wz]) + SESGO + rng.normal(0, np.radians(sigma_w_dps), 3)
        a[k] = R.T @ np.array([0.0, 0.0, G]) + rng.normal(0, sigma_a, 3)
    R_final = R0 @ so3.rot_z(angulo_giro(t_total - t_inicio_giro, T_giro))
    return t_ns, w, a, R_final


def error_angular_deg(R_a, R_b):
    return np.degrees(np.linalg.norm(so3.log_so3(R_a.T @ R_b)))


# =====================================================================
def test_inicializacion_sesgo_e_inclinacion():
    t_ns, w, a, _ = generar_imu(t_total=2.6, t_inicio_giro=10.0)
    integ = IntegradorGiro(CFG)
    for k in range(len(t_ns)):
        integ.procesar(t_ns[k], w[k], a[k])
        if integ.fase == FASE_INTEGRANDO:
            break                                # comprobar justo al inicializar
    assert integ.fase == FASE_INTEGRANDO
    assert np.allclose(integ.b_g, SESGO, atol=np.radians(0.02))
    yaw, pitch, roll = so3.euler_zyx(integ.R_corr)
    assert abs(np.degrees(roll - ROLL0)) < 0.1
    assert abs(np.degrees(pitch - PITCH0)) < 0.1
    assert abs(yaw) < 1e-9                       # yaw = 0 por convenio


def test_giro_de_90_grados():
    t_ns, w, a, R_final = generar_imu()
    integ = IntegradorGiro(CFG)
    for k in range(len(t_ns)):
        integ.procesar(t_ns[k], w[k], a[k])

    err_corr = error_angular_deg(R_final, integ.R_corr)
    err_bruto = error_angular_deg(R_final, integ.R_bruto)

    # Tiempo integrado tras la inicialización (≈ 2.5 s): deriva esperada ≈ ‖b_g‖·t
    t_int = (t_ns[-1] - t_ns[int(CFG["inicializacion"]["t_reposo_s"] * F_IMU)]) / 1e9
    deriva_esperada = np.degrees(np.linalg.norm(SESGO)) * t_int

    assert err_corr < 0.1, err_corr
    assert abs(err_bruto - deriva_esperada) < 0.25 * deriva_esperada, (err_bruto, deriva_esperada)
    yaw, _, _ = so3.euler_zyx(integ.R_corr)
    # Girar 90° sobre el z de una placa inclinada no da exactamente yaw = 90°,
    # por eso se compara con la verdad en SO(3) y aquí solo se comprueba el orden
    assert 80.0 < np.degrees(yaw) < 100.0


def test_reinicia_si_se_mueve_al_inicializar():
    # El giro empieza a los 0.5 s, dentro del tramo de inicialización
    t_ns, w, a, R_final = generar_imu(t_inicio_giro=0.5, t_total=6.0)
    integ = IntegradorGiro(CFG)
    hubo_reinicio = False
    for k in range(len(t_ns)):
        integ.procesar(t_ns[k], w[k], a[k])
        if integ.motivo_reinicio:
            hubo_reinicio = True
    assert hubo_reinicio
    assert integ.fase == FASE_INTEGRANDO          # al quedarse quieto, inicializa
    assert integ.motivo_reinicio is None
    # Se inicializó tras el giro: la referencia es la orientación ya girada con yaw 0
    _, pitch, roll = so3.euler_zyx(integ.R_corr)
    _, pitch_v, roll_v = so3.euler_zyx(R_final)
    assert abs(np.degrees(roll - roll_v)) < 0.2 and abs(np.degrees(pitch - pitch_v)) < 0.2


def test_ignora_timestamps_no_crecientes():
    t_ns, w, a, _ = generar_imu(t_total=3.0, t_inicio_giro=10.0)
    integ = IntegradorGiro(CFG)
    for k in range(len(t_ns)):
        integ.procesar(t_ns[k], w[k], a[k])
    R_antes = integ.R_corr.copy()
    integ.procesar(t_ns[-1], np.array([1.0, 1.0, 1.0]), a[-1])        # repetido
    integ.procesar(t_ns[-1] - 5_000_000, np.array([1.0, 1.0, 1.0]), a[-1])  # hacia atrás
    assert np.allclose(integ.R_corr, R_antes)


def escribir_euroc(carpeta, t_ns, w, a):
    os.makedirs(os.path.join(carpeta, "mav0", "imu0"))
    with open(os.path.join(carpeta, "mav0", "imu0", "data.csv"), "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for k in range(len(t_ns)):
            f.write(f"{t_ns[k]},{w[k,0]:.9f},{w[k,1]:.9f},{w[k,2]:.9f},"
                    f"{a[k,0]:.6f},{a[k,1]:.6f},{a[k,2]:.6f}\n")


def test_fuente_grabacion_equivale_a_datos_directos():
    """Misma salida del integrador leyendo de memoria o de una grabación EuRoC."""
    t_ns, w, a, _ = generar_imu()
    tmp = tempfile.mkdtemp()
    try:
        escribir_euroc(tmp, t_ns, w, a)
        eventos = list(fuente_grabacion_imu(tmp))
        assert len(eventos) == len(t_ns)
        assert all(ev.t_ns == int(t) for ev, t in zip(eventos, t_ns))

        integ_mem = IntegradorGiro(CFG)
        integ_arch = IntegradorGiro(CFG)
        for k in range(len(t_ns)):
            integ_mem.procesar(t_ns[k], w[k], a[k])
        for ev in eventos:
            integ_arch.procesar(ev.t_ns, ev.w, ev.a)
        assert error_angular_deg(integ_mem.R_corr, integ_arch.R_corr) < 1e-4
    finally:
        shutil.rmtree(tmp)


def test_fuente_grabacion_tiempo_real():
    t_ns, w, a, _ = generar_imu(t_total=0.5)
    tmp = tempfile.mkdtemp()
    try:
        escribir_euroc(tmp, t_ns, w, a)
        t0 = time.monotonic()
        n = sum(1 for _ in fuente_grabacion_imu(tmp, tiempo_real=True))
        transcurrido = time.monotonic() - t0
        assert n == len(t_ns)
        assert 0.45 < transcurrido < 0.8, transcurrido   # dura lo que dicen los timestamps
    finally:
        shutil.rmtree(tmp)


def test_protocolo_ida_y_vuelta():
    R = so3.exp_so3([0.1, -0.2, 0.3])
    m = {"tipo": "orientacion", "seq": 7, "t_ns": 10**15 + 123, "fase": FASE_INICIALIZANDO,
         "progreso": 0.5, "aviso": None, "R_corr": R, "R_bruto": np.eye(3),
         "b_g": np.array([1e-3, 2e-3, -3e-3]), "rp_acc": np.array([0.1, -0.05])}
    d = protocolo.decodificar(protocolo.codificar(m))
    assert d["seq"] == 7 and d["t_ns"] == 10**15 + 123 and d["aviso"] is None
    assert np.allclose(d["R_corr"], R) and d["R_corr"].shape == (3, 3)
    assert np.allclose(d["b_g"], m["b_g"]) and np.allclose(d["rp_acc"], m["rp_acc"])


def test_receptor_udp_cuenta_perdidas():
    from pc_visor import Receptor

    # Puerto libre elegido por el sistema
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    puerto = s.getsockname()[1]
    s.close()

    rx = Receptor(puerto, ventana_s=60)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for seq in (0, 1, 3, 4):        # falta el 2
            m = {"tipo": "orientacion", "seq": seq, "t_ns": 10**12 + seq * 33_000_000,
                 "fase": FASE_INTEGRANDO, "progreso": 1.0, "aviso": None,
                 "R_corr": so3.rot_z(0.1 * seq), "R_bruto": np.eye(3),
                 "b_g": np.zeros(3), "rp_acc": np.zeros(2)}
            tx.sendto(protocolo.codificar(m), ("127.0.0.1", puerto))
        time.sleep(0.5)
        ultimo, _, recibidos, perdidos, hist = rx.instantanea()
        assert recibidos == 4 and perdidos == 1
        assert ultimo["seq"] == 4 and len(hist) == 4
        assert abs(hist[-1, 1] - 0.4) < 1e-9          # yaw de rot_z(0.4)
    finally:
        tx.close()
        rx.cerrar()


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} tests superados")
