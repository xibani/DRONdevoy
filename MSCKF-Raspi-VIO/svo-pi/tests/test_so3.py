#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_so3.py — Tests de vio/so3.py.

Se ejecuta en: [Windows PowerShell] / [WSL/Ubuntu] / [Raspberry Pi]
  python tests/test_so3.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vio"))
import so3   # noqa: E402

TOL = 1e-10


def es_rotacion(R):
    return np.allclose(R.T @ R, np.eye(3), atol=1e-12) and abs(np.linalg.det(R) - 1) < 1e-12


def test_hat_vee():
    v = np.array([0.3, -1.2, 2.0])
    u = np.array([1.0, 0.5, -0.7])
    assert np.allclose(so3.hat(v) @ u, np.cross(v, u))
    assert np.allclose(so3.vee(so3.hat(v)), v)
    assert np.allclose(so3.hat(v), -so3.hat(v).T)


def test_exp_casos_conocidos():
    assert np.allclose(so3.exp_so3(np.zeros(3)), np.eye(3))
    # Giro de 90° alrededor de z = rot_z(90°)
    assert np.allclose(so3.exp_so3([0, 0, np.pi / 2]), so3.rot_z(np.pi / 2), atol=TOL)
    assert np.allclose(so3.exp_so3([0.4, 0, 0]), so3.rot_x(0.4), atol=TOL)
    assert np.allclose(so3.exp_so3([0, -0.7, 0]), so3.rot_y(-0.7), atol=TOL)


def test_exp_log_ida_y_vuelta():
    rng = np.random.default_rng(1)
    for _ in range(500):
        eje = rng.normal(size=3)
        eje /= np.linalg.norm(eje)
        angulo = rng.uniform(0, np.pi - 1e-3)
        phi = eje * angulo
        R = so3.exp_so3(phi)
        assert es_rotacion(R)
        assert np.allclose(so3.log_so3(R), phi, atol=1e-9)


def test_angulos_pequenos_y_cercanos_a_pi():
    phi = np.array([1e-10, -2e-10, 3e-10])          # rama de Taylor
    assert np.allclose(so3.log_so3(so3.exp_so3(phi)), phi, atol=1e-15)
    phi = np.array([0.0, 0.0, np.pi - 1e-8])         # rama de θ ≈ π
    R = so3.exp_so3(phi)
    assert abs(np.linalg.norm(so3.log_so3(R)) - (np.pi - 1e-8)) < 1e-6
    assert np.allclose(so3.exp_so3(so3.log_so3(R)), R, atol=1e-6)


def test_ortonormalizar():
    rng = np.random.default_rng(2)
    R = so3.exp_so3([0.3, 0.2, -0.5])
    R_ruido = R + 1e-4 * rng.normal(size=(3, 3))
    R_proy = so3.ortonormalizar(R_ruido)
    assert es_rotacion(R_proy)
    assert np.linalg.norm(so3.log_so3(R.T @ R_proy)) < 1e-3


def test_euler_zyx():
    yaw, pitch, roll = np.radians([30.0, -10.0, 5.0])
    R = so3.rot_z(yaw) @ so3.rot_y(pitch) @ so3.rot_x(roll)
    assert np.allclose(so3.euler_zyx(R), (yaw, pitch, roll), atol=TOL)


def test_R_desde_gravedad():
    g = 9.80665
    roll, pitch = np.radians(12.0), np.radians(-7.0)
    R_verdad = so3.rot_y(pitch) @ so3.rot_x(roll)          # yaw = 0
    f_i = R_verdad.T @ np.array([0.0, 0.0, g])              # lo que mide el acelerómetro
    R = so3.R_desde_gravedad(f_i)
    assert es_rotacion(R)
    assert np.allclose(R @ (f_i / g), [0, 0, 1], atol=TOL)  # lleva "arriba" de la IMU a z del mundo
    assert np.allclose(R, R_verdad, atol=1e-12)
    yaw_est, _, _ = so3.euler_zyx(R)
    assert abs(yaw_est) < TOL

    # Caso degenerado: eje x de la IMU vertical
    R = so3.R_desde_gravedad([g, 0.0, 0.0])
    assert es_rotacion(R)
    assert np.allclose(R @ [1, 0, 0], [0, 0, 1], atol=TOL)


if __name__ == "__main__":
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} tests superados")
