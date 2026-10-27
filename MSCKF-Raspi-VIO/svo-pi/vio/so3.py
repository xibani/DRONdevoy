#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vio/so3.py — Rotaciones: el grupo SO(3), su álgebra so(3) y utilidades.

Convenciones del proyecto
-------------------------
  - R_a_b ∈ SO(3) lleva vectores del marco b al marco a:  v_a = R_a_b · v_b
  - R_w_i: orientación de la IMU (marco i, "body") en el mundo (marco w).
    Sus COLUMNAS son los ejes x, y, z de la IMU expresados en el mundo.
  - Mundo con z hacia arriba (opuesto a la gravedad).

Notación
--------
  SO(3):  matrices 3×3 con Rᵀ·R = I y det(R) = +1 (rotaciones).
  so(3):  matrices 3×3 antisimétricas. Cada una corresponde a un vector
          φ ∈ ℝ³ a través de hat(φ) = [φ]×. El vector φ es un "vector de
          rotación": dirección = eje de giro, norma = ángulo [rad].
  Exp:    ℝ³ → SO(3), φ ↦ exp([φ]×)  (fórmula de Rodrigues).
  Log:    SO(3) → ℝ³, inversa de Exp para ángulos en [0, π).

Es la parte rotacional del mapa exponencial de la Ec. 3 del paper de SVO
(allí ξ = (ω, ν) y exp: se(3) → SE(3); aquí solo ω y SO(3)).
"""

import numpy as np

EPS_ANGULO = 1e-8      # por debajo, se usan desarrollos de Taylor


def hat(v):
    """[v]× : matriz antisimétrica tal que hat(v) @ u = v × u."""
    return np.array([[0.0,   -v[2],  v[1]],
                     [v[2],   0.0,  -v[0]],
                     [-v[1],  v[0],  0.0]])


def vee(M):
    """Inversa de hat: de matriz antisimétrica a vector."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def exp_so3(phi):
    """Exp(φ) por Rodrigues:  R = I + sin θ/θ · [φ]× + (1 - cos θ)/θ² · [φ]×²,  θ = ‖φ‖.

    Para θ → 0 los cocientes son indeterminados; se usan sus desarrollos:
      sin θ/θ ≈ 1 - θ²/6,   (1 - cos θ)/θ² ≈ 1/2 - θ²/24
    """
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.linalg.norm(phi)
    K = hat(phi)
    if theta < EPS_ANGULO:
        a = 1.0 - theta**2 / 6.0
        b = 0.5 - theta**2 / 24.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta**2
    return np.eye(3) + a * K + b * (K @ K)


def log_so3(R):
    """Log(R) = φ. Ángulo por la traza: cos θ = (tr R - 1)/2.

    Casos:
      θ ≈ 0: φ ≈ vee(R - Rᵀ)/2
      θ ≈ π: sin θ ≈ 0 y la fórmula general se indetermina; el eje se saca
             de R + I = 2·n·nᵀ (columna de mayor norma).
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < EPS_ANGULO:
        return vee(R - R.T) / 2.0

    if np.pi - theta < 1e-6:
        B = (R + np.eye(3)) / 2.0                  # = n·nᵀ
        j = int(np.argmax(np.diag(B)))
        n = B[:, j] / np.sqrt(B[j, j])
        return theta * n

    return theta / (2.0 * np.sin(theta)) * vee(R - R.T)


def ortonormalizar(R):
    """Proyección de R sobre SO(3) (la rotación más cercana en norma de Frobenius).

    Al encadenar miles de productos R ← R·Exp(·), el redondeo en coma
    flotante hace que R deje de ser exactamente ortonormal. Con la SVD
    R = U Σ Vᵀ, la rotación más cercana es U·Vᵀ (corrigiendo el signo
    del determinante si hiciera falta).
    """
    U, _, Vt = np.linalg.svd(R)
    R_proy = U @ Vt
    if np.linalg.det(R_proy) < 0:
        U[:, 2] = -U[:, 2]
        R_proy = U @ Vt
    return R_proy


def euler_zyx(R):
    """Ángulos (guiñada, cabeceo, alabeo) = (yaw, pitch, roll) [rad] de R_w_i.

    Convención ZYX intrínseca: R_w_i = Rz(yaw) · Ry(pitch) · Rx(roll).
    Solo para mostrar números al usuario; los cálculos se hacen con R.
    Singular en pitch = ±90° (bloqueo de cardán).
    """
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def R_desde_gravedad(a_i):
    """Orientación inicial R_w_i a partir de una medida de acelerómetro en reposo.

    En reposo el acelerómetro mide la fuerza específica f_i = R_w_iᵀ · (0, 0, g):
    un vector que apunta hacia ARRIBA, expresado en ejes de la IMU. Por tanto
    z_i = f_i / ‖f_i‖ es el eje z del mundo visto desde la IMU.

    Se construyen los ejes del mundo en coordenadas de la IMU:
      z = f/‖f‖
      x = proyección del eje x de la IMU sobre el plano ⟂ z   (fija yaw = 0)
      y = z × x
    R_i_w = [x y z] (columnas) y R_w_i = R_i_wᵀ.

    El yaw es inobservable con la gravedad (girar alrededor de la vertical no
    cambia f_i); se elige yaw = 0 por convenio. Solo roll y pitch son reales.
    """
    z = np.asarray(a_i, dtype=np.float64)
    z = z / np.linalg.norm(z)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(z @ ref) > 0.9:                 # x de la IMU casi vertical: usar y
        ref = np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)

    R_i_w = np.column_stack([x, y, z])
    return R_i_w.T
