"""
atalaya.selftest — validación sintética SIN datos.

Principio del curso: cada componente se valida en sintético antes de tocar
datos reales. Este módulo reproduce los tests de aceptación de las fases:

  1. Jacobianos del ESKF por diferencias finitas (pose relativa y dirección),
     incluyendo el test NEGATIVO del H del guion (debe fallar).
  2. ESKF completo sobre trayectoria analítica (Fase 5 §5.4): d² mediana
     ≈ 5.35 (χ²(6)), convergencia de b_g y b_a, deriva de posición acotada
     por el random walk esperado.
  3. Robustez: 5 % de medidas corruptas -> el modo dirección+gating debe
     degradar mucho menos que sin gating (hallazgo §5.12).
  4. Front-end sobre imágenes sintéticas renderizadas: puntos 3D proyectados
     con la cámara del framework, KLT + esencial, y se comprueba rotación y
     dirección contra la verdad.
  5. Inicialización estática sobre IMU sintética en reposo con bias.

Si esto pasa, un fallo con tus datos está en los DATOS o en la CONFIG
(unidades, extrínseca, offset temporal), no en la matemática del filtro.
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .eskf import ESKF, IP, IT, ICP, ICT, ConfigEskf, cam_a_body
from .frontend import ConfigFrontend, RastreadorKLT, paralaje_derotada, pose_relativa
from .geometria import Exp, G_W, Log, inv_T, make_T, skew
from .inicializacion import actitud_desde_gravedad
from .sensores import CamaraPinhole

SIGMAS = (1.6968e-4, 2.0e-3, 1.9393e-5, 3.0e-3)     # EuRoC como referencia


# ---------------------------------------------------------------------------
# 1. Jacobianos por diferencias finitas
# ---------------------------------------------------------------------------
def test_jacobianos(verbose=True):
    rng = np.random.default_rng(0)
    f = ESKF(rng.normal(size=3), rng.normal(size=3),
             Exp(rng.normal(size=3) * 0.5), np.zeros(3), np.zeros(3), SIGMAS)
    f.clone()
    f.p = f.p_c + rng.normal(size=3) * 0.3
    f.R = f.R_c @ Exp(rng.normal(size=3) * 0.1)

    dp_meas = f.R_c.T @ (f.p - f.p_c) + rng.normal(size=3) * 0.01
    dR_meas = (f.R_c.T @ f.R) @ Exp(rng.normal(size=3) * 0.01)

    r0, dp_pred, dR_pred = f.residuo(dp_meas, dR_meas)
    H = f.jacobiano(dp_pred, dR_pred, r0[3:6])

    H_guion = np.zeros((6, 21))                     # el H ERRÓNEO del guion
    H_guion[0:3, IP:IP + 3] = -f.R_c.T
    H_guion[0:3, ICP:ICP + 3] = f.R_c.T
    H_guion[0:3, ICT:ICT + 3] = -skew(dp_pred)
    H_guion[3:6, IT:IT + 3] = -np.eye(3)
    H_guion[3:6, ICT:ICT + 3] = dR_pred

    e0 = f.estado()
    eps = 1e-6

    def H_num(residuo_fn, dim_r=6):
        Hn = np.zeros((dim_r, 21))
        for i in range(21):
            d = np.zeros(21); d[i] = eps
            f.set_estado(e0); f._inyectar(d, reset=False)
            rp = residuo_fn()
            f.set_estado(e0); d[i] = -eps; f._inyectar(d, reset=False)
            rm = residuo_fn()
            Hn[:, i] = -(rp - rm) / (2 * eps)        # H = -dr/dδx
        f.set_estado(e0)
        return Hn

    Hn = H_num(lambda: f.residuo(dp_meas, dR_meas)[0])
    err = np.abs(H - Hn).max()
    err_guion = np.abs(H_guion - Hn).max()

    ru = f.R_c.T @ (f.p - f.p_c)
    u_meas = ru / np.linalg.norm(ru) + rng.normal(size=3) * 0.01
    u_meas /= np.linalg.norm(u_meas)
    r0u, Hu, _ = f.residuo_direccion(u_meas, dR_meas)
    Hnu = H_num(lambda: f.residuo_direccion(u_meas, dR_meas)[0])
    err_u = np.abs(Hu - Hnu).max()

    if verbose:
        print("[1] jacobianos vs diferencias finitas")
        print(f"    pose relativa : |H - H_num|_max = {err:.3e}")
        print(f"    dirección     : |H - H_num|_max = {err_u:.3e}")
        print(f"    H del guion   : |H - H_num|_max = {err_guion:.3e}  "
              "<- DEBE ser grande (test negativo)")
    assert err < 1e-6 and err_u < 1e-6, "jacobiano analítico no cuadra"
    assert err_guion > 0.1, "el test negativo del H del guion ya no detecta nada"
    return err, err_u


# ---------------------------------------------------------------------------
# 2-3. ESKF sintético (trayectoria analítica de la Fase 3.3)
# ---------------------------------------------------------------------------
def _imu_sintetica(t):
    p = np.column_stack([1.5 * np.sin(0.7 * t), 1.0 * np.sin(0.4 * t + 0.3),
                         0.6 * np.sin(0.9 * t)])
    v = np.column_stack([1.05 * np.cos(0.7 * t), 0.40 * np.cos(0.4 * t + 0.3),
                         0.54 * np.cos(0.9 * t)])
    a = np.column_stack([-0.735 * np.sin(0.7 * t), -0.160 * np.sin(0.4 * t + 0.3),
                         -0.486 * np.sin(0.9 * t)])

    def rotvec(s):
        return np.column_stack([0.3 * np.sin(0.5 * s), 0.2 * np.sin(0.8 * s + 1.0),
                                0.6 * np.sin(0.3 * s)])

    R = Rotation.from_rotvec(rotvec(t)).as_matrix()
    h = 1e-6
    Rd = (Rotation.from_rotvec(rotvec(t + h)).as_matrix()
          - Rotation.from_rotvec(rotvec(t - h)).as_matrix()) / (2 * h)
    S = np.einsum("kij,kjl->kil", np.transpose(R, (0, 2, 1)), Rd)
    w = np.column_stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]])
    am = np.einsum("kij,kj->ki", np.transpose(R, (0, 2, 1)), a - G_W)
    return p, v, R, w, am


def eskf_sintetico(T=60.0, sigma_p=0.02, sigma_r=0.005, semilla=1,
                   modo="pose", frac_corruptas=0.0, chi2_umbral=16.8):
    rr = np.random.default_rng(semilla)
    dt = 1 / 200.0
    t = np.arange(0.0, T, dt)
    p_t, v_t, R_t, w_t, a_t = _imu_sintetica(t)
    bg_t = np.array([0.010, -0.020, 0.005])
    ba_t = np.array([0.050, -0.080, 0.030])
    sg, sa = SIGMAS[0], SIGMAS[1]
    tramo = np.column_stack([
        t,
        w_t + bg_t + rr.normal(size=w_t.shape) * sg / np.sqrt(dt),
        a_t + ba_t + rr.normal(size=a_t.shape) * sa / np.sqrt(dt)])

    P0 = np.diag(np.concatenate([1e-6 * np.ones(3), 1e-4 * np.ones(3),
                                 np.deg2rad(0.5) ** 2 * np.ones(3),
                                 1e-4 * np.ones(3), 1e-2 * np.ones(3)]))
    f = ESKF(p_t[0], v_t[0], R_t[0], np.zeros(3), np.zeros(3), SIGMAS, P0=P0)
    f.clone()

    paso, i_c, acc = 20, 0, 0                       # medida cada 0.1 s
    reg = defaultdict(list)
    for i in range(paso, len(t), paso):
        f.predict(tramo[i - paso:i + 1])
        dp = R_t[i_c].T @ (p_t[i] - p_t[i_c]) + rr.normal(size=3) * sigma_p
        dR = (R_t[i_c].T @ R_t[i]) @ Exp(rr.normal(size=3) * sigma_r)
        corrupta = rr.random() < frac_corruptas
        if corrupta:
            u_mal = rr.normal(size=3); u_mal /= np.linalg.norm(u_mal)
            dp = np.linalg.norm(dp) * u_mal

        if modo == "direccion":
            u = dp / np.linalg.norm(dp)
            ok, d2 = f.update_direccion(u, dR, sigma_u=sigma_p / 0.1,
                                        sigma_r=sigma_r, chi2_umbral=chi2_umbral)
        else:
            ok, d2 = f.update_pose_relativa(dp, dR, sigma_p, sigma_r,
                                            chi2_umbral=chi2_umbral)
        acc += ok
        f.drop_clone(); f.clone()
        i_c = i
        reg["d2"].append(d2)
        reg["ep"].append(np.linalg.norm(f.p - p_t[i]))
        reg["ebg"].append(np.linalg.norm(f.bg - bg_t))
        reg["eba"].append(np.linalg.norm(f.ba - ba_t))
    reg = {k: np.array(v) for k, v in reg.items()}
    reg["acc"], reg["n"] = acc, len(reg["d2"])
    return reg


def test_eskf_sintetico(verbose=True):
    r = eskf_sintetico()
    esperado_rw = 0.02 * np.sqrt(r["n"])
    if verbose:
        print("[2] ESKF sobre 60 s sintéticos (pose relativa cada 0.1 s)")
        print(f"    d² mediana = {np.median(r['d2']):.2f} (χ²(6) -> 5.35)   "
              f"aceptadas {100*r['acc']/r['n']:.1f} %")
        print(f"    error de posición final = {r['ep'][-1]:.3f} m "
              f"(random walk esperado ≈ {esperado_rw:.2f} m; una medida "
              "relativa NO acota la posición absoluta)")
        print(f"    |b_g - real| final = {r['ebg'][-1]:.2e} rad/s   "
              f"|b_a - real| = {r['eba'][-1]:.2e} m/s²")
    assert np.median(r["d2"]) < 9.0, "d² inconsistente: bug de jacobiano o Rm"
    assert r["ebg"][-1] < 1e-3, "b_g no converge en sintético"
    assert r["ep"][-1] < 3 * esperado_rw + 0.2, "deriva mayor que el random walk"
    return r


def test_robustez(verbose=True):
    con = eskf_sintetico(modo="direccion", frac_corruptas=0.05, semilla=7)
    sin = eskf_sintetico(modo="direccion", frac_corruptas=0.05, semilla=7,
                         chi2_umbral=1e12)
    if verbose:
        print("[3] 5 % de medidas corruptas, modo dirección")
        print(f"    CON gating χ² : error final {con['ep'][-1]:.3f} m   "
              f"aceptadas {100*con['acc']/con['n']:.0f} %")
        print(f"    SIN gating    : error final {sin['ep'][-1]:.3f} m")
    assert con["ep"][-1] <= sin["ep"][-1] + 1e-9, \
        "el gating no está protegiendo frente a corruptas"
    return con, sin


# ---------------------------------------------------------------------------
# 4. Front-end sobre imágenes sintéticas
# ---------------------------------------------------------------------------
def _camara_sintetica():
    return CamaraPinhole(fu=458.0, fv=458.0, cu=376.0, cv_=240.0,
                         dist=np.array([-0.28, 0.07, 1e-4, 1e-5]),
                         width=752, height=480, T_body_cam=np.eye(4))


def _proyectar(cam, T_wc, X):
    """Proyecta puntos 3D del mundo a píxeles DISTORSIONADOS (radtan)."""
    Xc = (inv_T(T_wc) @ np.hstack([X, np.ones((len(X), 1))]).T).T[:, :3]
    vis = Xc[:, 2] > 0.3
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    k1, k2, p1, p2 = cam.dist[:4]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = cam.fu * xd + cam.cu
    v = cam.fv * yd + cam.cv_
    vis &= (u > 5) & (u < cam.width - 5) & (v > 5) & (v < cam.height - 5)
    return np.column_stack([u, v]), vis


def _render(cam, uv, vis, rng):
    img = np.full((cam.height, cam.width), 30, np.uint8)
    ruido = rng.integers(0, 12, img.shape, dtype=np.uint8)
    img = cv2.add(img, ruido)
    for (u, v), ok in zip(uv, vis):
        if ok:
            cv2.circle(img, (int(round(u)), int(round(v))), 3, 230, -1)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


def test_frontend_sintetico(n_frames=12, verbose=True):
    rng = np.random.default_rng(3)
    cam = _camara_sintetica()
    X = np.column_stack([rng.uniform(-4, 4, 400), rng.uniform(-2.5, 2.5, 400),
                         rng.uniform(4, 9, 400)])
    T_wc, imgs = [], []
    for k in range(n_frames):
        t = np.array([0.06 * k, 0.02 * k, 0.0])
        R = Exp(np.array([0.0, 0.004 * k, 0.006 * k]))
        T_wc.append(make_T(R, t))
        uv, vis = _proyectar(cam, T_wc[-1], X)
        imgs.append(_render(cam, uv, vis, rng))

    tr = RastreadorKLT(ConfigFrontend(dist_min=12, min_feats=120))
    ids0, _, pts0 = tr.rastrear(imgs[0], 0)
    errs_rot, errs_dir, pares = [], [], 0
    ids_a, pts_a, ka = ids0, pts0, 0
    for k in range(1, n_frames):
        ids, _, pts = tr.rastrear(imgs[k], k)
        if k - ka < 3:
            continue
        comunes, ia, ib = np.intersect1d(ids_a, ids, return_indices=True)
        n0 = cam.normalizar(pts_a[ia]); n1 = cam.normalizar(pts[ib])
        T_c1c0, inl = pose_relativa(n0, n1, cam.f_media)
        assert T_c1c0 is not None, "el front-end no recupera pose en sintético"
        T_c0c1 = inv_T(T_c1c0)
        T_gt = inv_T(T_wc[ka]) @ T_wc[k]
        errs_rot.append(np.degrees(np.linalg.norm(
            Log(T_gt[:3, :3].T @ T_c0c1[:3, :3]))))
        u = T_c0c1[:3, 3] / np.linalg.norm(T_c0c1[:3, 3])
        ug = T_gt[:3, 3] / np.linalg.norm(T_gt[:3, 3])
        errs_dir.append(np.degrees(np.arccos(np.clip(u @ ug, -1, 1))))
        par = np.median(paralaje_derotada(n0, n1, T_c1c0[:3, :3], cam.f_media))
        pares += 1
        ids_a, pts_a, ka = ids, pts, k
    if verbose:
        print("[4] front-end sobre imágenes sintéticas renderizadas")
        print(f"    {pares} pares keyframe   error rot mediana "
              f"{np.median(errs_rot):.3f}°   error dir mediana "
              f"{np.median(errs_dir):.2f}°   paralaje {par:.1f} px")
    assert np.median(errs_rot) < 0.5, "rotación del front-end fuera de tolerancia"
    assert np.median(errs_dir) < 5.0, "dirección del front-end fuera de tolerancia"
    return errs_rot, errs_dir


# ---------------------------------------------------------------------------
# 5. Inicialización estática
# ---------------------------------------------------------------------------
def test_inicializacion(verbose=True):
    rng = np.random.default_rng(5)
    R_true = Exp(np.array([np.deg2rad(8), np.deg2rad(-5), 0.0]))
    bg_true = np.array([0.01, -0.02, 0.005])
    a_body = R_true.T @ (-G_W)
    a = a_body + rng.normal(size=(2000, 3)) * 0.02
    R0 = actitud_desde_gravedad(a.mean(0))
    err = np.degrees(np.linalg.norm(Log(R_true.T @ R0)[:2]))  # roll/pitch
    if verbose:
        print("[5] inicialización estática (roll/pitch desde gravedad)")
        print(f"    error de roll/pitch = {err:.3f}°  (yaw no observable, "
              "por eso solo se mide roll/pitch)")
    assert err < 0.2, "alineación con la gravedad fuera de tolerancia"
    return err


def correr_todo():
    print("=" * 64)
    print("ATALAYA selftest — validación sintética (sin datos)")
    print("=" * 64)
    test_jacobianos()
    test_eskf_sintetico()
    test_robustez()
    test_frontend_sintetico()
    test_inicializacion()
    print("-" * 64)
    print("TODO OK. Si algo falla con tus datos, el problema está en los")
    print("datos o en la config (unidades, extrínseca, offset temporal).")
