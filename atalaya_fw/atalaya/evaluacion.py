"""
atalaya.evaluacion — los mismos resultados que las fases del curso, en un
informe reproducible.

Genera en el directorio de salida:
  * trayectoria.png     XY + altura, estimada vs GT (si hay) vs dead reckoning
  * filtro.png          bias, traza de P, d² del gating y tasa de aceptación
  * frontend.png        paralaje y (si hay GT) errores de rotación/dirección
  * est.tum / gt.tum    trayectorias en formato TUM (para `evo`)
  * resumen.txt         métricas numéricas
  * resultado.npz       (t, p_est, q_est_wxyz, bg, ba, ...) — mismo contrato
                        de claves que los .npz de las fases

Sobre las métricas: SE(3) solo tiene sentido si la estimación tiene escala y
marco absoluto (ESKF inicializado con GT). Con inicialización estática el yaw
y la posición son arbitrarios: la métrica honesta es Sim(3) o SE(3) tras
alinear, y el informe lo dice explícitamente en vez de esconderlo.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .geometria import Log, ate_rmse, save_tum


def _q_wxyz(poses):
    return Rot.from_matrix(poses[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]


def evaluar_contra_gt(seq, res):
    t = res["t"]
    p_est = res["pose"][:, :3, 3]
    p_gt, R_gt, _ = seq.gt_en(t)
    # Actitud TRAS alinear (Umeyama SE(3)): con inicialización estática el yaw
    # del marco del filtro es arbitrario y sin alinear la métrica no mide nada.
    from .geometria import align_umeyama
    _, R_al, _ = align_umeyama(p_gt, p_est, with_scale=False)
    err_ang = np.degrees([np.linalg.norm(Log(R_gt[i].T @ R_al
                                             @ res["pose"][i, :3, :3]))
                          for i in range(len(t))])
    return dict(
        ate_se3=ate_rmse(p_gt, p_est, with_scale=False),
        ate_sim3=ate_rmse(p_gt, p_est, with_scale=True),
        err_ang_mediana=float(np.median(err_ang)),
        p_gt=p_gt, err_ang=np.array(err_ang))


def informe(seq, res, dir_salida, res_dr=None, medidas=None, verbose=True):
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    t = res["t"]
    p = res["pose"][:, :3, 3]
    ev = evaluar_contra_gt(seq, res) if seq.tiene_gt else None

    lineas = [f"ATALAYA — informe de {seq.nombre}",
              f"ventana: [{t[0]:.2f}, {t[-1]:.2f}] s ({t[-1]-t[0]:.1f} s, "
              f"{len(t)} frames)",
              f"modo del filtro: {res.get('modo', '?')}"]

    # ---------------- trayectoria ------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(p[:, 0], p[:, 1], lw=1.2, label="ESKF")
    if res_dr is not None:
        pdr = res_dr["pose"][:, :3, 3]
        ax[0].plot(pdr[:, 0], pdr[:, 1], lw=0.9, ls="--", c="0.6",
                   label="IMU sola (dead reckoning)")
        ax[1].plot(res_dr["t"], pdr[:, 2], lw=0.9, ls="--", c="0.6")
    if ev is not None:
        ax[0].plot(ev["p_gt"][:, 0], ev["p_gt"][:, 1], lw=1.0, c="k",
                   alpha=0.7, label="GT")
        ax[1].plot(t, ev["p_gt"][:, 2], lw=1.0, c="k", alpha=0.7)
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")
    ax[0].axis("equal"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(t, p[:, 2], lw=1.2)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("z [m]"); ax[1].grid(alpha=0.3)
    titulo = "Trayectoria"
    if ev is not None:
        titulo += (f"   ATE SE(3) = {ev['ate_se3']:.3f} m   "
                   f"Sim(3) = {ev['ate_sim3']:.3f} m")
    plt.suptitle(titulo, y=1.0)
    plt.tight_layout(); plt.savefig(dir_salida / "trayectoria.png", dpi=130)
    plt.close(fig)

    # ---------------- salud del filtro -------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(t, res["bg"], lw=0.9)
    ax[0].set_ylabel("b_g [rad/s]"); ax[0].legend(["x", "y", "z"], fontsize=7)
    ax[1].plot(t, res["ba"], lw=0.9)
    ax[1].set_ylabel("b_a [m/s²]"); ax[1].legend(["x", "y", "z"], fontsize=7)
    if len(res.get("m_t", [])):
        d2 = np.asarray(res["m_d2"], float)
        okm = np.asarray(res["m_ok"], bool)
        ax[2].plot(res["m_t"], d2, ".", ms=3, alpha=0.6)
        ax[2].plot(np.asarray(res["m_t"])[~okm], d2[~okm], "x", ms=5, c="r",
                   label="rechazadas χ²")
        ax[2].axhline(16.8, c="r", ls="--", lw=0.8)
        ax[2].axhline(5.35, c="g", ls=":", lw=0.8, label="mediana χ²(6)")
        ax[2].set_ylabel("d² de la medida"); ax[2].set_ylim(0, 30)
        ax[2].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("t [s]"); a.grid(alpha=0.3)
    plt.suptitle("Salud del filtro", y=1.02)
    plt.tight_layout(); plt.savefig(dir_salida / "filtro.png", dpi=130)
    plt.close(fig)

    # ---------------- front-end --------------------------------------------
    if medidas:
        con_gt = "err_rot" in medidas[0]
        n_pan = 3 if con_gt else 1
        fig, ax = plt.subplots(1, n_pan, figsize=(4.7 * n_pan, 3.6), squeeze=False)
        ax = ax[0]
        t_m = [seq.t_cam[m["k"]] for m in medidas]
        ax[0].plot(t_m, [m["paralaje"] for m in medidas], lw=0.8)
        ax[0].set_ylabel("paralaje derotado [px]")
        if con_gt:
            ax[1].plot(t_m, [m["err_rot"] for m in medidas], lw=0.8, c="C1")
            ax[1].set_ylabel("error de rotación [°] (vs GT)")
            ax[2].plot(t_m, [m["err_dir"] for m in medidas], lw=0.8, c="C2")
            ax[2].set_ylabel("error de dirección [°] (vs GT)")
        for a in ax:
            a.set_xlabel("t [s]"); a.grid(alpha=0.3)
        plt.suptitle("Calidad del front-end por medida", y=1.02)
        plt.tight_layout(); plt.savefig(dir_salida / "frontend.png", dpi=130)
        plt.close(fig)

    # ---------------- TUM + npz --------------------------------------------
    save_tum(dir_salida / "est.tum", t, res["pose"], t_offset_ns=seq.t0_ns)
    if seq.tiene_gt:
        p_gt, R_gt, _ = seq.gt_en(t)
        q_gt = Rot.from_matrix(R_gt).as_quat()
        save_tum(dir_salida / "gt.tum", t, np.hstack([p_gt, q_gt]),
                 t_offset_ns=seq.t0_ns)

    np.savez_compressed(
        dir_salida / "resultado.npz",
        modo=res.get("modo", ""), t=t,
        p_est=p, q_est_wxyz=_q_wxyz(res["pose"]),
        bg=res["bg"], ba=res["ba"], trP=res["trP"],
        m_t=np.asarray(res.get("m_t", [])), m_d2=np.asarray(res.get("m_d2", [])),
        m_ok=np.asarray(res.get("m_ok", [])), m_s=np.asarray(res.get("m_s", [])),
        m_par=np.asarray(res.get("m_par", [])),
        T0_ns=np.int64(seq.t0_ns),
        **({"ate_se3": ev["ate_se3"], "ate_sim3": ev["ate_sim3"]} if ev else {}))

    # ---------------- resumen ----------------------------------------------
    n_med = res.get("n_medidas", 0)
    if n_med:
        tasa = 100.0 * res.get("n_aceptadas", 0) / n_med
        d2m = float(np.nanmedian(res["m_d2"]))
        lineas += [f"medidas visuales: {n_med}   aceptadas {tasa:.1f} % "
                   "(objetivo > 80 %)",
                   f"d² mediana: {d2m:.2f}   (χ²(6) mediana = 5.35; "
                   ">>5 jacobianos/ruido incoherentes, <<5 Rm pesimista)"]
    if ev is not None:
        lineas += [f"ATE SE(3):  {ev['ate_se3']:.3f} m",
                   f"ATE Sim(3): {ev['ate_sim3']:.3f} m",
                   f"error de actitud tras alinear (mediana): "
                   f"{ev['err_ang_mediana']:.2f}°"]
        if res_dr is not None and seq.tiene_gt:
            p_gt_dr, _, _ = seq.gt_en(res_dr["t"])
            ate_dr = ate_rmse(p_gt_dr, res_dr["pose"][:, :3, 3], with_scale=False)
            lineas += [f"IMU sola (misma ventana): ATE SE(3) = {ate_dr:.2f} m   "
                       f"-> mejora {ate_dr/max(ev['ate_se3'], 1e-9):.0f}x"]
        if str(res.get("init_tipo", "")) == "estatica":
            lineas += ["NOTA: inicialización estática -> yaw y origen "
                       "arbitrarios; la métrica comparable es la alineada "
                       "(ambas ATE ya alinean con Umeyama)."]
    else:
        lineas += ["sin GT: no hay ATE. Diagnósticos disponibles: tasa de "
                   "aceptación, d², continuidad de bias y consistencia de trP."]

    (dir_salida / "resumen.txt").write_text("\n".join(lineas) + "\n")
    if verbose:
        print("\n".join(lineas))
        print(f"\ninforme escrito en {dir_salida}/")
    return ev
