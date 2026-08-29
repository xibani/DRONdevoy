"""
atalaya.eskf — ESKF loosely-coupled (IMU predice, visión corrige).

Portado de la Fase 5 del curso con sus TRES correcciones al guion, todas
verificadas por diferencias finitas y test sintético (reproducibles con
`atalaya selftest`):

(1) H = +∂h/∂δx, SIN negar. El razonamiento "r = meas - pred, luego H lleva
    un menos" es circular: la negación ya está dentro del residuo. Con el
    signo del guion el filtro se va a ~3850 m con 2 % de aceptación.
(2) La medida se construye contra el KEYFRAME (el clon), nunca contra el
    frame anterior: comparar intervalos distintos es un sesgo de escala que
    ningún gating detecta.
(3) La escala multiplica SOLO la traslación de cámara:
        Δp_b = R_bc · (s·t̂_c) + (I - ΔR_b) · p_bc
    El brazo de palanca es geometría fija de la extrínseca y no lleva escala.

Y el hallazgo de §5.12: en modo "escala", el estimador de escala ABSORBE los
outliers (proyecta la predicción IMU sobre la dirección visual corrupta) y el
gating χ² queda neutralizado de facto. El modo "direccion" no tiene dónde
esconderlos y el χ² sí dispara: con 5 % de medidas corruptas, dirección+gating
bajó el ATE de 0.766 m a 0.060 m. Por eso el modo por defecto es "direccion".

Estado: [δp, δv, δθ, δb_g, δb_a] + clon (δp_c, δθ_c). Perturbación derecha
R = R̄·Exp(δθ). Joseph en el update. P simetrizada en cada paso.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .geometria import Exp, G_W, Log, jac_izq_inv, make_T, skew

IP, IV, IT, IBG, IBA = 0, 3, 6, 9, 12       # bloque IMU del estado de error
ICP, ICT = 15, 18                            # bloque del clon


class ESKF:
    """Error-state KF de 15 estados + 1 clon de pose (6), Joseph y gating χ²."""

    def __init__(self, p0, v0, R0, bg0, ba0, sigmas,
                 P0=None, factor_q=1.0, joseph=True, reset_error=True,
                 correlacion_clon=True):
        self.p = np.array(p0, dtype=float)
        self.v = np.array(v0, dtype=float)
        self.R = np.array(R0, dtype=float)
        self.bg = np.array(bg0, dtype=float)
        self.ba = np.array(ba0, dtype=float)

        self.P = P0.copy() if P0 is not None else np.diag(np.concatenate([
            1e-4 * np.ones(3), 1e-2 * np.ones(3),
            (np.deg2rad(1.0) ** 2) * np.ones(3),
            1e-6 * np.ones(3), 1e-4 * np.ones(3)]))

        sg, sa, sbg, sba = sigmas
        f = float(factor_q)
        self.Qc = np.diag(np.concatenate([
            (f * sg) ** 2 * np.ones(3), (f * sa) ** 2 * np.ones(3),
            (f * sbg) ** 2 * np.ones(3), (f * sba) ** 2 * np.ones(3)]))

        self.joseph = joseph
        self.reset_error = reset_error
        self.correlacion_clon = correlacion_clon

        self.n_clone = 0
        self.p_c = None
        self.R_c = None

    # ---------------- predicción -----------------------------------------
    def predict(self, tramo):
        """tramo: (M,7) [t, w(3), a(3)] cubriendo el intervalo completo."""
        for k in range(len(tramo) - 1):
            dt = tramo[k + 1, 0] - tramo[k, 0]
            if dt <= 0:
                continue
            w0 = tramo[k, 1:4] - self.bg
            w1 = tramo[k + 1, 1:4] - self.bg
            a0 = tramo[k, 4:7] - self.ba
            a1 = tramo[k + 1, 4:7] - self.ba
            self._paso(0.5 * (w0 + w1), a0, a1, dt)

    def _paso(self, w, a0, a1, dt):
        R_old = self.R
        R_new = R_old @ Exp(w * dt)
        a_w = 0.5 * (R_old @ a0 + R_new @ a1) + G_W       # midpoint (Fase 3)

        self.p = self.p + self.v * dt + 0.5 * a_w * dt ** 2
        self.v = self.v + a_w * dt
        self.R = R_new

        n = 15
        F = np.zeros((n, n))
        F[IP:IP + 3, IV:IV + 3] = np.eye(3)
        F[IV:IV + 3, IT:IT + 3] = -R_old @ skew(a0)
        F[IV:IV + 3, IBA:IBA + 3] = -R_old
        F[IT:IT + 3, IT:IT + 3] = -skew(w)
        F[IT:IT + 3, IBG:IBG + 3] = -np.eye(3)

        G = np.zeros((n, 12))
        G[IV:IV + 3, 3:6] = -R_old
        G[IT:IT + 3, 0:3] = -np.eye(3)
        G[IBG:IBG + 3, 6:9] = np.eye(3)
        G[IBA:IBA + 3, 9:12] = np.eye(3)

        Phi = np.eye(n) + F * dt + 0.5 * (F @ F) * dt ** 2       # 2º orden
        GQG = G @ self.Qc @ G.T
        Qd = 0.5 * dt * (Phi @ GQG @ Phi.T + GQG)                # trapecio

        P = self.P
        P[:n, :n] = Phi @ P[:n, :n] @ Phi.T + Qd
        if self.n_clone and self.correlacion_clon:
            P[:n, n:] = Phi @ P[:n, n:]
            P[n:, :n] = P[:n, n:].T
        self.P = 0.5 * (P + P.T)                                 # SIEMPRE

    # ---------------- stochastic cloning ---------------------------------
    def clone(self):
        self.p_c, self.R_c = self.p.copy(), self.R.copy()
        A = np.zeros((21, 15))
        A[:15, :15] = np.eye(15)
        A[ICP:ICP + 3, IP:IP + 3] = np.eye(3)
        A[ICT:ICT + 3, IT:IT + 3] = np.eye(3)
        P21 = A @ self.P[:15, :15] @ A.T
        self.P = 0.5 * (P21 + P21.T)
        self.n_clone = 1

    def drop_clone(self):
        """Marginalizar un bloque gaussiano = borrarlo."""
        self.P = self.P[:15, :15].copy()
        self.n_clone = 0
        self.p_c = self.R_c = None

    # ---------------- residuo y jacobiano --------------------------------
    def residuo(self, dp_meas, dR_meas):
        dp_pred = self.R_c.T @ (self.p - self.p_c)
        dR_pred = self.R_c.T @ self.R
        r = np.concatenate([dp_meas - dp_pred, Log(dR_pred.T @ dR_meas)])
        return r, dp_pred, dR_pred

    def jacobiano(self, dp_pred, dR_pred, r_theta, usar_jl=True):
        H = np.zeros((6, 21))
        H[0:3, IP:IP + 3] = self.R_c.T
        H[0:3, ICP:ICP + 3] = -self.R_c.T
        H[0:3, ICT:ICT + 3] = skew(dp_pred)
        Jli = jac_izq_inv(r_theta) if usar_jl else np.eye(3)
        H[3:6, IT:IT + 3] = Jli
        H[3:6, ICT:ICT + 3] = -Jli @ dR_pred.T
        return H

    def residuo_direccion(self, u_meas, dR_meas):
        """La medida es el vector unitario, sin escala. Bloque J de rango 2:
        no hay información radial — la escala queda observable solo vía IMU."""
        d = self.R_c.T @ (self.p - self.p_c)
        n = max(np.linalg.norm(d), 1e-9)
        u = d / n
        dR_pred = self.R_c.T @ self.R
        r = np.concatenate([u_meas - u, Log(dR_pred.T @ dR_meas)])
        J = (np.eye(3) - np.outer(u, u)) / n
        H = np.zeros((6, 21))
        H[0:3, IP:IP + 3] = J @ self.R_c.T
        H[0:3, ICP:ICP + 3] = -J @ self.R_c.T
        H[0:3, ICT:ICT + 3] = J @ skew(d)
        Jli = jac_izq_inv(r[3:6])
        H[3:6, IT:IT + 3] = Jli
        H[3:6, ICT:ICT + 3] = -Jli @ dR_pred.T
        return r, H, n

    # ---------------- corrección -----------------------------------------
    def update(self, r, H, Rm, chi2_umbral=16.8):
        S = H @ self.P @ H.T + Rm
        d2 = float(r @ np.linalg.solve(S, r))
        if d2 > chi2_umbral:
            return False, d2
        Kg = self.P @ H.T @ np.linalg.inv(S)
        self._inyectar(Kg @ r)
        if self.joseph:
            I_KH = np.eye(self.P.shape[0]) - Kg @ H
            self.P = I_KH @ self.P @ I_KH.T + Kg @ Rm @ Kg.T
        else:
            self.P = (np.eye(self.P.shape[0]) - Kg @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return True, d2

    def update_pose_relativa(self, dp_meas, dR_meas, sigma_p=0.05, sigma_r=0.02,
                             chi2_umbral=16.8):
        assert self.n_clone == 1
        r, dp_pred, dR_pred = self.residuo(dp_meas, dR_meas)
        H = self.jacobiano(dp_pred, dR_pred, r[3:6])
        Rm = np.diag(np.concatenate([sigma_p ** 2 * np.ones(3),
                                     sigma_r ** 2 * np.ones(3)]))
        return self.update(r, H, Rm, chi2_umbral)

    def update_direccion(self, u_meas, dR_meas, sigma_u=0.03, sigma_r=0.02,
                         chi2_umbral=16.8):
        assert self.n_clone == 1
        r, H, _ = self.residuo_direccion(u_meas, dR_meas)
        Rm = np.diag(np.concatenate([sigma_u ** 2 * np.ones(3),
                                     sigma_r ** 2 * np.ones(3)]))
        return self.update(r, H, Rm, chi2_umbral)

    def _inyectar(self, dx, reset=None):
        self.p = self.p + dx[IP:IP + 3]
        self.v = self.v + dx[IV:IV + 3]
        self.R = self.R @ Exp(dx[IT:IT + 3])
        self.bg = self.bg + dx[IBG:IBG + 3]
        self.ba = self.ba + dx[IBA:IBA + 3]
        if self.n_clone:
            self.p_c = self.p_c + dx[ICP:ICP + 3]
            self.R_c = self.R_c @ Exp(dx[ICT:ICT + 3])

        if reset if reset is not None else self.reset_error:
            n = self.P.shape[0]
            Gr = np.eye(n)
            Gr[IT:IT + 3, IT:IT + 3] = np.eye(3) - 0.5 * skew(dx[IT:IT + 3])
            if self.n_clone:
                Gr[ICT:ICT + 3, ICT:ICT + 3] = np.eye(3) - 0.5 * skew(dx[ICT:ICT + 3])
            self.P = 0.5 * (Gr @ self.P @ Gr.T + (Gr @ self.P @ Gr.T).T)

    # ---------------- utilidades -----------------------------------------
    @property
    def T_ws(self):
        return make_T(self.R, self.p)

    def traza(self):
        d = np.diag(self.P)
        return (d[IP:IP + 3].sum(), d[IV:IV + 3].sum(), d[IT:IT + 3].sum(),
                d[IBG:IBG + 3].sum(), d[IBA:IBA + 3].sum())

    def estado(self):
        return (self.p.copy(), self.v.copy(), self.R.copy(), self.bg.copy(),
                self.ba.copy(),
                None if self.p_c is None else self.p_c.copy(),
                None if self.R_c is None else self.R_c.copy())

    def set_estado(self, e):
        self.p, self.v, self.R, self.bg, self.ba, self.p_c, self.R_c = \
            [None if x is None else np.array(x, dtype=float) for x in e]


# ---------------------------------------------------------------------------
# Cámara -> body (corrección (3): la escala NO toca el brazo de palanca)
# ---------------------------------------------------------------------------
def cam_a_body(T_c0c1, R_bc, p_bc, escala=None):
    """T_c0c1 (frame c1 al c0) -> (Δp_body, ΔR_body) del clon al actual.
    Si `escala` no es None, la traslación se toma unitaria y se escala SOLO
    la parte de cámara; el brazo de palanca (I - ΔR)·p_bc queda intacto."""
    dR_b = R_bc @ T_c0c1[:3, :3] @ R_bc.T
    palanca = (np.eye(3) - dR_b) @ p_bc
    if escala is None:
        return R_bc @ T_c0c1[:3, 3] + palanca, dR_b
    u = T_c0c1[:3, 3] / max(np.linalg.norm(T_c0c1[:3, 3]), 1e-12)
    return R_bc @ (escala * u) + palanca, dR_b


def escala_desde_prediccion(dp_pred_b, dR_b, u_cam, R_bc, p_bc):
    """s = d_cam · û proyectando la predicción de la IMU sobre la dirección
    visual. AVISO §5.12: este mecanismo absorbe outliers y desactiva el χ²."""
    d_cam = R_bc.T @ (dp_pred_b - (np.eye(3) - dR_b) @ p_bc)
    return float(d_cam @ u_cam), d_cam


# ---------------------------------------------------------------------------
# Bucle principal (Fase 5 §5.8, parametrizado por Secuencia)
# ---------------------------------------------------------------------------
@dataclass
class ConfigEskf:
    modo: str = "direccion"        # "direccion" | "escala"
    sigma_p: float = 0.05
    sigma_r: float = 0.02
    sigma_u: float = 0.03
    factor_q: float = 4.0
    chi2_umbral: float = 16.8
    ventana_escala: int = 25
    # P0
    sigma_p0: float = 0.02
    sigma_v0: float = 0.05
    sigma_th0_deg: float = 0.5
    sigma_bg0: float = 2e-3
    sigma_ba0: float = 0.15


def hacer_P0(c: ConfigEskf):
    return np.diag(np.concatenate([
        c.sigma_p0 ** 2 * np.ones(3),
        c.sigma_v0 ** 2 * np.ones(3),
        np.deg2rad(c.sigma_th0_deg) ** 2 * np.ones(3),
        c.sigma_bg0 ** 2 * np.ones(3),
        c.sigma_ba0 ** 2 * np.ones(3)]))


def ejecutar_eskf(seq, medidas, k0, k1, estado0, cfg: ConfigEskf = None,
                  solo_imu=False):
    """Corre el filtro sobre [k0, k1) de la secuencia.

    estado0 = (p0, v0, R0, bg0, ba0). El orden del bucle importa: corregir y
    LUEGO clonar, para que el clon herede el estado corregido y su covarianza
    cruzada. Con solo_imu=True se ignoran las medidas (dead reckoning)."""
    c = cfg or ConfigEskf()
    T_bc = seq.camara.T_body_cam
    R_bc, p_bc = T_bc[:3, :3], T_bc[:3, 3]

    f = ESKF(*estado0, sigmas=seq.imu_params.como_tupla(),
             P0=hacer_P0(c), factor_q=c.factor_q)
    f.clone()

    por_frame = defaultdict(list)
    if not solo_imu:
        for m in medidas:
            por_frame[m["k"]].append(m)

    hist_s = []
    reg = defaultdict(list)
    n_med = n_ok = 0

    reg["t"].append(seq.t_cam[k0]); reg["pose"].append(f.T_ws)
    reg["bg"].append(f.bg.copy()); reg["ba"].append(f.ba.copy())
    reg["trP"].append(f.traza())

    for k in range(k0 + 1, k1):
        f.predict(seq.imu_entre(seq.t_cam[k - 1], seq.t_cam[k]))

        for m in por_frame.get(k, []):
            n_med += 1
            dp_pred_b = f.R_c.T @ (f.p - f.p_c)
            T_c0c1 = m["T_c0c1"]
            u_cam = T_c0c1[:3, 3]                     # ya unitario

            if c.modo == "direccion":
                dp_b_unit, dR_b = cam_a_body(T_c0c1, R_bc, p_bc)
                u_meas = dp_b_unit / np.linalg.norm(dp_b_unit)
                ok, d2 = f.update_direccion(u_meas, dR_b, c.sigma_u, c.sigma_r,
                                            c.chi2_umbral)
                s_usada = np.nan
            else:
                _, dR_b = cam_a_body(T_c0c1, R_bc, p_bc)
                s_k, _ = escala_desde_prediccion(dp_pred_b, dR_b, u_cam, R_bc, p_bc)
                hist_s.append(s_k)
                if len(hist_s) > c.ventana_escala:
                    hist_s.pop(0)
                escala = float(np.median(hist_s)) if len(hist_s) >= 5 else max(s_k, 1e-4)
                if escala <= 0:
                    ok, d2, s_usada = False, np.nan, escala
                else:
                    dp_meas = cam_a_body(T_c0c1, R_bc, p_bc, escala=escala)[0]
                    ok, d2 = f.update_pose_relativa(dp_meas, dR_b, c.sigma_p,
                                                    c.sigma_r, c.chi2_umbral)
                    s_usada = escala

            n_ok += ok
            reg["m_t"].append(seq.t_cam[k]); reg["m_d2"].append(d2)
            reg["m_ok"].append(bool(ok)); reg["m_s"].append(s_usada)
            reg["m_par"].append(m["paralaje"])
            reg["m_s_gt"].append(m.get("s_gt", np.nan))

            f.drop_clone(); f.clone()

        reg["t"].append(seq.t_cam[k]); reg["pose"].append(f.T_ws)
        reg["bg"].append(f.bg.copy()); reg["ba"].append(f.ba.copy())
        reg["trP"].append(f.traza())

    out = {kk: np.array(vv) for kk, vv in reg.items()}
    out["filtro"] = f
    out["n_medidas"], out["n_aceptadas"] = n_med, n_ok
    out["modo"] = "imu" if solo_imu else c.modo
    return out
