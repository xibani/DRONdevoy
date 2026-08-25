# Fase 5 — Escala métrica y fusión loosely-coupled (ESKF)

**Objetivo:** tu primer estimador real. IMU predice, visión corrige.
**Criterio de éxito:** ATE-RMSE **SE(3)** (sin ajustar escala) < 1.5 m sobre 60–120 s
de MH_01. Y, más importante, que el filtro **no diverja** y que los bias converjan
hacia los valores del GT.

Aquí es donde el curso se pone serio. Tómate el tiempo. La mecánica que montes aquí
es literalmente la misma que hay dentro del MSCKF de la Fase 6A.

---

## 5.1 Por qué "loosely-coupled" primero

| | Loosely-coupled | Tightly-coupled |
|---|---|---|
| Entrada visual | Una **pose relativa** ya estimada | Las **observaciones de features** en bruto |
| Estado | IMU (+ clones de pose) | IMU + clones + estructura implícita |
| Ventaja | Modular, depurable, front-end reemplazable | Más preciso, robusto con pocas features |
| Desventaja | Descarta información; correlaciones mal modeladas | Complejo, difícil de depurar |
| Ejemplos | SVO+MSF, ROVIO (parcial) | MSCKF, VINS-Mono, ORB-SLAM3-VI |

Loosely-coupled es peor. Se hace primero porque **puedes localizar los bugs**: si el
filtro diverge, o es la predicción (Fase 3, ya validada) o la corrección. En un
MSCKF, con el estado, los clones, la triangulación y la proyección al espacio nulo
mezclados, un bug es una semana.

---

## 5.2 El problema de la escala

Tu VO de la Fase 4 devuelve `t` con `‖t‖ = 1` en cada par. Tres estrategias:

### (a) Escala por ventana contra la IMU — *la que implementamos*

Sobre una ventana de N frames, la IMU (propagada desde un estado ya corregido) da
incrementos métricos `Δp_imu` y la VO da incrementos `Δp_vo` sin escala. Mínimos
cuadrados con solución cerrada:

```
s* = Σ_k (Δp_imu,k · Δp_vo,k) / Σ_k ‖Δp_vo,k‖²
```

```python
def estimate_scale(dp_imu, dp_vo):
    """dp_imu, dp_vo: (N,3) incrementos en el MISMO frame."""
    num = np.sum(dp_imu * dp_vo)
    den = np.sum(dp_vo * dp_vo)
    return num / den if den > 1e-9 else 1.0
```

Funciona porque, en ventanas de 0.5–2 s, la IMU **sí** es métricamente fiable
(Fase 3: 0.05 m de error a 2 s). Es una escala local, no global — y eso está bien:
la reestimas continuamente.

### (b) Solo la dirección — *más riguroso*

Si no quieres depender de la escala, usa como medida el **vector unitario** de
traslación (2 DoF de información) + la rotación (3 DoF). El residuo es
`u_meas − u_pred` con `u = Δp/‖Δp‖`. Con esto la escala del sistema queda
observable solo a través de la IMU, que es lo correcto teóricamente. Lo describo en
§5.7.

### (c) Escala en el estado

Añade `log s` como estado escalar y estímalo. Elegante, pero **solo válido si tu VO
tiene escala interna coherente** (VO con mapa local + PnP, §4.7), no con
`recoverPose` frame a frame donde cada `t` es unitaria por separado.

---

## 5.3 El ESKF: estado, predicción, corrección

### Estado

- **Nominal** (lo que reportas): `p_w ∈ R³`, `v_w ∈ R³`, `R_ws ∈ SO(3)`, `b_g`, `b_a`.
- **De error** (lo que filtras): `δx = [δp, δv, δθ, δb_g, δb_a] ∈ R¹⁵`,
  con `R = R̄ · Exp(δθ)` (perturbación local/derecha).

Que el filtro viva en `R¹⁵` y la orientación en `SO(3)` es *todo* el punto del ESKF.
La covarianza es 15×15 y no singular.

### Dinámica del error (continuo)

Con `a = a_m − b_a`, `w = w_m − b_g`, `R = R_ws`:

```
δṗ  = δv
δv̇  = −R [a]× δθ − R δb_a − R n_a
δθ̇  = −[w]× δθ − δb_g − n_g
δḃ_g = n_bg
δḃ_a = n_ba
```

Deriva estas cinco líneas a mano una vez. Está en Solà §5–6 con todo el detalle.

### Código

```python
import numpy as np
from anexo_utils import Exp, Log, skew, G_W, inv_T, make_T

IP, IV, IT, IBG, IBA = 0, 3, 6, 9, 12       # índices en el estado de error

class ESKF:
    def __init__(self, imu_params, p0, v0, R0, bg0, ba0, P0=None):
        self.p, self.v, self.R = p0.copy(), v0.copy(), R0.copy()
        self.bg, self.ba = bg0.copy(), ba0.copy()
        self.P = P0 if P0 is not None else np.diag(np.concatenate([
            1e-4*np.ones(3),      # p
            1e-2*np.ones(3),      # v
            (np.deg2rad(1.0)**2)*np.ones(3),   # theta
            1e-6*np.ones(3),      # bg
            1e-4*np.ones(3),      # ba
        ]))
        ip = imu_params
        self.Qc = np.diag(np.concatenate([
            ip.sigma_g**2 * np.ones(3),
            ip.sigma_a**2 * np.ones(3),
            ip.sigma_bg**2 * np.ones(3),
            ip.sigma_ba**2 * np.ones(3),
        ]))
        self.n_clone = 0            # 0 o 1 clon activo

    # ---------------- predicción -----------------------------------------
    def predict(self, imu_chunk):
        """imu_chunk: (M,7) [t, w(3), a(3)] cubriendo el intervalo completo."""
        for k in range(len(imu_chunk) - 1):
            dt = imu_chunk[k+1, 0] - imu_chunk[k, 0]
            if dt <= 0:
                continue
            w0 = imu_chunk[k,   1:4] - self.bg
            w1 = imu_chunk[k+1, 1:4] - self.bg
            a0 = imu_chunk[k,   4:7] - self.ba
            a1 = imu_chunk[k+1, 4:7] - self.ba
            self._step(0.5*(w0+w1), a0, a1, dt)

    def _step(self, w, a0, a1, dt):
        R_old = self.R
        R_new = R_old @ Exp(w * dt)
        a_w = 0.5*(R_old @ a0 + R_new @ a1) + G_W

        # --- nominal
        self.p = self.p + self.v*dt + 0.5*a_w*dt**2
        self.v = self.v + a_w*dt
        self.R = R_new

        # --- covarianza (solo el bloque IMU 15x15; los clones son constantes)
        n = 15
        F = np.zeros((n, n))
        F[IP:IP+3, IV:IV+3]   = np.eye(3)
        F[IV:IV+3, IT:IT+3]   = -R_old @ skew(a0)
        F[IV:IV+3, IBA:IBA+3] = -R_old
        F[IT:IT+3, IT:IT+3]   = -skew(w)
        F[IT:IT+3, IBG:IBG+3] = -np.eye(3)

        G = np.zeros((n, 12))
        G[IV:IV+3, 3:6]   = -R_old
        G[IT:IT+3, 0:3]   = -np.eye(3)
        G[IBG:IBG+3, 6:9] = np.eye(3)
        G[IBA:IBA+3, 9:12]= np.eye(3)

        Phi = np.eye(n) + F*dt + 0.5*(F @ F)*dt**2      # 2º orden
        Qd  = Phi @ (G @ self.Qc @ G.T) @ Phi.T * dt

        P = self.P
        P[:n, :n] = Phi @ P[:n, :n] @ Phi.T + Qd
        if self.n_clone:                                 # correlación IMU-clon
            P[:n, n:] = Phi @ P[:n, n:]
            P[n:, :n] = P[:n, n:].T
        self.P = 0.5*(P + P.T)                           # simetrizar SIEMPRE

    # ---------------- stochastic cloning ---------------------------------
    def clone(self):
        """Guarda una copia (p, R) del estado actual como referencia para la
        siguiente medida de pose relativa. Aumenta el estado de error a 21."""
        self.p_c, self.R_c = self.p.copy(), self.R.copy()
        A = np.zeros((21, 15))
        A[:15, :15] = np.eye(15)
        A[15:18, IP:IP+3] = np.eye(3)      # el clon de p es exactamente p
        A[18:21, IT:IT+3] = np.eye(3)      # el clon de theta es exactamente theta
        P15 = self.P[:15, :15]
        self.P = A @ P15 @ A.T
        self.P = 0.5*(self.P + self.P.T)
        self.n_clone = 1

    def drop_clone(self):
        self.P = self.P[:15, :15].copy()
        self.n_clone = 0

    # ---------------- corrección: pose relativa ---------------------------
    def update_relative_pose(self, dp_meas, dR_meas, sigma_p=0.05, sigma_r=0.02,
                             chi2_thresh=16.8):
        """dp_meas: traslación del clon al estado actual, EN FRAME DEL CLON,
        ya escalada a metros.  dR_meas: rotación relativa R_clone_current."""
        assert self.n_clone == 1
        R1, p1 = self.R_c, self.p_c
        dp_pred = R1.T @ (self.p - p1)
        dR_pred = R1.T @ self.R

        r = np.concatenate([dp_meas - dp_pred,
                            Log(dR_meas.T @ dR_pred)])

        H = np.zeros((6, 21))
        # residuo de traslación
        H[0:3, IP:IP+3]  = -R1.T                    # d/d δp2   (signo: r = meas - pred)
        H[0:3, 15:18]    =  R1.T
        H[0:3, 18:21]    = -skew(dp_pred)
        # residuo de rotación   (aprox. de 1er orden, válida con residuos pequeños)
        H[3:6, IT:IT+3]  = -np.eye(3)
        H[3:6, 18:21]    =  dR_pred.T

        Rm = np.diag(np.concatenate([sigma_p**2*np.ones(3), sigma_r**2*np.ones(3)]))
        S = H @ self.P @ H.T + Rm
        # --- gating chi2 (6 DoF, 99%): descarta medidas visuales corruptas
        d2 = float(r @ np.linalg.solve(S, r))
        if d2 > chi2_thresh:
            return False, d2

        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ r
        self._inject(dx)
        I_KH = np.eye(self.P.shape[0]) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T      # forma de Joseph
        self.P = 0.5*(self.P + self.P.T)
        return True, d2

    def _inject(self, dx):
        self.p  += dx[IP:IP+3]
        self.v  += dx[IV:IV+3]
        self.R   = self.R @ Exp(dx[IT:IT+3])
        self.bg += dx[IBG:IBG+3]
        self.ba += dx[IBA:IBA+3]
        if self.n_clone:
            self.p_c += dx[15:18]
            self.R_c  = self.R_c @ Exp(dx[18:21])

    @property
    def T_ws(self):
        return make_T(self.R, self.p)
```

**Detalles no negociables:**

- **Forma de Joseph** para la actualización de covarianza. La forma corta
  `P = (I−KH)P` pierde simetría y definición positiva tras unos cientos de pasos y
  el filtro explota. Con Joseph, no.
- **Simetrizar `P` en cada operación.** Cuesta nada y salva el filtro.
- **Gating χ²**: sin él, una sola pose visual mala (rotación pura, outliers) te
  arruina el estado. Umbral 16.8 para 6 DoF al 99 %.
- **Reset del error state**: al inyectar `δθ`, en rigor hay que corregir `P` con
  `G = I − ½[δθ]×` en el bloque de rotación. Con residuos pequeños el efecto es
  despreciable; menciono que existe porque en el MSCKF sí importa.

---

## 5.4 Bucle principal

```python
from anexo_utils import EurocSequence

def run_eskf(seq, k0=20, k1=1200, kf_stride=2):
    cam = seq.cam0
    T_i_c = cam.T_body_cam                 # T_body_from_cam
    T_c_i = inv_T(T_i_c)

    t0 = seq.t_cam[k0]
    p0, R0, v0, bg0, ba0 = seq.gt_at(t0)   # inicialización desde GT (ver Fase 8)
    f = ESKF(seq.imu_params, p0, v0, R0, bg0, ba0)

    tracker = KLTTracker()
    _ = tracker.track(seq.image(k0, undistort=True), k0)
    f.clone()
    p_ref_imu = f.p.copy()                 # para la escala
    hist_dp_imu, hist_dp_vo = [], []
    scale = 1.0

    times, poses, accepted = [t0], [f.T_ws], 0

    for k in range(k0+1, k1):
        f.predict(seq.imu_between(seq.t_cam[k-1], seq.t_cam[k]))
        img = seq.image(k, undistort=True)
        ids, p_prev, p_cur = tracker.track(img, k)

        if (k - k0) % kf_stride == 0 and p_prev is not None and len(p_prev) > 30:
            T_c1c0, inl = relative_pose(cam, p_prev, p_cur)
            if T_c1c0 is not None and not is_degenerate(p_prev, p_cur, 2.0):
                # cámara -> body:   T_b0_b1 = T_b_c @ T_c0_c1 @ T_c_b
                T_b0b1 = T_i_c @ inv_T(T_c1c0) @ T_c_i
                dp_vo, dR_vo = T_b0b1[:3, 3], T_b0b1[:3, :3]

                # --- escala por ventana contra la predicción de la IMU
                dp_imu_pred = f.R_c.T @ (f.p - f.p_c)
                hist_dp_imu.append(dp_imu_pred); hist_dp_vo.append(dp_vo)
                if len(hist_dp_imu) > 40:
                    hist_dp_imu.pop(0); hist_dp_vo.pop(0)
                if len(hist_dp_imu) >= 5:
                    scale = estimate_scale(np.array(hist_dp_imu), np.array(hist_dp_vo))

                ok, d2 = f.update_relative_pose(scale*dp_vo, dR_vo,
                                                sigma_p=0.05, sigma_r=0.02)
                accepted += ok
            f.drop_clone(); f.clone()

        times.append(seq.t_cam[k]); poses.append(f.T_ws)

    print(f"medidas aceptadas: {accepted}")
    return np.array(times), np.array(poses), f
```

Evaluación:

```python
times, poses, f = run_eskf(seq, 20, 1400)
p_est = poses[:, :3, 3]
p_gt  = np.array([seq.gt_at(t)[0] for t in times])
from anexo_utils import ate_rmse
print("ATE-RMSE SE(3):", ate_rmse(p_gt, p_est, with_scale=False), "m")
print("bias gyro final :", f.bg, " GT:", seq.gt_at(times[-1])[3])
print("bias accel final:", f.ba, " GT:", seq.gt_at(times[-1])[4])
```

---

## 5.5 Diagnóstico: qué mirar cuando no funciona

Un filtro que "va mal" es inútil sin diagnóstico. Registra siempre estas cuatro
señales:

1. **NEES / distancia de Mahalanobis** de cada medida. Si `d²` es sistemáticamente
   mucho mayor que 6, tu `R` de medida es demasiado optimista o hay un bug de
   jacobiano. Si es sistemáticamente ~0, es demasiado pesimista y la visión no está
   corrigiendo nada.
2. **Traza de `P`** por bloque (posición, velocidad, actitud, bias). Debe crecer
   entre medidas y caer en cada corrección. Si solo crece: las medidas se están
   rechazando. Si colapsa a cero: sobreconfianza, el filtro dejará de escuchar a la
   visión y derivará como en la Fase 3.
3. **Convergencia de bias** contra GT. El del giro debe converger en pocos segundos.
   El del acelerómetro es más lento y depende de la excitación.
4. **Tasa de aceptación** del gating. Objetivo: > 80 %.

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(13, 7))
ax[0,0].plot(log_t, log_d2); ax[0,0].axhline(16.8, c='r', ls='--')
ax[0,0].set_title("Mahalanobis d² de la medida visual")
ax[0,1].semilogy(log_t, log_trP_pos, label="pos")
ax[0,1].semilogy(log_t, log_trP_att, label="att"); ax[0,1].legend()
ax[0,1].set_title("traza de P")
ax[1,0].plot(log_t, log_bg); ax[1,0].set_title("bias gyro estimado")
ax[1,1].plot(log_t, log_scale); ax[1,1].set_title("escala estimada")
plt.tight_layout(); plt.show()
```

---

## 5.6 Sintonización

| Parámetro | Rango sensato | Efecto |
|---|---|---|
| `sigma_p` (medida traslación) | 0.02–0.15 m | Bajo → confía en la VO, deriva menos pero peta con outliers |
| `sigma_r` (medida rotación) | 0.005–0.05 rad | La rotación visual es buena; puedes ser agresivo |
| `Qc` del IMU | ×1 a ×10 del YAML | Subirlo hace el filtro más "elástico" y robusto |
| `kf_stride` | 2–5 frames | Más separación → más paralaje → mejor esencial |
| Ventana de escala | 20–50 medidas | Corta → ruidosa; larga → lenta ante cambios |

Regla general: si diverge, **sube `Qc`** antes de bajar `sigma_p`. Un filtro
demasiado confiado en su propia predicción es el modo de fallo más común.

---

## 5.7 Variante rigurosa: medida solo de dirección

Si no quieres la escala explícita, sustituye el bloque de traslación por:

```python
u_meas = dp_vo / np.linalg.norm(dp_vo)
d_pred = R1.T @ (self.p - p1)
n_pred = np.linalg.norm(d_pred)
u_pred = d_pred / max(n_pred, 1e-6)
r_u = u_meas - u_pred

# Jacobiano: d(u)/d(d) = (I - u u^T)/||d||
J = (np.eye(3) - np.outer(u_pred, u_pred)) / max(n_pred, 1e-6)
H[0:3, IP:IP+3] = -J
H[0:3, 15:18]   =  J
H[0:3, 18:21]   = -J @ skew(d_pred)
```

La covarianza `Rm` del bloque de dirección es de rango 2 (no hay información radial),
así que inflá­la un poco en la diagonal (`sigma_u² I` con `sigma_u ≈ 0.02`) para que
`S` sea invertible. Esta formulación es la correcta desde el punto de vista de
observabilidad y es la que verás en los papers de MSF.

---

## 5.8 Lo que has aprendido, y el puente a la Fase 6

Tres piezas que reutilizas tal cual en el MSCKF:

1. **Propagación del error state** con `Φ`, `G`, `Qd`. En el MSCKF es idéntica.
2. **Stochastic cloning.** Aquí clonas *una* pose. El MSCKF clona *N* poses
   (típicamente 10–20) y esa es literalmente la única diferencia estructural.
3. **Gating χ² + actualización de Joseph.**

Lo que cambia en la Fase 6: en vez de una pose relativa precocinada, el residuo se
construye a partir de las **observaciones de la feature en todos los clones**, y
la posición 3D de la feature se elimina proyectando al espacio nulo de su jacobiano.
Eso es el MSCKF.

---

## 5.9 Entregable

Notebook `05_eskf.ipynb` con:

1. Clase `ESKF` completa con Joseph, gating y cloning.
2. Bucle sobre ≥ 60 s de MH_01.
3. Los cuatro plots de diagnóstico (§5.5).
4. ATE-RMSE **SE(3)** reportado, comparado con la VO pura (Sim(3)) de la Fase 4.
5. Un barrido de al menos un parámetro (`sigma_p` o el factor de `Qc`) mostrando
   cómo cambia el ATE.
