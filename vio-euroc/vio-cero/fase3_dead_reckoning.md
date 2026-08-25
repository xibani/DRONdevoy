# Fase 3 — Dead reckoning de IMU (ver la deriva)

**Objetivo:** integrar solo la IMU, medir la deriva, y con ello fijar de una vez
por todas los convenios de ejes, gravedad y bias.
**Criterio de éxito:** con estado inicial y bias tomados del GT, el error de
posición a los **5 s** debe ser **< 0.5 m**. Si es de metros, tienes un bug, no
"deriva normal".

Este es el filtro de calidad más eficaz del curso: casi todos los bugs de convenio
se manifiestan aquí y son *imposibles* de localizar más tarde.

---

## 3.1 Las ecuaciones

Estado nominal: `x = (p_w, v_w, R_ws, b_g, b_a)`.

Medidas: `w_m` (rad/s, frame body), `a_m` (m/s², frame body, **fuerza específica**).

Modelo:

```
w = w_m - b_g - n_g
a = a_m - b_a - n_a

Ṙ_ws = R_ws · [w]×
v̇_w  = R_ws · a + g_w                 con g_w = (0, 0, -9.81)
ṗ_w  = v_w
ḃ_g  = n_bg          (random walk)
ḃ_a  = n_ba
```

**Por qué el acelerómetro no mide aceleración:** en caída libre marca cero; en
reposo marca `-g` rotado al body. Lo que mide es fuerza específica
`f = R_ws^T (a_w - g_w)`. Despejar `a_w = R_ws f + g_w` es *la* ecuación de la
Fase 3, y el signo de `g_w` es donde se equivoca todo el mundo.

### Discretización

**Euler hacia adelante** (sencillo, error O(dt)):

```
R_{k+1} = R_k · Exp(w_k · dt)
a_w     = R_k · a_k + g_w
p_{k+1} = p_k + v_k·dt + 0.5·a_w·dt²
v_{k+1} = v_k + a_w·dt
```

**Punto medio / midpoint** (error O(dt²), lo que debes usar):

```
w̄       = 0.5 (w_k + w_{k+1})
R_{k+1} = R_k · Exp(w̄ · dt)
ā_w     = 0.5 (R_k·a_k + R_{k+1}·a_{k+1}) + g_w
p_{k+1} = p_k + v_k·dt + 0.5·ā_w·dt²
v_{k+1} = v_k + ā_w·dt
```

A 200 Hz la diferencia entre Euler y midpoint en 5 s es de ~centímetros, pero se
acumula. Y midpoint es lo que usan VINS-Mono y la preintegración de GTSAM, así que
es lo que quieres tener interiorizado.

---

## 3.2 Implementación

```python
import numpy as np
from anexo_utils import EurocSequence, Exp, G_W

seq = EurocSequence("~/datasets/euroc/MH_01_easy")

def integrate_imu(imu, p0, v0, R0, bg, ba, method="midpoint"):
    """imu: (M,7) [t, w(3), a(3)]. Devuelve (t (M,), p (M,3), v (M,3), R (M,3,3))."""
    M = len(imu)
    ts = imu[:, 0]
    w_all = imu[:, 1:4] - bg
    a_all = imu[:, 4:7] - ba

    p = np.zeros((M, 3)); v = np.zeros((M, 3)); R = np.zeros((M, 3, 3))
    p[0], v[0], R[0] = p0, v0, R0

    for k in range(M - 1):
        dt = ts[k+1] - ts[k]
        if dt <= 0:
            p[k+1], v[k+1], R[k+1] = p[k], v[k], R[k]
            continue
        if method == "euler":
            R[k+1] = R[k] @ Exp(w_all[k] * dt)
            a_w = R[k] @ a_all[k] + G_W
        else:                                   # midpoint
            w_bar = 0.5 * (w_all[k] + w_all[k+1])
            R[k+1] = R[k] @ Exp(w_bar * dt)
            a_w = 0.5 * (R[k] @ a_all[k] + R[k+1] @ a_all[k+1]) + G_W
        p[k+1] = p[k] + v[k] * dt + 0.5 * a_w * dt**2
        v[k+1] = v[k] + a_w * dt
    return ts, p, v, R
```

### Ejecución con inicialización desde GT

```python
t_start = seq.t_gt[0] + 0.5              # deja margen
T_run   = 30.0
mask = (seq.t_imu >= t_start) & (seq.t_imu <= t_start + T_run)
imu = seq.imu[["t","wx","wy","wz","ax","ay","az"]].values[mask]

p0, R0, v0, bg0, ba0 = seq.gt_at(t_start)
ts, p, v, R = integrate_imu(imu, p0, v0, R0, bg0, ba0)

# Error contra GT
p_gt = np.array([seq.gt_at(t)[0] for t in ts])
err  = np.linalg.norm(p - p_gt, axis=1)
for T in (1, 2, 5, 10, 20, 30):
    i = np.searchsorted(ts - ts[0], T)
    if i < len(ts):
        print(f"t = {T:4.0f} s -> error de posicion = {err[i]:8.3f} m")
```

**Resultado esperado (orden de magnitud):**

```
t =    1 s ->  ~0.01 m
t =    2 s ->  ~0.05 m
t =    5 s ->  ~0.3  m
t =   10 s ->  ~1.5  m
t =   20 s ->  ~8    m
t =   30 s ->  ~25   m
```

El crecimiento es aproximadamente **cuadrático en el error de acelerómetro** y
**cúbico en el error de giróscopo** (un error de actitud rota la gravedad, que se
integra dos veces). Esta es la razón física de que la IMU sola sea inútil y de que
la cámara sea necesaria.

---

## 3.3 Los tres experimentos que debes hacer

### Experimento A — Euler vs midpoint

```python
for method in ("euler", "midpoint"):
    ts, p, _, _ = integrate_imu(imu, p0, v0, R0, bg0, ba0, method)
    p_gt = np.array([seq.gt_at(t)[0] for t in ts])
    print(method, "error a 10 s:",
          np.linalg.norm(p - p_gt, axis=1)[np.searchsorted(ts-ts[0], 10)])
```

### Experimento B — el peso del bias

```python
import matplotlib.pyplot as plt
configs = {
    "bias GT (gyro+accel)": (bg0, ba0),
    "sin bias":             (np.zeros(3), np.zeros(3)),
    "solo bias gyro":       (bg0, np.zeros(3)),
    "solo bias accel":      (np.zeros(3), ba0),
}
plt.figure(figsize=(9,5))
for name, (bg, ba) in configs.items():
    ts, p, _, _ = integrate_imu(imu, p0, v0, R0, bg, ba)
    p_gt = np.array([seq.gt_at(t)[0] for t in ts])
    plt.semilogy(ts-ts[0], np.linalg.norm(p-p_gt, axis=1), label=name)
plt.xlabel("t [s]"); plt.ylabel("error de posición [m]")
plt.legend(); plt.grid(alpha=.3, which="both"); plt.show()
```

Vas a ver que **quitar el bias del giróscopo es catastrófico** (el error de yaw
rota la gravedad y ese error entra en la doble integración) mientras que el del
acelerómetro es más benigno. Esta asimetría explica por qué todos los VIO estiman
el bias del giro con prioridad, y por qué la inicialización de VINS-Mono empieza
precisamente por él.

### Experimento C — el peso del estado inicial

```python
for sigma_v in (0.0, 0.05, 0.2):
    v_pert = v0 + np.random.randn(3)*sigma_v
    ts, p, _, _ = integrate_imu(imu, p0, v_pert, R0, bg0, ba0)
    p_gt = np.array([seq.gt_at(t)[0] for t in ts])
    e = np.linalg.norm(p-p_gt, axis=1)[np.searchsorted(ts-ts[0], 10)]
    print(f"sigma_v0={sigma_v}: error a 10 s = {e:.2f} m")
```

Un error de 0.2 m/s en la velocidad inicial produce ~2 m de error a los 10 s
(crece linealmente: `Δp = Δv·t`). Por eso la **inicialización visual-inercial**
(Fase 8) es un problema en sí mismo: si no arrancas con la velocidad y la gravedad
bien estimadas, el filtro nunca converge del todo.

---

## 3.4 Comparación de actitud

El error de posición mezcla todo. Separa la actitud:

```python
from anexo_utils import Log
R_gt = np.array([seq.gt_at(t)[1] for t in ts])
ang_err = np.array([np.degrees(np.linalg.norm(Log(Rg.T @ Re)))
                    for Rg, Re in zip(R_gt, R)])
plt.plot(ts-ts[0], ang_err); plt.xlabel("t [s]"); plt.ylabel("error de actitud [°]")
plt.grid(alpha=.3); plt.show()
print("error de actitud a 30 s:", ang_err[-1], "°")
```

Con el bias del GT deberías estar por debajo de **1–2°** a los 30 s. La actitud
deriva mucho más despacio que la posición: eso es lo que hace viable la fusión.

---

## 3.5 Visualización 3D

```python
fig = plt.figure(figsize=(7,6)); ax = fig.add_subplot(projection="3d")
ax.plot(*p_gt.T, label="GT", lw=1.5)
ax.plot(*p.T,   label="IMU dead reckoning", lw=1.5)
ax.legend(); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.show()
```

La forma será reconocible durante los primeros segundos y luego se irá "escapando",
típicamente hacia abajo o hacia un lado: es la gravedad mal cancelada por el error
de actitud acumulado.

---

## 3.6 Inicialización sin GT (previa a la Fase 8)

Para que esto sea realista, prueba a inicializar **sin** tocar el GT, usando el
tramo estático inicial:

```python
def static_init(seq, t_end=1.5, g=9.81):
    m = seq.t_imu < t_end
    w_mean = seq.imu.loc[m, ["wx","wy","wz"]].values.mean(0)
    a_mean = seq.imu.loc[m, ["ax","ay","az"]].values.mean(0)

    bg = w_mean                      # en reposo, todo lo que mide el giro es bias
    # La actitud: alinear -a_mean (dirección de la gravedad en body) con -z del mundo
    z_b = a_mean / np.linalg.norm(a_mean)     # "arriba" en frame body
    z_w = np.array([0., 0., 1.])
    v = np.cross(z_b, z_w); c = z_b @ z_w
    R_ws = np.eye(3) + skew(v) + skew(v) @ skew(v) * (1/(1+c))   # Rodrigues
    ba = a_mean - R_ws.T @ (-G_W)    # residuo tras quitar la gravedad
    return R_ws, bg, ba

from anexo_utils import skew
R_init, bg_init, ba_init = static_init(seq)
print("bias gyro:", bg_init, " vs GT:", seq.gt[["bgx","bgy","bgz"]].iloc[0].values)
print("actitud inicial estimada vs GT:",
      np.degrees(np.linalg.norm(Log(seq.gt_at(0.0)[1].T @ R_init))), "°")
```

Observaciones importantes:

- El **yaw no es observable**: `static_init` deja el yaw arbitrario (alineado con el
  eje x del body). El GT tiene un yaw definido por el mocap. Al comparar, el error
  angular incluirá ese yaw arbitrario → **para evaluar, alinea con Umeyama** (Fase 7).
- La separación entre `ba` y "gravedad mal alineada" es **ambigua en reposo**: un
  bias de acelerómetro y una inclinación producen la misma medida. Solo se separan
  con movimiento (excitación) — es la razón de que la inicialización VI necesite
  movimiento con aceleración no nula en varios ejes.

---

## 3.7 Trampas

| Trampa | Cómo se manifiesta |
|---|---|
| `g_w = (0,0,+9.81)` | El error se duplica y la trayectoria se va hacia arriba |
| Sumar `g` antes de rotar `a` | Deriva enorme en cuanto hay rotación |
| `R ← Exp(δ) · R` en vez de `R · Exp(δ)` | Deriva de actitud en cuanto no está horizontal |
| Usar `dt` fijo de 0.005 s | Sesgo pequeño pero sistemático; usa los `dt` reales |
| Reusar `R[k]` tras actualizarlo in-place | Bug clásico en midpoint: guarda `R_next` aparte |
| No restar el offset de timestamps | `dt` con ruido de cuantización; error un 10 % peor |

---

## 3.8 Entregable

Notebook `03_dead_reckoning.ipynb` con:

1. `integrate_imu` con las dos discretizaciones.
2. La tabla de error vs tiempo (1, 2, 5, 10, 20, 30 s) con bias del GT.
3. Los tres experimentos (A, B, C) con sus figuras.
4. Curva de error de actitud.
5. `static_init` y comparación con el GT.

**Antes de pasar a la Fase 4**, tu error a 5 s debe ser < 0.5 m. Si no, el bug está
aquí y arrastrarlo hará que el filtro de la Fase 5 no converja nunca y no sepas por qué.
