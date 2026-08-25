# Fase 1 — Carga y sincronización

**Objetivo:** un loader fiable que te da, para cada frame `k`, la imagen y **el
paquete de muestras de IMU que ocurren entre el frame `k-1` y el `k`**.
**Criterio de éxito:** `sum(dt) == t_cam[k] - t_cam[k-1]` con error < 1e-9 s para
todos los frames, y la trayectoria GT ploteada en 3D.

---

## 1.1 Por qué esta fase no es trivial

Parece "leer tres CSV". No lo es, por tres motivos:

1. **Resolución numérica.** Los timestamps son ns desde epoch (~1.4e18). En
   `float64` tienes ~15-16 dígitos significativos, así que `1.4e18 ns` en segundos
   deja unos 100 ns de resolución. Suficiente por poco, pero al restar dos tiempos
   grandes pierdes precisión relativa. **Resta el offset de la secuencia siempre.**

2. **El intervalo entre frames no contiene un número entero de muestras de IMU.**
   Cámara a 20 Hz (50 ms), IMU a 200 Hz (5 ms) → 10 muestras nominales, pero los
   relojes no caen alineados. Si simplemente coges las muestras que caen dentro,
   `sum(dt)` ≠ 50 ms y acumulas un sesgo sistemático en la integración. Solución:
   **interpolar muestras virtuales exactamente en `t_cam[k-1]` y `t_cam[k]`**.

3. **El GT tiene su propia rejilla temporal** (200 Hz en machine_hall, 100 Hz en
   Vicon rooms) y no coincide con la de la cámara. Para comparar necesitas
   interpolación: lineal para posición/velocidad, **SLERP para la orientación**.

---

## 1.2 Carga

Usa `anexo_utils.EurocSequence`, que ya hace todo lo anterior:

```python
from anexo_utils import EurocSequence
import numpy as np, matplotlib.pyplot as plt

seq = EurocSequence("~/datasets/euroc/MH_01_easy")
print(seq)
print(seq.imu.head())
print(seq.gt.head())
print("IMU dt mediano :", np.median(np.diff(seq.t_imu))*1000, "ms")
print("CAM dt mediano :", np.median(np.diff(seq.t_cam))*1000, "ms")
```

Si prefieres escribirlo tú (recomendable la primera vez), el núcleo es:

```python
imu = pd.read_csv(root/"imu0/data.csv")
imu.columns = ["t_ns","wx","wy","wz","ax","ay","az"]   # OJO: gyro ANTES que accel
t0 = int(min(imu.t_ns[0], cam.t_ns[0], gt.t_ns[0]))
imu["t"] = (imu.t_ns.astype(np.int64) - t0).astype(np.float64) * 1e-9
```

---

## 1.3 Asociación IMU ↔ cámara

La operación fundamental de todo el curso:

```python
def imu_between(seq, t0, t1):
    """(M,7) muestras [t, w(3), a(3)] con extremos interpolados en t0 y t1."""
    return seq.imu_between(t0, t1)   # ya implementado en anexo_utils
```

Test de que funciona (hazlo, no lo asumas):

```python
errs = []
for k in range(1, 200):
    chunk = seq.imu_between(seq.t_cam[k-1], seq.t_cam[k])
    dt = np.diff(chunk[:, 0])
    errs.append(abs(dt.sum() - (seq.t_cam[k] - seq.t_cam[k-1])))
    assert (dt > 0).all(), "dt no positivo -> muestras desordenadas o duplicadas"
print("error maximo de cobertura temporal:", max(errs), "s")   # debe ser ~1e-16
print("muestras por frame:", [len(seq.imu_between(seq.t_cam[k-1], seq.t_cam[k]))
                              for k in range(1,6)])
```

Deberías ver 11–12 muestras por frame (10 reales + los 2 extremos interpolados).

### Estructura de datos recomendada

No pases DataFrames al estimador. Convierte una vez a arrays y trabaja con un
generador:

```python
from dataclasses import dataclass

@dataclass
class Frame:
    k: int
    t: float
    img_path: str
    imu: np.ndarray        # (M,7) desde el frame anterior hasta este

def frames(seq, k0=0, k1=None):
    k1 = k1 or len(seq.cam)
    for k in range(k0+1, k1):
        yield Frame(k, seq.t_cam[k], seq.cam.path.iloc[k],
                    seq.imu_between(seq.t_cam[k-1], seq.t_cam[k]))
```

Esto es exactamente la interfaz que necesitarás en Fases 3, 5 y 6. Escríbela bien
ahora y no la vuelvas a tocar.

---

## 1.4 Visualización del ground truth

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa

gt = seq.gt
fig = plt.figure(figsize=(14, 5))

ax = fig.add_subplot(131, projection="3d")
ax.plot(gt.px, gt.py, gt.pz, lw=0.8)
ax.scatter(*gt[["px","py","pz"]].iloc[0], c="g", s=40, label="inicio")
ax.scatter(*gt[["px","py","pz"]].iloc[-1], c="r", s=40, label="fin")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
ax.set_title("Trayectoria GT"); ax.legend()
ax.set_box_aspect([np.ptp(gt.px), np.ptp(gt.py), np.ptp(gt.pz)])

ax2 = fig.add_subplot(132)
for c in ("px","py","pz"):
    ax2.plot(gt.t, gt[c], label=c)
ax2.set_xlabel("t [s]"); ax2.set_ylabel("m"); ax2.legend(); ax2.grid(alpha=.3)
ax2.set_title("Posición vs tiempo")

ax3 = fig.add_subplot(133)
speed = np.linalg.norm(gt[["vx","vy","vz"]].values, axis=1)
ax3.plot(gt.t, speed)
ax3.set_xlabel("t [s]"); ax3.set_ylabel("|v| [m/s]"); ax3.grid(alpha=.3)
ax3.set_title("Velocidad")
plt.tight_layout(); plt.show()

dist = np.linalg.norm(np.diff(gt[["px","py","pz"]].values, axis=0), axis=1).sum()
print(f"Distancia recorrida: {dist:.1f} m en {gt.t.iloc[-1]-gt.t.iloc[0]:.1f} s")
```

**Lo que debes observar en MH_01_easy:**

- Un tramo inicial de varios segundos con velocidad ≈ 0 (el dron en el suelo). Ese
  tramo es oro para estimar bias e inicializar la actitud.
- Velocidad de crucero moderada (≈1 m/s), sin picos agresivos. Por eso es "easy".
- Un recorrido de decenas de metros dentro del machine hall.

---

## 1.5 Inspección del IMU en bruto

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for c in ("wx","wy","wz"):
    axes[0].plot(seq.imu.t, seq.imu[c], lw=.4, label=c)
axes[0].set_ylabel("giro [rad/s]"); axes[0].legend(ncol=3); axes[0].grid(alpha=.3)
for c in ("ax","ay","az"):
    axes[1].plot(seq.imu.t, seq.imu[c], lw=.4, label=c)
axes[1].set_ylabel("accel [m/s²]"); axes[1].set_xlabel("t [s]")
axes[1].legend(ncol=3); axes[1].grid(alpha=.3)
plt.show()

# Ventana estática inicial
t_static_end = 1.0
m = seq.imu.t < t_static_end
bg0 = seq.imu.loc[m, ["wx","wy","wz"]].mean().values
a0  = seq.imu.loc[m, ["ax","ay","az"]].mean().values
print("bias gyro estimado:", bg0)
print("bias gyro GT      :", seq.gt[["bgx","bgy","bgz"]].iloc[0].values)
print("aceleracion media :", a0, " |a| =", np.linalg.norm(a0))
```

Observa que `az` en reposo **no** es ±9.81 exactamente: el IMU no está horizontal.
El vector `a0` normalizado te da la dirección de "arriba" en frame IMU, y eso es tu
inicialización de roll/pitch (el yaw no es observable con acelerómetro).

---

## 1.6 Interpolación del GT y test de cordura

```python
p, R, v, bg, ba = seq.gt_at(seq.t_cam[100])
print("pose GT en el frame 100:\n", seq.gt_T_ws(seq.t_cam[100]))
```

Test: la derivada numérica de la posición GT debe coincidir con `v_RS_R`.

```python
t = seq.t_gt
p = seq.gt[["px","py","pz"]].values
v_num = np.gradient(p, t, axis=0)
v_gt  = seq.gt[["vx","vy","vz"]].values
err = np.linalg.norm(v_num - v_gt, axis=1)
print("error mediano dp/dt vs v_GT:", np.median(err), "m/s")   # debe ser << 0.05
```

Si este test falla, tu lectura de columnas está mal o `v_RS_R` no está en el frame
que crees. Confirma antes de seguir: en Fase 3 vas a usar `v_RS_R` como estado
inicial y un frame equivocado te va a costar horas.

Otro test, más sutil, que valida rotación **y** velocidad a la vez:

```python
# w_medida - b_g debe reproducir la derivada de la orientación GT
from anexo_utils import Log
idx = 5000
t0, t1 = seq.t_gt[idx], seq.t_gt[idx+1]
_, R0, *_ = seq.gt_at(t0)
_, R1, *_ = seq.gt_at(t1)
w_gt = Log(R0.T @ R1) / (t1 - t0)          # velocidad angular en frame body
j = np.searchsorted(seq.t_imu, t0)
w_meas = seq.imu[["wx","wy","wz"]].values[j] - seq.gt[["bgx","bgy","bgz"]].values[idx]
print("w desde GT :", w_gt)
print("w medida   :", w_meas)              # deben parecerse a ~1e-2 rad/s
```

Si estos dos vectores no se parecen, el problema es de convenio de cuaternión
(`(w,x,y,z)` vs `(x,y,z,w)`) o de sentido de la rotación. **No pases de aquí.**

---

## 1.7 Trampas

- **Gyro antes que accel** en `imu0/data.csv`. En muchos otros formatos (y en los
  logs `.BIN` de ArduPilot que ya manejas) es al revés.
- Los nombres de columna del CSV llevan unidades y espacios; renombra por posición.
- `cam/data.csv` puede tener espacios en el nombre de archivo → `.strip()`.
- No hagas `merge_asof` de pandas para asociar IMU y cámara: te da *la muestra más
  cercana*, no *el intervalo*, y eso es lo que necesitas.
- El GT no cubre toda la secuencia en algunos datasets (TUM VI). En EuRoC sí, pero
  recórtalo siempre al solape común de los tres sensores.

---

## 1.8 Entregable de la fase

Un notebook `01_carga.ipynb` que:

1. Instancia `EurocSequence` e imprime el resumen.
2. Pasa los tres tests de cordura (cobertura temporal, `dp/dt == v`, `w == Log(R)/dt`).
3. Plotea trayectoria 3D, posición vs t, velocidad vs t, y señales IMU crudas.
4. Imprime bias inicial estimado vs GT.
