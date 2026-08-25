# Fase 6B — Camino de la optimización: factor graph con preintegración (GTSAM)

**Objetivo:** montar el back-end que usan VINS-Mono, ORB-SLAM3 y Kimera.
**Criterio de éxito:** un grafo con factores IMU preintegrados + factores de
proyección visual, optimizado con iSAM2, con ATE < 0.5 m sobre MH_01.

---

## 6B.1 Filtro vs optimización, en una tabla

| | MSCKF (filtro) | Factor graph |
|---|---|---|
| Linealización | Una vez, en la estimación actual | **Re-linealiza** en cada iteración |
| Estados pasados | Marginalizados (irrecuperables) | Se pueden re-optimizar |
| Coste | Constante y predecible | Variable; iSAM2 lo acota con actualizaciones incrementales |
| Precisión | Buena | Mejor (10–30 % típico en EuRoC) |
| Robustez a mala inicialización | Baja | Alta (con kernels robustos) |
| Sensibilidad a bugs de jacobiano | Diverge | Converge peor, pero converge |

El factor graph gana casi siempre. El filtro sigue vivo por coste computacional
acotado en plataformas embebidas — que es exactamente tu caso en ATALAYA.

---

## 6B.2 Preintegración de la IMU: el concepto

**El problema.** Entre dos keyframes hay ~200 muestras de IMU. Si metes cada una
como un estado, el grafo explota. Si integras la IMU para obtener un `Δpose`,
tendrías que **re-integrar las 200 muestras cada vez que el optimizador cambia la
actitud o el bias inicial** — porque la integración depende del estado inicial.

**La solución (Lupton & Sukkarieh; Forster et al. 2017).** Reescribir la integración
en un frame **relativo al primer keyframe**, de modo que el resultado sea
independiente del estado inicial:

```
ΔR_ij = Π_k Exp((w_k − b_g) Δt)
Δv_ij = Σ_k ΔR_ik (a_k − b_a) Δt
Δp_ij = Σ_k [ Δv_ik Δt + ½ ΔR_ik (a_k − b_a) Δt² ]
```

Estas tres cantidades **no dependen de `R_i`, `v_i`, `p_i`**. Solo del bias. Y para
el bias, en vez de re-integrar, se guarda el **jacobiano de primer orden**
`∂ΔR/∂b_g`, `∂Δv/∂b_a`, ... y se corrige linealmente:

```
ΔR_ij(b_g) ≈ ΔR_ij(b̄_g) · Exp( ∂ΔR/∂b_g · δb_g )
```

Solo hay que re-integrar si el bias cambia mucho (GTSAM lo hace con
`resetIntegrationAndSetBias`).

**El residuo del factor IMU** relaciona dos `NavState` (pose + velocidad) y el bias:

```
r_ΔR = Log( ΔR_ij(b̄)ᵀ · R_iᵀ R_j )
r_Δv = R_iᵀ (v_j − v_i − g Δt_ij) − Δv_ij(b̄)
r_Δp = R_iᵀ (p_j − p_i − v_i Δt_ij − ½ g Δt_ij²) − Δp_ij(b̄)
```

9 dimensiones. La covarianza sale de propagar el ruido del IMU a lo largo de la
preintegración. Todo esto está implementado en GTSAM; tú solo alimentas medidas.

Lee el paper de Forster, Carlone, Dellaert & Scaramuzza (T-RO 2017). Es la
referencia y es legible.

---

## 6B.3 Hola mundo: IMU + priors de pose

Antes de meter visión, valida la mecánica con "medidas de pose" tomadas del GT (esto
es el equivalente de `IMUKittiExampleGPS.py`). Si esto no funciona, el problema no
es la visión.

```python
import numpy as np, gtsam
from gtsam import (NonlinearFactorGraph, Values, Pose3, Rot3, Point3,
                   PriorFactorPose3, PriorFactorVector,
                   PreintegratedImuMeasurements, ImuFactor)
from gtsam.symbol_shorthand import X, V, B
from anexo_utils import EurocSequence

seq = EurocSequence("~/datasets/euroc/MH_01_easy")
ip  = seq.imu_params

# --- Parámetros de preintegración -----------------------------------------
params = gtsam.PreintegrationParams.MakeSharedU(9.81)   # U = mundo z-arriba, g=-9.81 z
params.setAccelerometerCovariance(np.eye(3) * ip.sigma_a**2)
params.setGyroscopeCovariance(np.eye(3) * ip.sigma_g**2)
params.setIntegrationCovariance(np.eye(3) * 1e-8)

def nav_prior_noise():
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01,0.01,0.01, 0.05,0.05,0.05]))    # rot(3) luego trad(3)

graph = NonlinearFactorGraph()
init  = Values()

k0, k1, stride = 40, 640, 5
kfs = list(range(k0, k1, stride))

# Estado inicial desde GT
p0, R0, v0, bg0, ba0 = seq.gt_at(seq.t_cam[k0])
pose0 = Pose3(Rot3(R0), Point3(*p0))
bias0 = gtsam.imuBias.ConstantBias(ba0, bg0)

graph.add(PriorFactorPose3(X(0), pose0, nav_prior_noise()))
graph.add(PriorFactorVector(V(0), v0,
          gtsam.noiseModel.Isotropic.Sigma(3, 0.05)))
graph.add(gtsam.PriorFactorConstantBias(B(0), bias0,
          gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)))
init.insert(X(0), pose0); init.insert(V(0), v0); init.insert(B(0), bias0)

state = gtsam.NavState(pose0, v0)
for i in range(1, len(kfs)):
    ta, tb = seq.t_cam[kfs[i-1]], seq.t_cam[kfs[i]]
    pim = PreintegratedImuMeasurements(params, bias0)
    chunk = seq.imu_between(ta, tb)
    for k in range(len(chunk)-1):
        dt = chunk[k+1,0] - chunk[k,0]
        if dt > 0:
            pim.integrateMeasurement(chunk[k,4:7], chunk[k,1:4], dt)

    graph.add(ImuFactor(X(i-1), V(i-1), X(i), V(i), B(0), pim))

    # "medida externa" de pose (aquí GT; en el sistema real, la visión)
    p, R, v, *_ = seq.gt_at(tb)
    graph.add(PriorFactorPose3(X(i), Pose3(Rot3(R), Point3(*p)),
              gtsam.noiseModel.Diagonal.Sigmas(
                  np.array([0.02,0.02,0.02, 0.1,0.1,0.1]))))

    state = pim.predict(state, bias0)          # buena inicialización
    init.insert(X(i), state.pose())
    init.insert(V(i), state.velocity())

result = gtsam.LevenbergMarquardtOptimizer(graph, init).optimize()
err = [np.linalg.norm(result.atPose3(X(i)).translation() - seq.gt_at(seq.t_cam[kfs[i]])[0])
       for i in range(len(kfs))]
print("error medio de posicion:", np.mean(err), "m")   # debe salir ~cm
```

**Puntos críticos:**

- **`integrateMeasurement(acc, gyro, dt)`** — acelerómetro **primero**, al revés que
  el orden de columnas del CSV de EuRoC. Error clásico.
- **`MakeSharedU(9.81)`** = "gravedad Up", es decir `g = (0,0,−9.81)`. Si tu mundo
  fuera z-abajo (NED), sería `MakeSharedD`. EuRoC es z-arriba → `U`.
- Las covarianzas que pasas son **densidades al cuadrado** (`σ²`), en unidades de
  continuo. GTSAM hace la conversión a discreto internamente con el `dt` de cada
  `integrateMeasurement`.
- El **orden de sigmas en `Pose3`** es `[rot(3), trans(3)]`. Al revés de lo que
  mucha gente asume.
- Usa **`pim.predict()`** para inicializar el siguiente estado. Un factor graph mal
  inicializado no converge; esto te lo da gratis.

### Variante recomendada: `CombinedImuFactor`

`ImuFactor` asume bias constante entre `i` y `j` y usa un único `B(0)`. Para estimar
el bias como random walk necesitas un `B(i)` por keyframe:

```python
params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
params.setAccelerometerCovariance(np.eye(3)*ip.sigma_a**2)
params.setGyroscopeCovariance(np.eye(3)*ip.sigma_g**2)
params.setIntegrationCovariance(np.eye(3)*1e-8)
params.setBiasAccCovariance(np.eye(3)*ip.sigma_ba**2)
params.setBiasOmegaCovariance(np.eye(3)*ip.sigma_bg**2)
params.setBiasAccOmegaInit(np.eye(6)*1e-5)

pim = gtsam.PreintegratedCombinedMeasurements(params, bias0)
...
graph.add(gtsam.CombinedImuFactor(X(i-1), V(i-1), X(i), V(i), B(i-1), B(i), pim))
init.insert(B(i), bias0)
```

Esto es lo correcto y es lo que usarás en el sistema final.

---

## 6B.4 Añadir la visión: factores de proyección

Ahora sustituye los priors de pose GT por observaciones reales de features.

```python
from gtsam import Cal3_S2, GenericProjectionFactorCal3_S2
from gtsam.symbol_shorthand import L

cam = seq.cam0
K = Cal3_S2(cam.fu, cam.fv, 0.0, cam.cu, cam.cv_)
body_P_sensor = gtsam.Pose3(cam.T_body_cam)     # T_BS: body <- cam. GTSAM lo espera así.
px_noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    gtsam.noiseModel.Isotropic.Sigma(2, 1.0))   # 1 px

# Para cada observación (keyframe i, landmark j, píxel uv) de tu KLTTracker:
graph.add(GenericProjectionFactorCal3_S2(
    gtsam.Point2(uv[0], uv[1]), px_noise, X(i), L(j), K, body_P_sensor))
```

**Qué tienes que resolver tú:**

1. **Qué landmarks incluir.** Solo los vistos en ≥ 3 keyframes y con paralaje
   suficiente. Un landmark con 2 observaciones y poca paralaje mete un modo plano en
   el problema y rompe la optimización.
2. **Inicializar `L(j)`.** Triangula con las poses actuales (`cv2.triangulatePoints`
   o `gtsam.triangulatePoint3`). Un landmark mal inicializado no converge.
3. **Fijar el gauge.** Con solo visión + IMU, la posición y el yaw globales son
   libres. La gravedad fija roll/pitch. Ancla el primer keyframe con un
   `PriorFactorPose3` fuerte.
4. **Kernel robusto.** Huber sobre el ruido de píxel. Sin él, un outlier del KLT
   arrastra toda la solución.

### Alternativa: smart factors (estructura eliminada)

GTSAM implementa el equivalente al truco del MSCKF: los **smart projection factors**
marginalizan el landmark con complemento de Schur y no lo añaden al grafo.

```python
smart_params = gtsam.SmartProjectionParams()
sf = gtsam.SmartProjectionPose3Factor(px_noise, K, body_P_sensor, smart_params)
for (i, uv) in observaciones_de_la_feature:
    sf.add(gtsam.Point2(*uv), X(i))
graph.add(sf)
```

Ventajas: menos variables, sin inicialización de landmarks, degeneraciones tratadas
internamente. Es lo que usa Kimera-VIO.

> Nota práctica: los nombres exactos de las clases de smart factors varían entre
> versiones del binding de Python. Comprueba con
> `[n for n in dir(gtsam) if "Smart" in n]` antes de escribir código.

---

## 6B.5 iSAM2: hacerlo incremental

Optimizar el grafo entero en cada keyframe es `O(n³)`. **iSAM2** reordena y
re-linealiza solo la parte del árbol afectada por los factores nuevos.

```python
isam_params = gtsam.ISAM2Params()
isam_params.setRelinearizeThreshold(0.01)
isam_params.relinearizeSkip = 1
isam = gtsam.ISAM2(isam_params)

for i, kf in enumerate(kfs):
    new_factors = NonlinearFactorGraph()
    new_values  = Values()
    # ... añadir factores IMU + visuales y valores iniciales del keyframe i ...
    isam.update(new_factors, new_values)
    isam.update()                      # una iteración extra ayuda a la convergencia
    est = isam.calculateEstimate()
    pose_i = est.atPose3(X(i))
```

Puntos que te van a morder:

- **Nunca insertes un valor dos veces.** iSAM2 lanza `ValuesKeyAlreadyExists`.
  Lleva un `set` de claves ya insertadas.
- **Todo símbolo que aparezca en un factor debe tener valor inicial** en el mismo
  `update()` o en uno anterior. Si no: `IndeterminantLinearSystemException`.
- Esa misma excepción también aparece cuando el sistema está **underconstrained**:
  landmark con una sola observación, keyframe sin factor IMU que lo conecte, o gauge
  no fijado. Cuando salte, GTSAM te dice la clave culpable — úsala.

---

## 6B.6 Ventana deslizante (fixed-lag)

Para tiempo real acotado, usa `gtsam.BatchFixedLagSmoother` o
`IncrementalFixedLagSmoother`, que marginalizan lo que sale de la ventana:

```python
lag = 3.0     # segundos
smoother = gtsam.IncrementalFixedLagSmoother(lag, isam_params)
timestamps = gtsam.FixedLagSmootherKeyTimestampMap()
timestamps.insert((X(i), t_i)); timestamps.insert((V(i), t_i)); timestamps.insert((B(i), t_i))
smoother.update(new_factors, new_values, timestamps)
```

Esto es esencialmente lo que hace VINS-Mono con su ventana de 10 keyframes.

---

## 6B.7 Arquitectura final sugerida

```
KLTTracker (Fase 4)
      │  observaciones con ID persistente
      ▼
Selector de keyframes  ──► paralaje > 15 px  o  tracked < 60  o  Δt > 0.5 s
      │
      ▼
Constructor del grafo:
      ├── CombinedImuFactor(X_{i-1},V_{i-1},B_{i-1} → X_i,V_i,B_i)
      ├── SmartProjectionFactor por landmark con ≥3 obs
      └── PriorFactor en X(0), V(0), B(0)
      │
      ▼
IncrementalFixedLagSmoother (lag 2–4 s)
      │
      ▼
Trayectoria → TUM → evo (Fase 7)
```

---

## 6B.8 Trampas

| Trampa | Síntoma |
|---|---|
| `integrateMeasurement(gyro, acc, dt)` invertido | Trayectoria disparatada desde el primer factor |
| `MakeSharedD` en vez de `MakeSharedU` | Todo cae/sube; error ~2g |
| Sigmas de `Pose3` en orden `[trans, rot]` | Prior mal ponderado, convergencia lenta |
| `body_P_sensor` invertido | Error sistemático de ~7 cm y actitud sesgada |
| Landmarks con 1–2 observaciones | `IndeterminantLinearSystemException` |
| Sin kernel robusto | Un outlier destruye la solución completa |
| Inicializar sin `pim.predict()` | LM no converge o converge a un mínimo local |
| Ventana `lag` muy corta (<1 s) | Marginaliza antes de que el bias sea observable |

---

## 6B.9 Recursos

- **Ejemplos oficiales del binding Python de GTSAM**: `ImuFactorExample.py`,
  `ImuFactorISAM2Example.py`, `IMUKittiExampleGPS.py`, `VisualISAM2Example.py`.
  Están en `python/gtsam/examples` del repo.
- **"GTSAM by Example"** (`https://gtbook.github.io/gtsam-examples/`) — el
  libro-web, con notebooks ejecutables.
- **Forster et al., "On-Manifold Preintegration for Real-Time Visual-Inertial
  Odometry"**, IEEE T-RO 2017 — la referencia teórica.
- **VINS-Mono** (`HKUST-Aerial-Robotics/VINS-Mono`) — lee `estimator.cpp` y
  `initial_alignment.cpp` aunque sea C++: la lógica de inicialización y ventana es
  transferible directamente.

---

## 6B.10 Entregable

1. El "hola mundo" IMU + priors GT convergiendo a error de centímetros.
2. Versión con `CombinedImuFactor` y bias por keyframe; plot de bias estimado vs GT.
3. Versión con factores visuales reales (proyección o smart), sin GT en el grafo
   salvo el prior inicial.
4. Migración a `IncrementalFixedLagSmoother` y medida del tiempo por keyframe.
5. ATE/RPE con `evo` y comparación contra tu ESKF de la Fase 5 y tu MSCKF de 6A.
