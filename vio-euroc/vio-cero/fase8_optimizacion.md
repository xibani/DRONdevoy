# Fase 8 — Optimización y refactor

**Objetivo:** convertir un prototipo que funciona en MH_01 en un sistema que
funciona en secuencias difíciles y arranca solo.
**Criterio de éxito:** el sistema inicializa sin GT, converge en < 5 s, y produce
ATE < 1 m en MH_05 y V1_03.

Esta fase no tiene un final: es donde vive la investigación. Va ordenada por
relación mejora/esfuerzo.

---

## 8.1 Refactor: la arquitectura que quieres

```
vio/
├── data/
│   └── euroc.py            # EurocSequence (Fase 1)
├── geometry/
│   ├── so3.py              # Exp, Log, skew, jacobianos derechos
│   ├── se3.py              # composición, adjunta
│   └── camera.py           # PinholeCamera, undistort, proyección + jacobiano
├── frontend/
│   ├── tracker.py          # KLTTracker con IDs
│   ├── keyframe.py         # política de selección
│   └── outliers.py         # RANSAC, forward-backward, gating
├── backend/
│   ├── eskf.py             # Fase 5
│   ├── msckf.py            # Fase 6A
│   └── graph.py            # Fase 6B (GTSAM)
├── init/
│   └── vi_init.py          # inicialización visual-inercial (§8.2)
├── eval/
│   └── metrics.py          # export TUM, wrappers de evo
└── configs/
    └── euroc.yaml          # TODOS los hiperparámetros, ninguno hardcodeado
```

Dos reglas que pagan solas:

1. **Ningún número mágico en el código.** Todo a `configs/*.yaml`. Un barrido de
   hiperparámetros con números hardcodeados es imposible.
2. **Tests unitarios sobre los jacobianos.** Comparación analítico vs numérico con
   `pytest`. Cuando toques la geometría, un test roto te dice dónde en segundos.

```python
# tests/test_jacobians.py
import numpy as np, pytest
from vio.geometry.camera import project, project_jacobian

def numeric_jac(f, x, eps=1e-7):
    f0 = f(x); J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        d = np.zeros_like(x); d[i] = eps
        J[:, i] = (f(x + d) - f0) / eps
    return J

@pytest.mark.parametrize("seed", range(20))
def test_projection_jacobian(seed):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=3) + np.array([0, 0, 5.0])
    J_num = numeric_jac(project, p)
    J_ana = project_jacobian(p)
    assert np.allclose(J_ana, J_num, rtol=1e-4, atol=1e-6)
```

---

## 8.2 Inicialización visual-inercial (lo más importante)

Hasta ahora has arrancado con estado y bias del GT. En un sistema real hay que
estimar, a partir de unos segundos de datos: **bias del giro, gravedad (dirección),
velocidades de los keyframes y escala**.

El método de VINS-Mono (`initial_alignment.cpp`) es el estándar y se hace en cuatro
pasos desacoplados:

### Paso 1 — SfM visual sobre una ventana

Construye un mapa local con ~10 keyframes usando VO puro (esencial + triangulación +
PnP + bundle adjustment pequeño). Obtienes poses **hasta escala** `s·p̄_ck` y
rotaciones `R̄_ck` (estas sí, métricas: la rotación no tiene ambigüedad de escala).

### Paso 2 — Bias del giróscopo

Las rotaciones visuales son fiables. La preintegración del giro debe reproducirlas:

```
min_{b_g} Σ_k ‖ Log( (R̄_ck^T R̄_c,k+1)^T · ΔR_{k,k+1}(b_g) ) ‖²
```

Linealizando `ΔR(b_g) ≈ ΔR(0)·Exp(J_g δb_g)` queda un sistema lineal 3×3. Se resuelve
en una iteración y da un `b_g` excelente.

**Este paso es el que más rendimiento da por línea de código.** Recuerda la Fase 3:
el bias del giro es el que domina la deriva.

### Paso 3 — Velocidad, gravedad y escala (lineal)

Con `b_g` fijado, re-preintegra y plantea, para cada par de keyframes consecutivos,
las restricciones de preintegración de posición y velocidad. Las incógnitas son:

```
X = [ v_b0, v_b1, ..., v_bn, g_c0 (3), s (1) ]     dimensión 3(n+1) + 4
```

y las ecuaciones son **lineales** en `X`:

```
Δp_k = R_bk^T ( s(p̄_{k+1} − p̄_k) − v_bk Δt + ½ g Δt² ) + (términos de extrínseca)
Δv_k = R_bk^T ( v_{k+1} − v_k + g Δt )
```

Un `lstsq` y listo. La escala métrica sale de aquí.

### Paso 4 — Refinar la gravedad

`g` tiene norma conocida (9.81), así que solo tiene 2 grados de libertad. Se
reparametriza en la tangente de la esfera (`g = ĝ·9.81 + b1·w1 + b2·w2`) y se
re-resuelve el sistema 2–3 veces. Esto mejora notablemente la escala.

### Condiciones para que funcione

- **Excitación**: hace falta aceleración no nula en varios ejes. Un dron en hover no
  inicializa nunca (la escala no es observable con aceleración constante). Detecta
  la excitación con la varianza del acelerómetro y **espera** si es baja.
- Paralaje suficiente en la ventana visual.
- Suele necesitar 0.5–2 s de datos. VINS-Mono repite hasta que converge.

**Test de la inicialización:** compara `s`, `g` y `b_g` estimados con el GT. Errores
aceptables: escala < 5 %, gravedad < 2°, bias del giro < 0.002 rad/s.

---

## 8.3 Estimación de bias online y calibración

- **Bias en el estado** ya lo tienes (ESKF/MSCKF) o vía `CombinedImuFactor` (GTSAM).
  Lo que falta es **vigilarlo**: si el bias del acelerómetro se aleja más de
  ~0.5 m/s² del inicial, algo va mal (probablemente estás absorbiendo un error de
  gravedad o de escala en el bias).
- **Calibración online de la extrínseca `T_cam_imu`.** VINS-Mono y OpenVINS la
  estiman. En EuRoC no hace falta (la calibración es buena) pero en ATALAYA sí:
  añade `δθ_ci`, `δp_ci` (6 estados) y sus jacobianos. Ganancia grande cuando la
  calibración de partida es mediocre.
- **Time offset cámara-IMU `t_d`.** Un estado escalar más. El jacobiano se obtiene
  de la velocidad de la feature en la imagen:
  `∂z/∂t_d = −v_píxel`. En un setup casero es la mejora individual más grande.

---

## 8.4 Marginalización y ventana deslizante

En un filtro, marginalizar un clon es borrar filas/columnas de `P` (Fase 6A).
En un optimizador es más sutil: hay que **convertir los factores que tocan el estado
eliminado en un prior sobre los que quedan**, vía complemento de Schur:

```
[ H_mm  H_mr ] [ δx_m ]   [ b_m ]
[ H_rm  H_rr ] [ δx_r ] = [ b_r ]

H_prior = H_rr − H_rm H_mm⁻¹ H_mr
b_prior = b_r  − H_rm H_mm⁻¹ b_m
```

Ese `H_prior` se añade como factor lineal denso sobre los estados restantes. Tres
consecuencias que hay que conocer:

1. **`H_prior` es denso** — conecta todo con todo. Por eso la ventana se mantiene
   pequeña (10–15 keyframes).
2. **Está linealizado en el punto de marginalización** y no se puede re-linealizar.
   De aquí vienen las **inconsistencias** del sliding window BA, y por eso VINS-Mono
   aplica FEJ a los estados que aparecen en el prior.
3. **Qué marginalizar importa**: VINS-Mono marginaliza el keyframe más antiguo si el
   penúltimo es keyframe, y si no, descarta directamente el penúltimo frame
   (sin marginalizar) para no meter información redundante.

En GTSAM, `IncrementalFixedLagSmoother` hace todo esto por ti.

---

## 8.5 Robustez para secuencias difíciles

MH_04, MH_05, V1_03, V2_03 tienen motion blur, iluminación pobre y dinámica
agresiva. Lo que hay que añadir:

| Problema | Solución |
|---|---|
| Motion blur | Detectar (varianza del laplaciano) y saltar el frame; la IMU cubre el hueco |
| Iluminación variable | Ecualización adaptativa (CLAHE) antes del KLT |
| Pocas features | Bajar `qualityLevel`, subir `maxLevel` del KLT, usar rejilla para forzar distribución espacial |
| Movimiento rápido | **Predecir la posición de la feature con la IMU** y darle esa `initialFlow` al KLT |
| Rotación pura | Detectar por paralaje y no actualizar traslación |
| Outliers persistentes | Gating χ² + kernel de Huber + prueba de dos vistas con la rotación de la IMU |

La predicción KLT con IMU es la mejora de robustez más rentable:

```python
# Predice dónde estará la feature usando la rotación integrada de la IMU
R_c1c0 = T_c_i[:3,:3] @ R_i0i1.T @ T_i_c[:3,:3]     # rotación entre cámaras
H = cam.K @ R_c1c0 @ np.linalg.inv(cam.K)           # homografía de rotación pura
p_pred = (H @ np.hstack([p_prev, np.ones((len(p_prev),1))]).T)
p_pred = (p_pred[:2] / p_pred[2]).T.astype(np.float32)

cur, st, _ = cv2.calcOpticalFlowPyrLK(
    img_prev, img_cur, p_prev.reshape(-1,1,2), p_pred.reshape(-1,1,2),
    flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk_params)
```

Esto es exactamente la homografía de rotación pura que ya usas en el régimen planar
de ATALAYA, aplicada aquí como *predictor* en vez de como *estimador*.

---

## 8.6 Escalada a otros datasets

**TUM VI** — segundo dataset natural. Diferencias que te van a costar tiempo:

- Cámaras **fisheye 1024×1024**: el modelo pinhole+radtan no sirve. Usa
  **Kannala-Brandt** (`cv2.fisheye`) o **doble esfera** (el que da TUM). Tu
  `PinholeCamera` necesita una abstracción de modelo de cámara.
- **Calibración fotométrica** disponible (respuesta, viñeteado): usarla mejora el
  KLT notablemente.
- El GT de mocap solo cubre los tramos inicial y final (salvo las secuencias
  `room`). Empieza por `room1`–`room6`, que tienen GT completo.

**KITTI raw** — el caso coche. IMU a ~100 Hz y GT de RTK. Movimiento casi planar y
velocidad alta: el VIO monocular sufre porque la escala es débilmente observable con
aceleración baja y trayectoria casi rectilínea. Buen ejemplo pedagógico de un caso
en el que el VIO no brilla.

**UZH-FPV** — el reto. Dinámica de drone racing que rompe la mayoría de estimadores.
Directamente relevante para ATALAYA si algún día vuelas agresivo. No lo intentes
hasta tener MH_05 y V1_03 sólidos.

**TartanAir** — sintético, GT perfecto, sin IMU nativa (se genera ajustando splines a
las poses y diferenciando, p. ej. con `ov_sim` de OpenVINS). Ideal si quieres
explorar híbridos deep+geométrico con datos ilimitados. Ojo a la convención NED.

---

## 8.7 Extensiones naturales

Por orden de coste creciente:

1. **Estéreo.** Con `cam1` la escala es observable desde el primer frame, la
   inicialización deja de ser un problema y el ATE mejora ~2×. Es la extensión con
   mejor relación resultado/esfuerzo.
2. **Detección de loop closure** (DBoW2 / NetVLAD) + pose graph optimization. Elimina
   la deriva global. Convierte tu VIO en VI-SLAM.
3. **Deep learning en el bucle.** Aquí es donde tu perfil aporta:
   - **TLIO** (Liu et al., RA-L 2020): una red regresa desplazamiento 3D + su
     incertidumbre a partir de la IMU, y eso alimenta un EKF tightly-coupled.
     Reportan una reducción de ~33 % en deriva de posición y ~27 % en yaw frente a
     la baseline 3D-RoNIN. Es la línea más prometedora para tu caso: **funciona sin
     cámara**, lo que da un modo de respaldo cuando el visual falla.
   - **RoNIN / IONet**: odometría inercial aprendida, base conceptual de TLIO.
   - Front-ends aprendidos: **SuperPoint + SuperGlue** o **XFeat** en lugar de
     KLT/ORB. Mejora la robustez a blur y cambios de iluminación.
   - **DBA-Fusion** y similares: bundle adjustment denso acoplado con IMU.
4. **Calibración online completa** (extrínsecas, intrínsecas, time offset).

---

## 8.8 Checklist de "sistema terminado"

- [ ] Inicializa solo, sin GT, en < 5 s, y detecta cuando no puede (falta excitación).
- [ ] Config en YAML, cero números mágicos.
- [ ] Tests de jacobianos analíticos vs numéricos en CI.
- [ ] Corre en MH_01–MH_05 y V1_01–V1_03 sin cambiar parámetros.
- [ ] Reporta ATE, RPE, deriva %, NEES y tiempo/frame automáticamente.
- [ ] Degrada con gracia: sin features, propaga con IMU y lo señala en la salida.
- [ ] Publica covarianza junto con la pose (para el consumidor aguas abajo).
- [ ] Tiempo por frame estable y medido, no medio.

---

## 8.9 Lecturas para esta fase

- Qin, Li & Shen, **"VINS-Mono: A Robust and Versatile Monocular Visual-Inertial
  State Estimator"** (T-RO 2018) — la inicialización y la marginalización, bien
  explicadas.
- Huang, Mourikis & Roumeliotis sobre **observabilidad y FEJ** — por qué el EKF-VIO
  es inconsistente y cómo se arregla.
- Delmerico & Scaramuzza, **"A Benchmark Comparison of Monocular Visual-Inertial
  Odometry Algorithms for Flying Robots"** (ICRA 2018) — compara MSCKF, OKVIS,
  ROVIO, VINS-Mono y SVO+MSF en precisión **y en recursos**. Lectura obligada antes
  de elegir qué llevar a hardware embebido.
- Documentación de **OpenVINS** (`docs.openvins.com`) — las derivaciones de
  jacobianos más limpias que hay publicadas.
- Solà, **"Quaternion kinematics for the error-state Kalman filter"**
  (arXiv:1711.02508) — vuelve a él cada vez que dudes de un jacobiano.
