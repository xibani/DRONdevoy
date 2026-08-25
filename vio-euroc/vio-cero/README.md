# Curso práctico de Visual-Inertial Odometry (VIO) sobre EuRoC

**De cero a un MSCKF / factor graph funcionando, en Python, con evaluación cuantitativa.**

Este curso sigue exactamente las 8 fases que planteaste. Cada fase es un archivo
independiente con: teoría mínima necesaria, código ejecutable, trampas conocidas
(las que te van a costar horas si no te avisan) y un **criterio de éxito** para
saber si puedes pasar a la siguiente.

---

## Índice

| Fase | Archivo | Qué consigues | Tiempo est. |
|---|---|---|---|
| 0 | [`fase0_preparacion.md`](fase0_preparacion.md) | Entorno, dataset descargado y verificado | 1–2 h |
| 1 | [`fase1_carga_sincronizacion.md`](fase1_carga_sincronizacion.md) | Loader EuRoC + asociación IMU↔cámara + GT ploteado | 2–3 h |
| 2 | [`fase2_calibracion.md`](fase2_calibracion.md) | `K`, distorsión, `T_cam_imu`, undistort validado | 2–3 h |
| 3 | [`fase3_dead_reckoning.md`](fase3_dead_reckoning.md) | Integración pura de IMU y su deriva medida | 4–6 h |
| 4 | [`fase4_frontend_visual.md`](fase4_frontend_visual.md) | VO monocular KLT + esencial + `recoverPose` | 6–8 h |
| 5 | [`fase5_escala_y_ekf.md`](fase5_escala_y_ekf.md) | Escala métrica + ESKF loosely-coupled | 8–12 h |
| 6A | [`fase6a_msckf.md`](fase6a_msckf.md) | MSCKF (filtro tightly-coupled) entendido línea a línea | 15–25 h |
| 6B | [`fase6b_factor_graph_gtsam.md`](fase6b_factor_graph_gtsam.md) | Factor graph con preintegración + iSAM2 | 12–20 h |
| 7 | [`fase7_evaluacion.md`](fase7_evaluacion.md) | ATE/RPE con `evo`, protocolo reproducible | 3–4 h |
| 8 | [`fase8_optimizacion.md`](fase8_optimizacion.md) | Inicialización VI, bias online, marginalización | abierto |

Anexos:

- [`anexo_convenios.md`](anexo_convenios.md) — **léelo antes que nada**. El 80 % de los bugs
  en VIO son de convenios: quién multiplica a quién, JPL vs Hamilton, signo de la
  gravedad, `T_BS` vs `T_SB`.
- [`anexo_utils.py`](anexo_utils.py) — módulo con las funciones que se repiten en
  todas las fases (loader, SO(3), interpolación, export TUM). Impórtalo desde tus
  notebooks en vez de copiar-pegar.

---

## Filosofía del curso

**Regla 1: nunca avances con un componente sin validar.** VIO es un sistema donde
tres errores pequeños se combinan en un resultado que diverge sin decirte por qué.
Cada fase tiene un test contra ground truth precisamente para que el bug se
localice en la fase donde nació.

**Regla 2: el ground truth es tu instrumento de laboratorio, no tu entrada.**
Durante el aprendizaje lo vas a usar para inicializar estados, para verificar
jacobianos, para medir escala. Está bien. Lo que no vale es usarlo dentro del
estimador y luego reportar el ATE.

**Regla 3: escribe el estimador con estado de error (ESKF), no con estado nominal.**
La orientación vive en SO(3), no en R³. Si intentas meter un cuaternión de 4
componentes en un EKF estándar con covarianza 4×4 vas a tener una matriz singular
y una semana perdida. Fase 3 y el anexo de convenios te fuerzan a esto desde el
principio.

**Regla 4: mide siempre en unidades físicas.** ATE-RMSE en metros, deriva en
%-de-distancia-recorrida, RPE en m/s. "Se ve bien en el plot" no es un resultado.

---

## Qué esperar como resultado (referencias de MH_01_easy)

Números orientativos de ATE-RMSE tras alineamiento SE(3) (Sim(3) para monocular),
sobre MH_01_easy (≈80 m de recorrido, ≈3 min):

| Método | ATE-RMSE típico | Comentario |
|---|---|---|
| IMU dead reckoning (Fase 3) | decenas–cientos de m a los 30 s | Diverge. Es el resultado esperado. |
| VO mono KLT+esencial (Fase 4) | escala libre; deriva 5–20 % | Sin escala métrica. |
| ESKF loosely-coupled (Fase 5) | 0.3–1.5 m | Tu primer resultado "real". |
| MSCKF propio (Fase 6A) | 0.15–0.5 m | Ya compite con literatura antigua. |
| Factor graph GTSAM (Fase 6B) | 0.1–0.4 m | Mejor con ventana + iSAM2. |
| OpenVINS / VINS-Mono (referencia) | ~0.07–0.15 m | Estado del arte afinado. |

Si tu MSCKF casero llega a 0.3 m en MH_01, has entendido el algoritmo. No persigas
el 0.07 m: eso es afinado de outliers, inicialización y calibración online.

---

## Conexión con ATALAYA

Dos cosas de este curso se transfieren directamente a tu stack de navegación
GPS-denied:

1. **La mecánica del ESKF y la preintegración** (Fases 3, 5, 6B) es exactamente la
   que necesitas para fusionar tu flujo óptico métrico con la IMU del autopiloto en
   vez de tratarlos como fuentes independientes.
2. **El protocolo de evaluación** (Fase 7) — exportar a TUM, alinear con Umeyama,
   reportar ATE/RPE — es el mismo que ya usas y te da una métrica comparable entre
   tu pipeline de homografía y cualquier baseline publicada.

La diferencia clave: EuRoC es 6-DoF general en interior; tu régimen de homografía
asume escena plana y vuelo a altura conocida. El MSCKF es el puente entre ambos si
algún día quieres soltar la hipótesis de planaridad.
