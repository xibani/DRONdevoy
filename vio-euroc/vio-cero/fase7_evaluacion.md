# Fase 7 — Evaluación

**Objetivo:** un protocolo de evaluación reproducible y honesto.
**Criterio de éxito:** un script que, dado un `.tum` de tu estimador, produce
ATE-RMSE, RPE y las figuras, sin intervención manual.

Esta fase es corta pero es la que convierte "he programado un VIO" en "he medido un
VIO". Sin ella no puedes iterar: no sabrás si un cambio mejora o empeora.

---

## 7.1 Formato TUM

```
timestamp tx ty tz qx qy qz qw
```

- Tiempo en **segundos**, float.
- Cuaternión **Hamilton, escalar al final** `(qx, qy, qz, qw)`. EuRoC lo da con
  escalar al principio: hay que reordenar.
- Un espacio como separador, sin cabecera, comentarios con `#`.
- La pose es la del **body**, no la de la cámara. Si tu estimador reporta poses de
  cámara, conviértelas: `T_w_body = T_w_cam @ inv(T_body_cam)`.

```python
from anexo_utils import save_tum, euroc_gt_to_tum, inv_T
import numpy as np

# Estimación (poses del body)
save_tum("est.tum", times, poses_body)

# Ground truth con la MISMA base de tiempo
euroc_gt_to_tum(seq, "gt.tum", t_lo=times[0]-0.1, t_hi=times[-1]+0.1)
```

Atajo: `evo` entiende el CSV de EuRoC nativamente.

```bash
evo_traj euroc ~/datasets/euroc/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
         --save_as_tum
```

⚠️ Si usas esto, el GT queda con timestamps **absolutos en segundos desde epoch**,
mientras que tu estimación (con el offset restado, como manda la Fase 1) empieza en
~0. `evo` no encontrará ninguna correspondencia y te dirá que no hay poses comunes.
Solución: exporta tu estimación **sumando el offset** (`save_tum(..., t_offset_ns=seq.t0_ns)`)
o exporta el GT con `euroc_gt_to_tum`, que ya usa tu misma base. Elige una y sé
consistente.

---

## 7.2 ATE: error absoluto de trayectoria

Mide la **consistencia global**: cuánto se parece tu trayectoria completa a la real,
tras alinear ambas rígidamente.

```bash
# Monocular con escala libre -> Sim(3)
evo_ape tum gt.tum est.tum -as --plot --plot_mode xyz --save_results ate_mono.zip

# Con escala métrica (VIO) -> SE(3)
evo_ape tum gt.tum est.tum -a --plot --plot_mode xyz --save_results ate_vio.zip
```

| Flag | Significado |
|---|---|
| `-a` / `--align` | Alineamiento **SE(3)** por Umeyama (rotación + traslación) |
| `-s` / `--correct_scale` | Añade escala → **Sim(3)** |
| `-as` | Ambos (Sim(3) completo) |
| `-r trans_part` | Solo la parte de traslación (por defecto en APE) |
| `-r angle_deg` | Error de rotación en grados |
| `--t_max_diff 0.02` | Tolerancia de asociación temporal |

**Regla de honestidad**, y esto importa mucho cuando reportes resultados:

- **VO monocular puro** (Fase 4) → Sim(3) (`-as`). La escala no es observable, es
  legítimo ajustarla.
- **VIO** (Fases 5, 6) → **SE(3)** (`-a`). La escala **sí** es observable gracias a
  la IMU; corregirla enmascara un error real. Usar `-as` en un VIO es hacer trampa,
  y en la literatura se marca explícitamente cuando se hace.

Salida típica:

```
       max      1.283271
      mean      0.312664
    median      0.287441
       min      0.041902
      rmse      0.354117
       sse      41.28
       std      0.166122
```

**El número que reportas es el RMSE.**

---

## 7.3 RPE: error relativo de pose

Mide la **precisión local** (deriva por unidad de recorrido), insensible a la deriva
acumulada global. Es la métrica que de verdad te dice si tu odometría es buena.

```bash
# Deriva por metro recorrido
evo_rpe tum gt.tum est.tum -r trans_part --delta 1 --delta_unit m -a --plot

# Deriva angular por metro
evo_rpe tum gt.tum est.tum -r angle_deg --delta 1 --delta_unit m -a

# Por segundo (útil para dron)
evo_rpe tum gt.tum est.tum -r trans_part --delta 1 --delta_unit s -a
```

`--all_pairs` promedia sobre todos los pares con ese delta en vez de una partición;
es más estable estadísticamente.

**Interpretación conjunta:**

| ATE | RPE | Diagnóstico |
|---|---|---|
| Alto | Bajo | Odometría local buena, deriva acumulada. Normal en VIO sin loop closure. |
| Bajo | Alto | Improbable; suele indicar sobreajuste al GT o error de asociación temporal. |
| Alto | Alto | Bug o mala sintonización. |
| Bajo | Bajo | Enhorabuena. |

---

## 7.4 Comparar varios métodos

```bash
evo_ape tum gt.tum eskf.tum   -a --save_results res/eskf.zip
evo_ape tum gt.tum msckf.tum  -a --save_results res/msckf.zip
evo_ape tum gt.tum gtsam.tum  -a --save_results res/gtsam.zip

evo_res res/*.zip --use_filenames --plot --save_table res/tabla.csv
```

Y la comparación visual de trayectorias:

```bash
evo_traj tum eskf.tum msckf.tum gtsam.tum --ref gt.tum \
         -a --plot_mode xy --plot --save_plot res/trayectorias
```

---

## 7.5 API de Python (para automatizar barridos)

```python
from evo.core import sync, metrics
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
import copy

def evaluate(gt_file, est_file, align=True, correct_scale=False):
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est,
                                                     max_diff=0.02)
    traj_est_al = copy.deepcopy(traj_est)
    traj_est_al.align(traj_ref, correct_scale=correct_scale)

    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est_al))

    rpe = metrics.RPE(metrics.PoseRelation.translation_part,
                      delta=1.0, delta_unit=metrics.Unit.meters,
                      all_pairs=True)
    rpe.process_data((traj_ref, traj_est_al))

    return {
        "ate_rmse": ape.get_statistic(metrics.StatisticsType.rmse),
        "ate_mean": ape.get_statistic(metrics.StatisticsType.mean),
        "ate_max":  ape.get_statistic(metrics.StatisticsType.max),
        "rpe_rmse": rpe.get_statistic(metrics.StatisticsType.rmse),
        "n_poses":  traj_est.num_poses,
        "length_m": traj_ref.path_length,
    }
```

Con esto puedes hacer barridos:

```python
import itertools, pandas as pd
rows = []
for sigma_p, q_factor in itertools.product([0.02, 0.05, 0.1], [1, 3, 10]):
    t, poses, _ = run_eskf(seq, sigma_p=sigma_p, q_factor=q_factor)
    save_tum("tmp.tum", t, poses)
    r = evaluate("gt.tum", "tmp.tum")
    rows.append({"sigma_p": sigma_p, "q": q_factor, **r})
print(pd.DataFrame(rows).sort_values("ate_rmse"))
```

**Advertencia metodológica:** si sintonizas hiperparámetros maximizando el ATE sobre
MH_01 y luego reportas el ATE de MH_01, estás reportando error de entrenamiento.
Sintoniza en MH_01/MH_02 y reporta en MH_03/MH_04/V1_02. Es exactamente el mismo
problema de fuga de datos que ya manejas en ML.

---

## 7.6 Métricas adicionales que merece la pena reportar

```python
# Deriva relativa: ATE / distancia recorrida
drift_pct = 100 * ate_rmse / traj_ref.path_length

# Consistencia del filtro: NEES promedio (si tienes covarianza)
#   NEES = e^T P^-1 e ;  debe estar cerca de la dimensión (3 para posición)
#   NEES >> 3  -> filtro sobreconfiado (típico sin FEJ)
#   NEES << 3  -> filtro pesimista

# Tiempo de cómputo por frame
print(f"{1000*np.mean(dts):.1f} ms/frame  ->  {1/np.mean(dts):.0f} Hz")
```

El **NEES** es la métrica que separa "funciona" de "funciona y sabe cuánto se
equivoca". Para navegación autónoma real, un filtro sobreconfiado es peligroso
aunque su ATE sea bueno: el planificador se fiará de una posición que no merece
confianza. Esto es directamente relevante para ATALAYA.

---

## 7.7 Plantilla de reporte

Para cada experimento, registra:

```
Secuencia:        MH_01_easy
Tramo:            frames 40–1400  (t = 2.0–70.0 s, 62.3 m recorridos)
Método:           ESKF loosely-coupled, keyframe cada 2 frames
Inicialización:   estado y bias del GT en t=2.0 s
Alineamiento:     SE(3) (Umeyama, sin escala)
--------------------------------------------------
ATE-RMSE:         0.412 m      (0.66 % de la distancia)
ATE-mediana:      0.351 m
ATE-max:          1.284 m
RPE (1 m):        0.038 m      RPE angular (1 m): 0.41 °
NEES posición:    5.2          (sobreconfiado)
Tiempo:           18.3 ms/frame (54 Hz)
Medidas acept.:   87 %
```

Si no puedes rellenar todas las líneas, tu experimento no es reproducible.

---

## 7.8 Trampas

| Trampa | Efecto |
|---|---|
| Bases de tiempo distintas entre `gt.tum` y `est.tum` | `evo` dice "no matching timestamps" o asocia mal |
| Cuaternión `(w,x,y,z)` escrito como `(x,y,z,w)` | ATE de traslación normal, ATE angular disparatado |
| Reportar Sim(3) en un VIO | Ocultas el error de escala; resultado no comparable |
| Comparar poses de cámara con GT de body | Sesgo constante de ~7 cm |
| `--t_max_diff` por defecto (0.01 s) con datos a 20 Hz | Descarta la mitad de las poses en silencio |
| Evaluar tramos distintos en cada método | Comparación sin sentido; fija el tramo |

---

## 7.9 Entregable

Un script `evaluate.py` que:

1. Toma un `.tum` y produce la tabla completa (ATE, RPE, deriva %, tiempo).
2. Genera las figuras (trayectorias superpuestas, error vs tiempo, mapa de color).
3. Compara todos los métodos del curso en una única tabla y una única figura.
