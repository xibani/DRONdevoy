# Fase 2 — Calibración e intrínsecos

**Objetivo:** tener `K`, la distorsión, y `T_cam_imu` correctamente construidas y
**verificadas geométricamente**, no solo leídas del YAML.
**Criterio de éxito:** reproyectar el GT sobre las imágenes y ver que los puntos
caen donde deben (test de reproyección con dos frames y triangulación).

---

## 2.1 Lectura del YAML

`cam0/sensor.yaml` de EuRoC:

```yaml
sensor_type: camera
comment: VI-Sensor cam0 (MT9M034)
T_BS:
  cols: 4
  rows: 4
  data: [0.0148655429818, -0.999880929698,  0.00414029679422, -0.0216401454975,
         0.999557249008,   0.0149672133247, 0.025715529948,   -0.064676986768,
        -0.0257744366974,  0.00375618835797,0.999660727178,    0.00981073058949,
         0.0, 0.0, 0.0, 1.0]
rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375]      # fu, fv, cu, cv
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
```

```python
from anexo_utils import EurocSequence
seq = EurocSequence("~/datasets/euroc/MH_01_easy")
cam = seq.cam0

print("K =\n", cam.K)
print("dist (k1,k2,p1,p2) =", cam.dist)
print("T_body_cam =\n", cam.T_body_cam)
print("T_cam_imu  =\n", cam.T_cam_imu)
```

**Interpretación de cada número:**

- `intrinsics = [fu, fv, cu, cv]` → matriz `K = [[fu,0,cu],[0,fv,cv],[0,0,1]]`.
  `fu ≈ fv ≈ 458` píxeles para 752×480 ⇒ FOV horizontal ≈ 2·atan(376/458) ≈ **78°**.
- `distortion_model: radial-tangential` con 4 coeficientes ⇒ el modelo
  `plumb_bob` de OpenCV con `k3 = 0`. **Pásale a OpenCV `[k1, k2, p1, p2]`** (4
  elementos) o `[k1, k2, p1, p2, 0.0]`. No pases 5 valores con `k3` inventado.
- `T_BS` es **`T_body_from_cam`**: `p_body = T_BS @ p_cam`. Su traslación
  `(-0.0216, -0.0647, 0.0098)` es la posición física de cam0 en el frame IMU: ~2 cm
  a un lado, ~6.5 cm al otro, ~1 cm arriba. Coherente con el VI-Sensor real.
- Para `imu0/sensor.yaml`, `T_BS = I₄`: **el body frame ES el frame del IMU.**

Por tanto:

```python
T_cam_imu = np.linalg.inv(cam.T_body_cam)     # p_cam = T_cam_imu @ p_imu
```

---

## 2.2 Undistort

Dos formas, y usas **ambas** en momentos distintos:

### (a) Rectificar la imagen entera — para el front-end de features

```python
import cv2
map1, map2 = cam.undistort_maps()               # mantiene los MISMOS K
img  = seq.image(0)
img_u = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

Con `newCameraMatrix = K` conservas los intrínsecos, así que puedes seguir usando
`K` para todo lo demás. Si usaras `cv2.getOptimalNewCameraMatrix` tendrías un `K'`
distinto y tendrías que propagarlo. **No lo hagas**: complica sin ganar nada aquí.

Coste: un `remap` por frame (≈1 ms para 752×480). En Fase 4 esto es aceptable.

### (b) Deshacer la distorsión solo en los puntos — para el back-end

```python
pts_u = cam.undistort_points(pts)   # (N,2) píxeles ideales, mismo K
pts_n = cam.normalize(pts)          # (N,2) coords normalizadas x/z, y/z
```

Más eficiente si solo tienes cientos de features. Es lo que hace el MSCKF: trabaja
con **coordenadas normalizadas** y así el modelo de medida es simplemente
`z = [X/Z, Y/Z]`, sin `K` de por medio.

**Regla:** rectifica imágenes para el tracking KLT (los parches deben ser
consistentes), y normaliza puntos para la geometría.

### Comprobación visual

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].imshow(img, cmap="gray");   ax[0].set_title("original")
ax[1].imshow(img_u, cmap="gray"); ax[1].set_title("rectificada")
ax[2].imshow(cv2.absdiff(img, img_u), cmap="inferno")
ax[2].set_title("diferencia (max en los bordes)")
for a in ax: a.axis("off")
plt.show()
```

Con `k1 = -0.283` la distorsión es de barril apreciable: las líneas rectas de las
estructuras metálicas del machine hall deben quedar **más rectas** tras rectificar.
Si empeoran, has puesto los signos o el orden de los coeficientes mal.

Cuantificación del efecto:

```python
import numpy as np
corners = np.array([[0,0],[751,0],[0,479],[751,479],[376,240]], dtype=np.float32)
print(np.abs(cam.undistort_points(corners) - corners).max(axis=0), "px de desplazamiento")
```

Espera decenas de píxeles en las esquinas. Ignorar la distorsión **no** es una
opción en EuRoC.

---

## 2.3 Test geométrico de `T_cam_imu` (el importante)

Leer el YAML no demuestra que lo hayas interpretado bien. Este test sí.

**Idea:** toma dos frames con el GT. Detecta y empareja features. Triangula con las
poses del GT. Si `T_cam_imu` es correcta, los puntos 3D salen delante de las
cámaras (Z > 0) y el error de reproyección es de subpíxel. Si la has invertido,
sale basura.

```python
import numpy as np, cv2
from anexo_utils import inv_T

ka, kb = 300, 310                      # ~0.5 s de separación
Ia, Ib = seq.image(ka, undistort=True), seq.image(kb, undistort=True)
ta, tb = seq.t_cam[ka], seq.t_cam[kb]

# Poses GT del body y de la cámara
T_w_ia, T_w_ib = seq.gt_T_ws(ta), seq.gt_T_ws(tb)
T_i_c = cam.T_body_cam                       # T_BS del YAML
T_w_ca = T_w_ia @ T_i_c
T_w_cb = T_w_ib @ T_i_c

# Matrices de proyección P = K [R|t] con T_c_w = inv(T_w_c)
def P_from(T_w_c):
    T_c_w = inv_T(T_w_c)
    return cam.K @ T_c_w[:3, :]

Pa, Pb = P_from(T_w_ca), P_from(T_w_cb)

# Correspondencias por KLT
pa = cv2.goodFeaturesToTrack(Ia, 300, 0.01, 15)
pb, st, _ = cv2.calcOpticalFlowPyrLK(Ia, Ib, pa, None,
                                     winSize=(21,21), maxLevel=3)
m = st.ravel() == 1
pa_, pb_ = pa[m].reshape(-1,2), pb[m].reshape(-1,2)

X = cv2.triangulatePoints(Pa, Pb, pa_.T, pb_.T)
X = (X[:3] / X[3]).T                          # (N,3) puntos en el mundo

# Profundidad en la cámara A
Xc = (inv_T(T_w_ca)[:3,:3] @ X.T).T + inv_T(T_w_ca)[:3,3]
print(f"puntos con Z>0: {(Xc[:,2] > 0).mean()*100:.1f} %")
print(f"profundidad mediana: {np.median(Xc[:,2]):.2f} m")

# Error de reproyección en B
proj = (Pb @ np.hstack([X, np.ones((len(X),1))]).T)
proj = (proj[:2] / proj[2]).T
err = np.linalg.norm(proj - pb_, axis=1)
print(f"reproyección: mediana {np.median(err):.3f} px, p90 {np.percentile(err,90):.3f} px")
```

**Resultados esperados si todo está bien:**

| Métrica | Valor esperado |
|---|---|
| % puntos con Z > 0 | > 95 % |
| Profundidad mediana | 2–15 m (interior del machine hall) |
| Reproyección mediana | < 1 px |

**Diagnóstico si falla:**

| Síntoma | Causa |
|---|---|
| ~50 % con Z<0, profundidades ±grandes | `T_i_c` invertida: prueba `inv_T(cam.T_body_cam)` |
| Reproyección de cientos de px | Poses GT mal (cuaternión `(w,x,y,z)` vs `(x,y,z,w)`) |
| Reproyección de 3–10 px | No has rectificado las imágenes o usas puntos distorsionados con `K` |
| Todo NaN | `triangulatePoints` con `P` en float32 vs float64, o puntos duplicados |

Este test es tu **regresión permanente**: cada vez que toques transformadas en las
fases siguientes, vuelve a correrlo.

---

## 2.4 Visualizar la geometría del rig

```python
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
ax = plt.figure(figsize=(6,6)).add_subplot(projection="3d")
plot_transform(ax=ax, A2B=np.eye(4), s=0.05, name="IMU/body")
plot_transform(ax=ax, A2B=cam.T_body_cam, s=0.05, name="cam0")
if seq.cam1:
    plot_transform(ax=ax, A2B=seq.cam1.T_body_cam, s=0.05, name="cam1")
ax.set_xlim(-.1,.1); ax.set_ylim(-.1,.1); ax.set_zlim(-.1,.1)
plt.show()

if seq.cam1:
    baseline = np.linalg.norm(seq.cam1.T_body_cam[:3,3] - cam.T_body_cam[:3,3])
    print(f"baseline estéreo: {baseline*100:.1f} cm")   # ~11 cm en EuRoC
```

Verás que el eje z de la cámara (hacia adelante, azul en pytransform3d) apunta en
una dirección distinta a los ejes del IMU: la rotación entre ambos es ~90°, que es
lo esperable dado que el frame óptico es z-adelante/y-abajo y el del IMU no.

---

## 2.5 Nota sobre calibración propia (relevante para ATALAYA)

En EuRoC la calibración viene dada y es buena. En tu hardware no:

- **Intrínsecos**: `cv2.calibrateCamera` con tablero de ajedrez. Objetivo: RMS de
  reproyección < 0.3 px y ≥ 20 vistas cubriendo todo el campo, incluidas esquinas.
- **Extrínsecos cámara-IMU** (`T_cam_imu`) y **time offset**: usa **Kalibr**. Es la
  herramienta estándar y no hay atajo razonable. Necesitas un target Aprilgrid y
  excitar los 6 DoF.
- El **time offset** cámara-IMU en un setup casero (Pi + autopiloto) es de decenas
  de ms y **domina el error** si no lo estimas. Kalibr lo estima; algunos VIO
  (VINS-Mono, OpenVINS) lo estiman online. En EuRoC es cero porque hay sincronismo
  hardware — disfrútalo mientras dure.

---

## 2.6 Entregable

Notebook `02_calibracion.ipynb` con:

1. Impresión de `K`, `dist`, `T_body_cam`, `T_cam_imu`.
2. Comparación visual original / rectificada / diferencia.
3. El test de triangulación con GT, con las tres métricas numéricas.
4. Plot 3D de los frames del rig.
