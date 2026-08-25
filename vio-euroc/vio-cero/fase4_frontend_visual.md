# Fase 4 — Front-end visual (KLT + pose)

**Objetivo:** una odometría visual monocular funcionando: detectar, trackear,
estimar movimiento relativo, componer trayectoria.
**Criterio de éxito:** trayectoria VO cuya forma se parece a la GT tras
alineamiento **Sim(3)**, con ATE-RMSE (escala libre) < 2 m en los primeros 60 s de
MH_01. La escala será arbitraria: es lo esperado.

---

## 4.1 Anatomía del front-end

```
imagen k-1 ──► features trackeadas (KLT) ──► outliers fuera (RANSAC + esencial)
                     │                                    │
                     ▼                                    ▼
            ¿hay < N features?                    R, t (escala unitaria)
                     │                                    │
                     ▼                                    ▼
            redetectar con máscara            componer pose acumulada
```

Cuatro decisiones y sus valores razonables para EuRoC:

| Decisión | Valor | Por qué |
|---|---|---|
| Nº de features | 200–300 | Más no mejora en 752×480; cuesta tiempo |
| Separación mínima | 15–20 px | Evita features amontonadas en una esquina |
| Ventana KLT | 21×21, 3 niveles | 752×480 con movimiento moderado |
| Redetección | cuando N < 150 | Mantener el track largo importa más que la densidad |

---

## 4.2 Tracker KLT con IDs persistentes

La diferencia entre "sé hacer KLT" y "tengo un front-end de VIO" son los **IDs
persistentes**: el MSCKF de la Fase 6 necesita saber que la feature #4237 se ha
visto en los frames 100..118. Escríbelo ya con IDs.

```python
import numpy as np, cv2
from collections import defaultdict

class KLTTracker:
    def __init__(self, max_feats=250, min_feats=150, quality=0.01,
                 min_dist=18, win=(21, 21), levels=3, fb_thresh=1.0):
        self.max_feats, self.min_feats = max_feats, min_feats
        self.quality, self.min_dist = quality, min_dist
        self.lk = dict(winSize=win, maxLevel=levels,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.fb_thresh = fb_thresh
        self.prev_img = None
        self.pts = np.zeros((0, 2), np.float32)   # posiciones actuales
        self.ids = np.zeros((0,), np.int64)
        self._next_id = 0
        self.tracks = defaultdict(list)           # id -> [(frame_k, u, v), ...]

    # -- detección con máscara para no repetir features existentes ----------
    def _detect(self, img):
        n_new = self.max_feats - len(self.pts)
        if n_new <= 0:
            return
        mask = np.full(img.shape, 255, np.uint8)
        for p in self.pts.astype(int):
            cv2.circle(mask, tuple(p), self.min_dist, 0, -1)
        new = cv2.goodFeaturesToTrack(img, n_new, self.quality, self.min_dist,
                                      mask=mask, blockSize=7)
        if new is None:
            return
        new = new.reshape(-1, 2).astype(np.float32)
        ids = np.arange(self._next_id, self._next_id + len(new))
        self._next_id += len(new)
        self.pts = np.vstack([self.pts, new])
        self.ids = np.concatenate([self.ids, ids])

    # -- un frame ------------------------------------------------------------
    def track(self, img, k):
        """Devuelve (ids, pts_prev, pts_cur) de las features supervivientes."""
        if self.prev_img is None:
            self.prev_img = img
            self._detect(img)
            for i, p in zip(self.ids, self.pts):
                self.tracks[int(i)].append((k, *p))
            return self.ids.copy(), None, self.pts.copy()

        prev_pts = self.pts.reshape(-1, 1, 2)
        cur, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_img, img, prev_pts, None, **self.lk)
        # forward-backward check: el mejor filtro de outliers del KLT
        back, st2, _ = cv2.calcOpticalFlowPyrLK(img, self.prev_img, cur, None, **self.lk)
        fb = np.linalg.norm(prev_pts - back, axis=2).ravel()

        h, w = img.shape
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_thresh)
        c = cur.reshape(-1, 2)
        good &= (c[:, 0] > 3) & (c[:, 0] < w - 4) & (c[:, 1] > 3) & (c[:, 1] < h - 4)

        ids_ok = self.ids[good]
        prev_ok = self.pts[good]
        cur_ok = c[good]

        self.pts, self.ids, self.prev_img = cur_ok, ids_ok, img
        if len(self.pts) < self.min_feats:
            self._detect(img)
        for i, p in zip(self.ids, self.pts):
            self.tracks[int(i)].append((k, *p))
        return ids_ok, prev_ok, cur_ok
```

Puntos que importan y que la mayoría de tutoriales omiten:

- **Forward-backward check**: trackear de vuelta y descartar si no vuelves al punto
  de partida. Elimina la mayoría de los tracks que "se enganchan" a un borde
  distinto. Umbral 1 px.
- **Máscara al redetectar**: sin ella `goodFeaturesToTrack` te devuelve las mismas
  esquinas que ya estás trackeando y acabas con features duplicadas.
- **Filtro de bordes**: KLT devuelve `status=1` para puntos que se salen medio
  parche fuera de la imagen. Recórtalos.

Diagnóstico del tracker (hazlo antes de estimar poses):

```python
from anexo_utils import EurocSequence
seq = EurocSequence("~/datasets/euroc/MH_01_easy")
tr = KLTTracker()
n_feats, flows = [], []
for k in range(0, 400):
    img = seq.image(k, undistort=True)
    ids, prev, cur = tr.track(img, k)
    n_feats.append(len(ids))
    if prev is not None and len(prev):
        flows.append(np.median(np.linalg.norm(cur - prev, axis=1)))

lens = np.array([len(v) for v in tr.tracks.values()])
print(f"features/frame: {np.mean(n_feats):.0f}")
print(f"flujo mediano : {np.median(flows):.2f} px/frame")
print(f"longitud de track: mediana {np.median(lens):.0f}, p90 {np.percentile(lens,90):.0f}")
```

**Valores sanos en MH_01**: 200–250 features/frame, flujo mediano 3–8 px/frame,
longitud de track mediana 8–20 frames. Si el flujo mediano supera 25 px/frame, sube
`maxLevel`. Si la longitud mediana es 2–3, tu forward-backward está mal o las
imágenes no están rectificadas.

---

## 4.3 Estimación de la pose relativa

Con puntos en **coordenadas normalizadas** (sin `K`), la matriz esencial se estima
directamente y el umbral de RANSAC se expresa en píxeles dividido por la focal.

```python
def relative_pose(cam, p0, p1, prob=0.999, px_thresh=1.0):
    """p0, p1: (N,2) píxeles en imagen RECTIFICADA. Devuelve (T_c1_c0, inliers)."""
    if len(p0) < 8:
        return None, None
    n0 = cv2.undistortPoints(p0.reshape(-1,1,2), cam.K, None).reshape(-1,2)
    n1 = cv2.undistortPoints(p1.reshape(-1,1,2), cam.K, None).reshape(-1,2)
    thr = px_thresh / ((cam.fu + cam.fv) / 2.0)

    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.USAC_MAGSAC,
                                   prob=prob, threshold=thr)
    if E is None or E.shape != (3, 3):
        return None, None
    n_in, R, t, mask_pose = cv2.recoverPose(E, n0, n1, np.eye(3), mask=mask)
    if n_in < 10:
        return None, None
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t.ravel()   # p_c1 = T @ p_c0
    return T, (mask_pose.ravel() > 0)
```

**Detalles que importan:**

- Como ya has rectificado la imagen, al normalizar pasas `distCoeffs=None`. Si
  usaras la imagen original tendrías que pasar `cam.dist` aquí y **no** rectificar.
  Haz una cosa u otra, nunca las dos (error clásico: distorsión aplicada dos veces).
- `cv2.recoverPose` devuelve `R, t` tal que **`p_c1 = R·p_c0 + t`**, es decir
  `T_c1_c0`. Para componer la trayectoria necesitas la inversa.
- `t` tiene **norma 1**. La escala del monocular es libre. Este es el punto entero
  de la Fase 5.
- `USAC_MAGSAC` es notablemente mejor que `RANSAC` para esto en OpenCV ≥ 4.5. Si tu
  versión no lo tiene, usa `cv2.RANSAC`.

### Degeneración por rotación pura

La matriz esencial es **degenerada cuando la traslación es cero**. En los primeros
segundos de MH_01 el dron está quieto: `t` sale como ruido puro y la trayectoria
"vibra". Hay que detectarlo:

```python
def is_degenerate(p0, p1, min_parallax_px=3.0, min_ratio=0.5):
    """Si el desplazamiento mediano es minúsculo, no hay traslación observable."""
    d = np.linalg.norm(p1 - p0, axis=1)
    return np.median(d) < min_parallax_px
```

Si es degenerado: **no actualices la posición**, solo la rotación (estimada con una
homografía o directamente con el giro de la IMU). En la Fase 5, esto lo resuelve la
IMU de forma natural.

---

## 4.4 Pipeline de VO completo

```python
from anexo_utils import inv_T, make_T

def run_vo(seq, k0=0, k1=600, min_parallax=3.0):
    cam = seq.cam0
    tr = KLTTracker()
    T_wc = np.eye(4)               # pose de la cámara en el mundo (arbitraria)
    poses, times = [T_wc.copy()], [seq.t_cam[k0]]

    prev_ids, prev_pts = None, None
    for k in range(k0, k1):
        img = seq.image(k, undistort=True)
        ids, p_prev, p_cur = tr.track(img, k)
        if p_prev is None or len(p_prev) < 20:
            poses.append(T_wc.copy()); times.append(seq.t_cam[k]); continue

        if is_degenerate(p_prev, p_cur, min_parallax):
            poses.append(T_wc.copy()); times.append(seq.t_cam[k]); continue

        T_c1c0, inl = relative_pose(cam, p_prev, p_cur)
        if T_c1c0 is None:
            poses.append(T_wc.copy()); times.append(seq.t_cam[k]); continue

        T_wc = T_wc @ inv_T(T_c1c0)          # T_w_c1 = T_w_c0 @ T_c0_c1
        poses.append(T_wc.copy()); times.append(seq.t_cam[k])
    return np.array(times), np.array(poses), tr

times, poses_c, tracker = run_vo(seq, 0, 600)
```

### De pose de cámara a pose de body

Para comparar con el GT (que es del body/IMU):

```python
T_i_c = seq.cam0.T_body_cam
poses_b = np.array([P @ inv_T(T_i_c) for P in poses_c])   # T_w_body = T_w_cam @ T_cam_body
```

Cuidado: `poses_c[0] = I`, así que tu trayectoria está en un frame arbitrario
alineado con la primera cámara, no en el frame del mundo del GT. Eso lo arregla el
alineamiento Umeyama en la evaluación.

---

## 4.5 Evaluación rápida (escala libre)

```python
from anexo_utils import ate_rmse, align_umeyama
p_est = poses_b[:, :3, 3]
p_gt  = np.array([seq.gt_at(t)[0] for t in times])

print("ATE-RMSE Sim(3) (escala libre):", ate_rmse(p_gt, p_est, with_scale=True), "m")
print("ATE-RMSE SE(3)  (escala 1):    ", ate_rmse(p_gt, p_est, with_scale=False), "m")
s, R, t = align_umeyama(p_gt, p_est, True)
print("escala recuperada:", s)

import matplotlib.pyplot as plt
p_al = (s * (R @ p_est.T)).T + t
fig = plt.figure(figsize=(7,6)); ax = fig.add_subplot(projection="3d")
ax.plot(*p_gt.T, label="GT"); ax.plot(*p_al.T, label="VO alineada Sim(3)")
ax.legend(); plt.show()
```

**Lo que debes ver:** la forma general de la trayectoria reconocible, con deriva
acumulada que crece; ATE Sim(3) del orden de 0.5–2 m en 60 s. El ATE SE(3) será
absurdo porque la escala no es 1.

---

## 4.6 Por qué esta VO deriva (y cómo lo arregla la fusión)

Tres fuentes de deriva, en orden de importancia:

1. **Deriva de escala.** Cada par de frames tiene su propia escala (‖t‖=1). Al
   componer, las escalas se multiplican de forma inconsistente y la trayectoria se
   "encoge" o "estira". Es el error dominante y es **irreparable en monocular puro**.
   → La IMU da escala métrica. Fase 5.

2. **Rotación pura / poca paralaje.** Cuando el dron gira sin trasladarse, la
   esencial es degenerada y metes ruido. → La IMU propaga durante esos tramos.

3. **Composición frame a frame (VO incremental).** Cada estimación tiene error, y al
   componer se acumula sin ningún mecanismo de corrección. → Un back-end (ventana
   deslizante o factor graph) reobserva features antiguas y reduce esta deriva.
   Fase 6.

Una mejora barata dentro de esta misma fase: en vez de frame-a-frame, estimar la
pose **contra un keyframe** que solo cambias cuando la paralaje supera un umbral.
Reduce la deriva bastante:

```python
def need_keyframe(p_kf, p_cur, n_tracked, parallax_thresh=25.0, min_tracked=100):
    return (np.median(np.linalg.norm(p_cur - p_kf, axis=1)) > parallax_thresh
            or n_tracked < min_tracked)
```

---

## 4.7 Alternativa: PnP con triangulación (para tu intuición)

La esencial descarta toda la estructura entre pares. La alternativa clásica:

1. Triangula puntos 3D entre los dos primeros keyframes.
2. Para cada frame nuevo, usa `cv2.solvePnPRansac` con las correspondencias 2D-3D.
3. Triangula puntos nuevos y extiende el mapa.

Ventaja: la escala es coherente en toda la secuencia (fijada por la primera
triangulación) y la deriva es menor. Es lo que hace ORB-SLAM. Merece la pena que
lo implementes como variante — el código de triangulación te sirve tal cual en la
Fase 6A para el MSCKF.

```python
ok, rvec, tvec, inliers = cv2.solvePnPRansac(
    pts3d, pts2d, cam.K, None, flags=cv2.SOLVEPNP_ITERATIVE,
    reprojectionError=2.0, confidence=0.999, iterationsCount=200)
R_cw, _ = cv2.Rodrigues(rvec)
T_c_w = make_T(R_cw, tvec.ravel())
T_w_c = inv_T(T_c_w)
```

---

## 4.8 Trampas

| Trampa | Efecto |
|---|---|
| Rectificar la imagen **y** pasar `dist` a `undistortPoints` | Distorsión aplicada dos veces; reproyecciones de 5–10 px |
| Confundir `T_c1_c0` con `T_c0_c1` al componer | La trayectoria sale invertida / hacia atrás |
| No detectar la rotación pura | La trayectoria "explota" en los primeros segundos estáticos |
| Umbral de RANSAC en píxeles con puntos normalizados | Todo inlier o todo outlier |
| Redetectar sin máscara | Features duplicadas, matriz esencial mal condicionada |
| No hacer forward-backward check | 10–20 % de tracks corruptos, deriva doble |

---

## 4.9 Entregable

Notebook `04_vo.ipynb` con:

1. `KLTTracker` con IDs y su diagnóstico (features/frame, flujo, longitud de track).
2. Visualización de tracks superpuestos sobre la imagen (dibuja las estelas).
3. `relative_pose` con detección de degeneración.
4. Trayectoria VO vs GT alineada con Sim(3), y la escala recuperada.
5. Comparación frame-a-frame vs keyframe.
