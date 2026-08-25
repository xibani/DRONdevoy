# Fase 0 — Preparación

**Objetivo:** entorno reproducible + `MH_01_easy` descargado y verificado.
**Criterio de éxito:** un notebook que abre una imagen del dataset, la muestra, e
imprime las 5 primeras filas del CSV del IMU con los `dt` correctos (~0.005 s).

---

## 0.1 Entorno

Usa un entorno aislado. `gtsam` y `evo` tienen dependencias que no quieres mezclar
con tus proyectos de PyTorch.

```bash
python -m venv .venv-vio          # o conda create -n vio python=3.11
source .venv-vio/bin/activate

pip install numpy scipy pandas matplotlib pyyaml tqdm
pip install opencv-python          # NO opencv-contrib salvo que quieras SIFT/SURF
pip install evo --upgrade
pip install gtsam
pip install pytransform3d
pip install jupyterlab
```

Notas:

- **`scipy` es imprescindible** aunque no estuviera en tu lista: `scipy.spatial.transform.Rotation`
  te ahorra escribir (y depurar) conversiones cuaternión↔matriz.
- **`gtsam`**: las ruedas de PyPI cubren CPython 3.8–3.12 en Linux x86_64 y macOS.
  Si estás en aarch64 (como tu DGX Spark) puede que tengas que compilar desde
  fuente con `-DGTSAM_BUILD_PYTHON=ON`. Si eso te bloquea, **haz primero el Camino A
  (MSCKF) de la Fase 6**, que solo necesita numpy.
- **`evo`**: si te da problemas con matplotlib, instala con
  `pip install evo --upgrade --no-binary evo`.
- **`pytransform3d`** lo usarás sobre todo para dibujar frames en 3D y sanidad
  visual de las transformadas. No es crítico.

Verificación:

```python
import numpy, scipy, pandas, cv2, yaml, matplotlib
print(cv2.__version__)          # >= 4.5
import gtsam; print(gtsam.__version__)
```

```bash
evo_traj --help
```

---

## 0.2 Descarga del dataset

**Descarga solo una secuencia.** El dataset completo son cientos de GB y no lo
necesitas.

```bash
mkdir -p ~/datasets/euroc && cd ~/datasets/euroc

wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip

unzip MH_01_easy.zip -d MH_01_easy
```

Patrón general de URL:

```
http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/<grupo>/<secuencia>/<secuencia>.zip
grupos: machine_hall | vicon_room1 | vicon_room2
```

Si el mirror de ETH está caído, los archivos están también en la ETH Research
Collection (DOI `10.3929/ethz-b-000690084`) y la página oficial es
`https://projects.asl.ethz.ch/datasets/euroc-mav/`.

**Descarga el formato ASL (`.zip`), no el ROS bag.** El ASL son CSV + PNG y no
requiere ROS para nada.

Segunda secuencia recomendada para más adelante: `V1_01_easy` (Vicon room 1,
GT a 100 Hz, más textura cercana).

---

## 0.3 Estructura de carpetas que debes obtener

```
MH_01_easy/
└── mav0/
    ├── cam0/
    │   ├── data/                 # 000000000.png ... (nombre = timestamp en ns)
    │   ├── data.csv              # timestamp, filename
    │   └── sensor.yaml           # intrínsecos + T_BS
    ├── cam1/                     # idéntico, cámara derecha
    ├── imu0/
    │   ├── data.csv              # timestamp, wx,wy,wz, ax,ay,az
    │   └── sensor.yaml           # T_BS = I, ruidos
    ├── leica0/                   # posición 3D del prisma (solo machine_hall)
    │   ├── data.csv
    │   └── sensor.yaml
    ├── state_groundtruth_estimate0/
    │   ├── data.csv              # 17 columnas, el GT que vas a usar
    │   └── sensor.yaml
    └── body.yaml
```

Ojo: en algunas descargas el zip descomprime directamente `mav0/` sin la carpeta
`MH_01_easy/` encima. Ajusta tus rutas.

---

## 0.4 Script de verificación

Guárdalo como `verify_dataset.py` y ejecútalo antes de escribir nada más.

```python
#!/usr/bin/env python3
"""Verificación de integridad y cordura de una secuencia EuRoC en formato ASL."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
import cv2

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/euroc/MH_01_easy/mav0").expanduser()

def check(name, cond, extra=""):
    print(f"[{'OK ' if cond else 'FAIL'}] {name} {extra}")
    return cond

ok = True

# --- 1. Estructura -----------------------------------------------------------
for sub in ["cam0/data.csv", "cam0/sensor.yaml", "imu0/data.csv",
            "imu0/sensor.yaml", "state_groundtruth_estimate0/data.csv"]:
    ok &= check(f"existe {sub}", (ROOT / sub).exists())

# --- 2. IMU ------------------------------------------------------------------
imu = pd.read_csv(ROOT / "imu0/data.csv")
imu.columns = ["t_ns", "wx", "wy", "wz", "ax", "ay", "az"]
t_imu = imu.t_ns.values.astype(np.int64)
dt = np.diff(t_imu) * 1e-9
ok &= check("IMU ~200 Hz", abs(np.median(dt) - 0.005) < 1e-4,
            f"(dt mediano = {np.median(dt)*1000:.3f} ms)")
ok &= check("IMU sin huecos grandes", dt.max() < 0.05,
            f"(dt max = {dt.max()*1000:.1f} ms)")

# --- 3. Cámara ---------------------------------------------------------------
cam = pd.read_csv(ROOT / "cam0/data.csv")
cam.columns = ["t_ns", "filename"]
t_cam = cam.t_ns.values.astype(np.int64)
dtc = np.diff(t_cam) * 1e-9
ok &= check("cam0 ~20 Hz", abs(np.median(dtc) - 0.05) < 1e-3,
            f"(dt mediano = {np.median(dtc)*1000:.2f} ms)")
n_files = len(list((ROOT / "cam0/data").glob("*.png")))
ok &= check("nº PNG == filas del CSV", n_files == len(cam),
            f"({n_files} vs {len(cam)})")

img = cv2.imread(str(ROOT / "cam0/data" / cam.filename.iloc[0]), cv2.IMREAD_GRAYSCALE)
ok &= check("imagen 752x480 mono", img is not None and img.shape == (480, 752),
            f"({None if img is None else img.shape})")

# --- 4. Ground truth ---------------------------------------------------------
gt = pd.read_csv(ROOT / "state_groundtruth_estimate0/data.csv")
ok &= check("GT tiene 17 columnas", gt.shape[1] == 17, f"({gt.shape[1]})")
q = gt.iloc[:, 4:8].values
ok &= check("cuaterniones GT normalizados",
            np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-3))

# --- 5. Solape temporal ------------------------------------------------------
t_gt = gt.iloc[:, 0].values.astype(np.int64)
lo = max(t_imu[0], t_cam[0], t_gt[0]); hi = min(t_imu[-1], t_cam[-1], t_gt[-1])
ok &= check("solape temporal IMU/cam/GT", hi > lo,
            f"({(hi-lo)*1e-9:.1f} s comunes)")

# --- 6. Cordura física: gravedad --------------------------------------------
static = imu.iloc[:200]           # primer segundo: el dron está quieto
acc_norm = np.linalg.norm(static[["ax","ay","az"]].values, axis=1).mean()
ok &= check("||a|| ~ 9.81 en reposo", abs(acc_norm - 9.81) < 0.5,
            f"({acc_norm:.3f} m/s^2)")
gyro_bias0 = static[["wx","wy","wz"]].values.mean(axis=0)
print(f"       bias gyro estimado en reposo: {gyro_bias0} rad/s")
print(f"       bias gyro segun GT (t0):      {gt.iloc[0, 11:14].values} rad/s")

# --- 7. Calibración ----------------------------------------------------------
with open(ROOT / "cam0/sensor.yaml") as f:
    c0 = yaml.safe_load(f)
T_BS = np.array(c0["T_BS"]["data"]).reshape(4, 4)
print("       T_BS(cam0) traslacion:", T_BS[:3, 3])
ok &= check("T_BS con traslacion de pocos cm",
            np.linalg.norm(T_BS[:3, 3]) < 0.2)
ok &= check("T_BS bloque rotacion ortonormal",
            np.allclose(T_BS[:3, :3] @ T_BS[:3, :3].T, np.eye(3), atol=1e-6))
print("       intrinsics:", c0["intrinsics"], " dist:", c0["distortion_coefficients"])

print("\n==>", "TODO CORRECTO" if ok else "HAY FALLOS, revísalos antes de seguir")
```

Ejecución:

```bash
python verify_dataset.py ~/datasets/euroc/MH_01_easy/mav0
```

---

## 0.5 Qué has aprendido aquí (y por qué importa)

El check 6 (gravedad) es el más importante y el que la gente se salta. Te da tres
cosas gratis:

1. Confirma que las columnas del CSV están en el orden que crees (**gyro antes que
   accel** — al revés que en muchos otros datasets, incluido el formato de logs de
   ArduPilot con el que trabajas en ATALAYA).
2. Te da una estimación inicial del bias del giro (media en reposo) que usarás en
   Fase 3, y te permite compararla con la del GT.
3. Te confirma que las unidades son SI (rad/s, m/s²) y no g's o deg/s.

---

## 0.6 Trampas

- **`opencv-python` vs `opencv-python-headless`**: si vas a trabajar por SSH en una
  máquina sin display (como tu Pi), instala la headless y usa matplotlib para ver
  imágenes. `cv2.imshow` te colgará el kernel.
- **Rutas con `~`**: `pd.read_csv` no expande `~` de forma fiable en todas las
  versiones. Usa `Path(...).expanduser()`.
- **El CSV del GT tiene espacios en los nombres de columna** (`" p_RS_R_x [m]"`).
  No confíes en los nombres: renombra por posición como en el script.
- **No versiones el dataset en git.** Añade `datasets/` a `.gitignore` desde el
  minuto cero.
