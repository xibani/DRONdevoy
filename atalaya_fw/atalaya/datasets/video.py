"""
atalaya.datasets.video — compone los frames de un vídeo en formato EuRoC.

Extrae los frames a `directorio_frames/data/*.png` y escribe
`directorio_frames/data.csv` con columnas `t_ns, filename` (t en la base de
tiempo que elijas). A partir de ahí, cualquier adaptador puede usarlos igual
que una carpeta cam0 de EuRoC.

Fuentes de timestamp (bloque dataset.camara del YAML):
  timestamps: "fps"      -> t_k = t0 + k / fps            (t0 y fps del YAML)
  timestamps: "csv"      -> un CSV con una columna de tiempos en s (o ns si
                            los valores son enormes), una fila por frame
  timestamps: "cam_msg"  -> mensajes CAM/TRIG del log de ArduPilot; exige que
                            nº de triggers == nº de frames (o se recorta al
                            mínimo con aviso)

El offset temporal cámara-IMU (decenas de ms en un setup Pi + autopiloto) NO
se aplica aquí: se aplica al cargar la secuencia (dataset.camara.
offset_temporal_s) y se puede estimar con `atalaya offset`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def extraer_frames(ruta_video, dir_salida, escala_gris=True, cada_n=1,
                   max_frames=None, verbose=True):
    """Vuelca los frames del vídeo a PNG. Devuelve (rutas, indices)."""
    dir_salida = Path(dir_salida)
    (dir_salida / "data").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"no puedo abrir el vídeo: {ruta_video}")
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    rutas, indices, k = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % cada_n == 0:
            if escala_gris and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ruta = dir_salida / "data" / f"{k:08d}.png"
            cv2.imwrite(str(ruta), frame)
            rutas.append(str(ruta))
            indices.append(k)
            if max_frames and len(rutas) >= max_frames:
                break
        k += 1
    cap.release()
    if verbose:
        print(f"{len(rutas)} frames extraídos a {dir_salida}/data "
              f"(vídeo a {fps_video:.2f} fps, 1 de cada {cada_n})")
    return rutas, np.array(indices), fps_video


def _leer_csv_tiempos(ruta) -> np.ndarray:
    t = np.loadtxt(ruta, delimiter=",", ndmin=1)
    if t.ndim > 1:
        t = t[:, 0]
    if np.median(np.abs(t)) > 1e12:          # eran nanosegundos
        t = t.astype(np.float64) * 1e-9
    return t


def componer_frames(cfg_dataset: dict, verbose=True):
    """Extrae frames y les asigna timestamps. Devuelve (rutas, t_s).
    Escribe data.csv en el directorio de frames para poder recargar sin
    re-extraer (si data.csv ya existe, se reutiliza)."""
    c = cfg_dataset.get("camara", {}) or {}
    dir_frames = Path(c.get("directorio_frames", "frames")).expanduser()
    csv_salida = dir_frames / "data.csv"

    if csv_salida.exists():
        import pandas as pd
        df = pd.read_csv(csv_salida)
        rutas = [str(dir_frames / "data" / f) for f in df.filename]
        if verbose:
            print(f"reutilizando {len(rutas)} frames ya extraídos ({csv_salida})")
        return rutas, df.t_ns.values.astype(np.float64) * 1e-9

    modo = str(c.get("timestamps", "fps")).lower()
    rutas, indices, fps_video = extraer_frames(
        c["video"], dir_frames,
        cada_n=int(c.get("cada_n", 1)),
        max_frames=c.get("max_frames"), verbose=verbose)

    if modo == "fps":
        fps = float(c.get("fps") or fps_video)
        t0 = float(c.get("t0", 0.0))
        t = t0 + indices / fps
    elif modo == "csv":
        t_todos = _leer_csv_tiempos(Path(c["csv"]).expanduser())
        if len(t_todos) < indices.max() + 1:
            raise ValueError(f"el CSV trae {len(t_todos)} tiempos y el vídeo "
                             f"tiene al menos {indices.max()+1} frames")
        t = t_todos[indices]
    elif modo == "cam_msg":
        from .ardupilot import tiempos_trigger
        t_trig = tiempos_trigger(Path(cfg_dataset["log"]).expanduser())
        n = min(len(t_trig), len(rutas))
        if n == 0:
            raise ValueError("el log no trae mensajes CAM/TRIG")
        if len(t_trig) != len(rutas):
            print(f"  AVISO: {len(t_trig)} triggers vs {len(rutas)} frames; "
                  f"recorto a {n}. Verifica que no se pierden frames por el camino.")
        rutas, t = rutas[:n], t_trig[:n]
    else:
        raise ValueError(f"timestamps='{modo}' no reconocido (fps|csv|cam_msg)")

    import pandas as pd
    pd.DataFrame({"t_ns": np.round(np.asarray(t) * 1e9).astype(np.int64),
                  "filename": [Path(r).name for r in rutas]}).to_csv(
        csv_salida, index=False)
    if verbose:
        dt = np.diff(t)
        print(f"timestamps '{modo}': dt mediana {np.median(dt)*1e3:.1f} ms, "
              f"std {np.std(dt)*1e3:.2f} ms -> escrito {csv_salida}")
    return rutas, np.asarray(t, dtype=np.float64)
