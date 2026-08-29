"""
atalaya.datasets.generico — CSVs arbitrarios con mapeo de columnas.

Para datos que no son ni EuRoC ni un log de ArduPilot: un CSV de IMU, una
carpeta de frames con data.csv (formato EuRoC, generable con
`atalaya extraer-frames`) y, opcionalmente, un CSV de GT.

Bloque YAML:
    dataset:
      tipo: generico
      imu_csv: imu.csv
      imu_columnas: {t: time_s, wx: gx, wy: gy, wz: gz, ax: ax, ay: ay, az: az}
      unidades: {t: s, gyro: rad_s, accel: m_s2}   # o t: ns/us/ms, gyro: deg_s, accel: g
      directorio_frames: frames/
      gt_csv: gt.csv                                # opcional
      gt_columnas: {t: t, px: x, py: y, pz: z, qw: qw, qx: qx, qy: qy, qz: qz}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..geometria import quat_wxyz_to_R
from ..sensores import ParamsImu
from .base import Secuencia

_FACTOR_T = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
_FACTOR_G = {"rad_s": 1.0, "deg_s": np.pi / 180.0}
_FACTOR_A = {"m_s2": 1.0, "g": 9.81}


def cargar_generico(cfg_dataset: dict, camara, imu_params: ParamsImu | None = None,
                    verbose=True) -> Secuencia:
    u = cfg_dataset.get("unidades", {}) or {}
    f_t = _FACTOR_T[str(u.get("t", "s"))]
    f_g = _FACTOR_G[str(u.get("gyro", "rad_s"))]
    f_a = _FACTOR_A[str(u.get("accel", "m_s2"))]

    col = cfg_dataset["imu_columnas"]
    df = pd.read_csv(Path(cfg_dataset["imu_csv"]).expanduser())
    t_imu = df[col["t"]].values.astype(np.float64) * f_t
    gyro = df[[col["wx"], col["wy"], col["wz"]]].values.astype(float) * f_g
    accel = df[[col["ax"], col["ay"], col["az"]]].values.astype(float) * f_a

    dir_frames = Path(cfg_dataset["directorio_frames"]).expanduser()
    fr = pd.read_csv(dir_frames / "data.csv")
    t_cam = fr.t_ns.values.astype(np.float64) * 1e-9 \
        + float((cfg_dataset.get("camara", {}) or {}).get("offset_temporal_s", 0.0))
    rutas = [str(dir_frames / "data" / f) for f in fr.filename]

    kw_gt = {}
    if cfg_dataset.get("gt_csv"):
        cg = cfg_dataset["gt_columnas"]
        g = pd.read_csv(Path(cfg_dataset["gt_csv"]).expanduser())
        t_gt = g[cg["t"]].values.astype(np.float64) * _FACTOR_T[str(u.get("t_gt", u.get("t", "s")))]
        q = g[[cg["qw"], cg["qx"], cg["qy"], cg["qz"]]].values.astype(float)
        kw_gt = dict(t_gt=t_gt,
                     gt_p=g[[cg["px"], cg["py"], cg["pz"]]].values.astype(float),
                     gt_R=quat_wxyz_to_R(q), gt_v=None)

    t0 = float(min([t_imu[0], t_cam[0]] + ([kw_gt["t_gt"][0]] if kw_gt else [])))
    t_imu = t_imu - t0
    t_cam = t_cam - t0
    if kw_gt:
        kw_gt["t_gt"] = kw_gt["t_gt"] - t0

    seq = Secuencia(nombre=Path(cfg_dataset["imu_csv"]).stem, t0_ns=int(round(t0 * 1e9)),
                    t_imu=t_imu, gyro=gyro, accel=accel, t_cam=t_cam, rutas=rutas,
                    camara=camara, imu_params=imu_params or ParamsImu(), **kw_gt)
    if verbose:
        print(seq)
    return seq
