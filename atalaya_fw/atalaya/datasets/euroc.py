"""
atalaya.datasets.euroc — adaptador del formato ASL de EuRoC.

Idéntico en comportamiento a EurocSequence de anexo_utils, pero devolviendo el
contrato común `Secuencia`. El cuaternión del GT viene (w,x,y,z): se reordena
al leerlo (error clásico documentado en anexo_convenios A.3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..geometria import quat_wxyz_to_R
from ..sensores import CamaraPinhole, ParamsImu
from .base import Secuencia


def cargar_euroc(raiz, nombre_cam: str = "cam0") -> Secuencia:
    raiz = Path(raiz).expanduser()
    if (raiz / "mav0").exists():
        raiz = raiz / "mav0"

    imu = pd.read_csv(raiz / "imu0/data.csv")
    imu.columns = ["t_ns", "wx", "wy", "wz", "ax", "ay", "az"]
    cam = pd.read_csv(raiz / nombre_cam / "data.csv")
    cam.columns = ["t_ns", "filename"]

    ruta_gt = raiz / "state_groundtruth_estimate0/data.csv"
    gt = None
    if ruta_gt.exists():
        gt = pd.read_csv(ruta_gt)
        gt.columns = ["t_ns", "px", "py", "pz", "qw", "qx", "qy", "qz",
                      "vx", "vy", "vz", "bgx", "bgy", "bgz", "bax", "bay", "baz"]

    t0_ns = int(min([imu.t_ns.iloc[0], cam.t_ns.iloc[0]]
                    + ([gt.t_ns.iloc[0]] if gt is not None else [])))
    for df in filter(lambda d: d is not None, (imu, cam, gt)):
        df["t"] = (df.t_ns.astype(np.int64) - t0_ns).astype(np.float64) * 1e-9

    rutas = [str(raiz / nombre_cam / "data" / f.strip()) for f in cam.filename]

    with open(raiz / nombre_cam / "sensor.yaml") as f:
        c = yaml.safe_load(f)
    fu, fv, cu, cvv = c["intrinsics"]
    camara = CamaraPinhole(
        fu=fu, fv=fv, cu=cu, cv_=cvv,
        dist=np.array(c["distortion_coefficients"], dtype=float),
        width=int(c["resolution"][0]), height=int(c["resolution"][1]),
        T_body_cam=np.array(c["T_BS"]["data"], dtype=float).reshape(4, 4),
        modelo="radtan", rate_hz=float(c.get("rate_hz", 20)))

    with open(raiz / "imu0/sensor.yaml") as f:
        ci = yaml.safe_load(f)
    imu_params = ParamsImu(
        rate_hz=float(ci.get("rate_hz", 200)),
        sigma_g=float(ci["gyroscope_noise_density"]),
        sigma_a=float(ci["accelerometer_noise_density"]),
        sigma_bg=float(ci["gyroscope_random_walk"]),
        sigma_ba=float(ci["accelerometer_random_walk"]))

    kw = {}
    if gt is not None:
        kw = dict(t_gt=gt.t.values,
                  gt_p=gt[["px", "py", "pz"]].values.astype(float),
                  gt_R=quat_wxyz_to_R(gt[["qw", "qx", "qy", "qz"]].values),
                  gt_v=gt[["vx", "vy", "vz"]].values.astype(float))

    return Secuencia(
        nombre=raiz.parent.name, t0_ns=t0_ns,
        t_imu=imu.t.values, gyro=imu[["wx", "wy", "wz"]].values.astype(float),
        accel=imu[["ax", "ay", "az"]].values.astype(float),
        t_cam=cam.t.values, rutas=rutas,
        camara=camara, imu_params=imu_params, **kw)
