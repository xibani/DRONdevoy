"""
atalaya.datasets.ardupilot — logs DataFlash de ArduPilot (.BIN / .log).

Requiere `pymavlink` (pip install pymavlink). Import perezoso: el resto del
framework funciona sin él.

Frames y convenios
------------------
* Body de ArduPilot: FRD (x adelante, y derecha, z abajo). El IMU mide en FRD.
  El framework NO lo cambia: el body de la Secuencia es el FRD del autopiloto.
* Mundo del framework: ENU z-ARRIBA (g = (0,0,-9.81)), para mantener el
  convenio del curso. ArduPilot razona en NED; la conversión es
      R_ws(ENU<-FRD) = R_enu_ned @ R_ned_frd,   R_enu_ned = [[0,1,0],[1,0,0],[0,0,-1]]
* Posición: lat/lon/alt -> ENU local respecto del primer fix (aprox.
  equirectangular; válida para vuelos de cientos de metros).
* Tiempos: TimeUS (µs desde el arranque del autopiloto). Se resta el primer
  TimeUS usado, como en todo el curso.

Ground truth
------------
El "GT" de un log es la salida del EKF de ArduPilot (ATT + POS): es un
estimado, no una verdad de mocap. El ATE contra él mide ACUERDO con el EKF3,
no error absoluto. Útil como referencia, peligroso como métrica de récords.

Mensajes soportados (elegibles por config):
  imu.mensaje = "IMU"       -> IMU.{GyrX..AccZ}, filtrado por instancia (campo I)
  imu.mensaje = "GYR+ACC"   -> GYR.{GyrX..} + ACC.{AccX..} (raw a alta tasa),
                               emparejados por interpolación de ACC sobre GYR
  gt.fuente   = "att_pos"   -> ATT (Roll,Pitch,Yaw en grados, NED) + POS (Lat,Lng,Alt)
  gt.fuente   = "ninguna"
Los triggers de cámara (mensajes CAM o TRIG) se pueden usar para timestamping
del vídeo: ver atalaya.datasets.video.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometria import G_MAG
from ..sensores import ParamsImu
from .base import Secuencia

R_ENU_NED = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, -1.0]])
R_TIERRA = 6378137.0


def _dfreader(ruta):
    try:
        from pymavlink import DFReader
    except ImportError as e:
        raise ImportError(
            "El adaptador ArduPilot necesita pymavlink: pip install pymavlink") from e
    ruta = str(ruta)
    if ruta.lower().endswith(".bin"):
        return DFReader.DFReader_binary(ruta)
    return DFReader.DFReader_text(ruta)


def volcar_mensajes(ruta_log, tipos) -> dict:
    """Lee el log UNA vez y devuelve {tipo: lista de dicts de campos}."""
    tipos = set(tipos)
    log = _dfreader(ruta_log)
    out = {t: [] for t in tipos}
    while True:
        m = log.recv_match(type=list(tipos))
        if m is None:
            break
        out[m.get_type()].append(m.to_dict())
    return out


def inventario(ruta_log) -> dict:
    """Cuenta mensajes por tipo y estima tasas. Para `atalaya inspeccionar`."""
    log = _dfreader(ruta_log)
    cuentas, t_min, t_max = {}, {}, {}
    while True:
        m = log.recv_match()
        if m is None:
            break
        tt = m.get_type()
        cuentas[tt] = cuentas.get(tt, 0) + 1
        d = m.to_dict()
        if "TimeUS" in d:
            t = d["TimeUS"] * 1e-6
            t_min[tt] = min(t_min.get(tt, t), t)
            t_max[tt] = max(t_max.get(tt, t), t)
    filas = []
    for tt in sorted(cuentas):
        dur = t_max.get(tt, 0) - t_min.get(tt, 0)
        hz = cuentas[tt] / dur if dur > 0 else 0.0
        filas.append((tt, cuentas[tt], hz))
    return dict(filas=filas)


def _campo(d, *nombres, defecto=None):
    for n in nombres:
        if n in d:
            return d[n]
    if defecto is not None:
        return defecto
    raise KeyError(f"ninguno de {nombres} está en el mensaje: {sorted(d)}")


def _lat_lng_grados(v):
    """POS/GPS pueden venir en grados float o en 1e7·grados int según versión."""
    v = float(v)
    return v * 1e-7 if abs(v) > 1000.0 else v


def cargar_ardupilot(cfg_dataset: dict, camara, imu_params: ParamsImu | None = None,
                     rutas_frames=None, t_frames=None, verbose=True) -> Secuencia:
    """Construye una Secuencia desde un log DataFlash.

    cfg_dataset: bloque `dataset:` del YAML (tipo ardupilot).
    camara: CamaraPinhole ya construida desde el bloque `camara:`.
    rutas_frames / t_frames: si ya extrajiste los frames del vídeo
      (atalaya extraer-frames) se pasan aquí; si no, la secuencia sale sin
      cámara utilizable (solo IMU/GT: sirve para dead reckoning e inspección).
    """
    ruta = Path(cfg_dataset["log"]).expanduser()
    c_imu = cfg_dataset.get("imu", {}) or {}
    mensaje = str(c_imu.get("mensaje", "IMU")).upper()
    instancia = int(c_imu.get("instancia", 0))
    fuente_gt = str((cfg_dataset.get("gt", {}) or {}).get("fuente", "att_pos")).lower()

    tipos = {"ATT", "POS"} if fuente_gt == "att_pos" else set()
    tipos |= {"GYR", "ACC"} if mensaje == "GYR+ACC" else {"IMU"}
    dump = volcar_mensajes(ruta, tipos)

    # ---------------- IMU --------------------------------------------------
    if mensaje == "GYR+ACC":
        gyr = [d for d in dump.get("GYR", []) if int(d.get("I", 0)) == instancia]
        acc = [d for d in dump.get("ACC", []) if int(d.get("I", 0)) == instancia]
        if not gyr or not acc:
            raise ValueError("no hay mensajes GYR/ACC de esa instancia en el log "
                             "(¿está activado el logging raw, LOG_BITMASK?)")
        t_g = np.array([_campo(d, "SampleUS", "TimeUS") for d in gyr], float) * 1e-6
        w = np.array([[d["GyrX"], d["GyrY"], d["GyrZ"]] for d in gyr], float)
        t_a = np.array([_campo(d, "SampleUS", "TimeUS") for d in acc], float) * 1e-6
        a = np.array([[d["AccX"], d["AccY"], d["AccZ"]] for d in acc], float)
        a_i = np.column_stack([np.interp(t_g, t_a, a[:, j]) for j in range(3)])
        t_imu, gyro, accel = t_g, w, a_i
    else:
        rows = [d for d in dump.get("IMU", []) if int(d.get("I", 0)) == instancia]
        if not rows:
            raise ValueError(f"no hay mensajes IMU de la instancia {instancia}")
        t_imu = np.array([d["TimeUS"] for d in rows], float) * 1e-6
        gyro = np.array([[d["GyrX"], d["GyrY"], d["GyrZ"]] for d in rows], float)
        accel = np.array([[d["AccX"], d["AccY"], d["AccZ"]] for d in rows], float)

    # orden temporal estricto (los logs pueden traer duplicados)
    orden = np.argsort(t_imu, kind="stable")
    t_imu, gyro, accel = t_imu[orden], gyro[orden], accel[orden]
    unicos = np.concatenate([[True], np.diff(t_imu) > 0])
    t_imu, gyro, accel = t_imu[unicos], gyro[unicos], accel[unicos]

    # test de cordura de unidades del acelerómetro (A.5 del anexo)
    n_med = float(np.median(np.linalg.norm(accel[:200], axis=1)))
    if verbose:
        print(f"||a|| mediana al inicio = {n_med:.2f} m/s²  (esperado ≈ {G_MAG})")
    if not (0.5 * G_MAG < n_med < 2.0 * G_MAG):
        print("  AVISO: el módulo del acelerómetro no parece m/s². Revisa el "
              "mensaje/instancia elegidos antes de seguir.")

    # ---------------- GT (EKF de ArduPilot) --------------------------------
    kw_gt = {}
    if fuente_gt == "att_pos" and dump.get("ATT") and dump.get("POS"):
        from scipy.spatial.transform import Rotation as Rot
        att, pos = dump["ATT"], dump["POS"]
        t_att = np.array([d["TimeUS"] for d in att], float) * 1e-6
        rpy = np.deg2rad(np.array([[d["Roll"], d["Pitch"], d["Yaw"]] for d in att], float))
        # ZYX intrínseco: R_ned_frd = Rz(yaw) Ry(pitch) Rx(roll)
        R_ned_frd = Rot.from_euler("ZYX", rpy[:, [2, 1, 0]]).as_matrix()
        R_ws = np.einsum("ij,kjl->kil", R_ENU_NED, R_ned_frd)

        t_pos = np.array([d["TimeUS"] for d in pos], float) * 1e-6
        lat = np.array([_lat_lng_grados(d["Lat"]) for d in pos])
        lng = np.array([_lat_lng_grados(d["Lng"]) for d in pos])
        alt = np.array([float(d["Alt"]) for d in pos])
        lat0, lng0, alt0 = lat[0], lng[0], alt[0]
        x_e = np.deg2rad(lng - lng0) * np.cos(np.deg2rad(lat0)) * R_TIERRA
        y_n = np.deg2rad(lat - lat0) * R_TIERRA
        z_u = alt - alt0
        p_pos = np.column_stack([x_e, y_n, z_u])
        # interpolar posición sobre los tiempos de ATT
        p = np.column_stack([np.interp(t_att, t_pos, p_pos[:, j]) for j in range(3)])
        v = np.gradient(p, t_att, axis=0)
        kw_gt = dict(t_gt=t_att, gt_p=p, gt_R=R_ws, gt_v=v)
        if verbose:
            print(f"GT desde ATT+POS: {len(t_att)} muestras "
                  f"({1.0/np.median(np.diff(t_att)):.0f} Hz). Recuerda: es el EKF "
                  "del autopiloto, no un mocap.")

    # ---------------- cámara ------------------------------------------------
    offset = float((cfg_dataset.get("camara", {}) or {}).get("offset_temporal_s", 0.0))
    if rutas_frames is None:
        rutas_frames, t_frames = [], np.array([])
    t_cam = np.asarray(t_frames, dtype=float) + offset

    # ---------------- base de tiempo común ---------------------------------
    candidatos = [t_imu[0]] + ([t_cam[0]] if len(t_cam) else []) \
        + ([kw_gt["t_gt"][0]] if kw_gt else [])
    t0 = float(min(candidatos))
    t_imu = t_imu - t0
    if len(t_cam):
        t_cam = t_cam - t0
    if kw_gt:
        kw_gt["t_gt"] = kw_gt["t_gt"] - t0

    return Secuencia(
        nombre=ruta.stem, t0_ns=int(round(t0 * 1e9)),
        t_imu=t_imu, gyro=gyro, accel=accel,
        t_cam=t_cam, rutas=list(rutas_frames),
        camara=camara, imu_params=imu_params or ParamsImu(), **kw_gt)


def tiempos_trigger(ruta_log) -> np.ndarray:
    """Tiempos [s, base TimeUS] de los mensajes CAM/TRIG (triggers de cámara)."""
    dump = volcar_mensajes(ruta_log, {"CAM", "TRIG"})
    filas = dump.get("CAM", []) + dump.get("TRIG", [])
    if not filas:
        return np.array([])
    t = np.array([d["TimeUS"] for d in filas], float) * 1e-6
    return np.sort(t)
