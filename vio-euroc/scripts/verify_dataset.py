"""
verify_dataset.py - Test de aceptación de la Fase 0 para el dataset EuRoC (formato ASL).

Uso:
    python verify_dataset.py ~/data/MH_01_easy/mav0

No implementa nada de VIO. Su único trabajo es responder: ¿los datos que tengo
en disco son los que creo que son? Si algo aqui falla, arreglalo ANTES de pasar
a la Fase 1.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NS = 1e9  # nanosegundos -> segundos


def separador(titulo):
    print(f"\n{'=' * 62}\n{titulo}\n{'=' * 62}")


def check(condicion, mensaje_ok, mensaje_fallo):
    """Imprime el resultado de una comprobación y devuelve el booleano."""
    print(
        f"   [{'OK ' if condicion else 'FAIL'}] {mensaje_ok if condicion else mensaje_fallo}"
    )
    return condicion


def main(ruta_mav0):
    mav0 = Path(ruta_mav0).expanduser().resolve()
    fallos = []

    # -----------------------------------------------------------------
    # 1. ESTRUCTURA DE CARPETAS
    # -----------------------------------------------------------------
    separador("1. ESTRUCTURA")
    print(f"   Raíz: {mav0}")

    esperados = [
        "cam0/data.csv",
        "cam0/sensor.yaml",
        "cam1/data.csv",
        "imu0/data.csv",
        "imu0/sensor.yaml",
        "state_groundtruth_estimate0/data.csv",
    ]
    for rel in esperados:
        if not check((mav0 / rel).exists(), f"existe {rel}", f"FALTA {rel}"):
            fallos.append(rel)

    if fallos:
        print("\n  Faltan ficheros esenciales. ¿Apuntaste a la carpeta 'mav0'?")
        print("\n  Pista: prueba 'find <tu_carpeta> -name sensor.yaml'")
        return

    # -----------------------------------------------------------------
    # 2. IMU
    # -----------------------------------------------------------------
    separador("2. IMU (imu0)")

    # Las columnas van en este orden: GIRO primero, ACELERÓMETRO después.
    imu = pd.read_csv(
        mav0 / "imu0/data.csv",
        comment="#",
        header=0,
        names=["t_ns", "wx", "wy", "wz", "ax", "ay", "az"],
    )
    imu["t_ns"] = imu["t_ns"].astype(np.int64)
    t_imu = imu["t_ns"].to_numpy() / NS

    print(f"  Muestras: {len(imu)}")
    print(f"  Duración: {t_imu[-1] - t_imu[0]:.2f} s")

    dt = np.diff(t_imu)
    hz_imu = 1.0 / np.median(dt)
    print(f"  Frecuencia mediana: {hz_imu:.1f} Hz")
    print(f"  dt: min={dt.min() * 1e3:.3f} ms  max={dt.max() * 1e3:.3f} ms")

    check(
        np.all(dt > 0),
        "timestamps estrictamente crecientes",
        "hay timestamps repetidos o hacia atras",
    )
    check(
        190 < hz_imu < 210,
        "frecuencia ~200 Hz (esperado en EuRoC)",
        f"frecuencia inesperada: {hz_imu:.1f} Hz",
    )
    check(
        dt.max() < 0.05,
        "sin huecos grandes (>50 ms) en la IMU",
        f"hueco de {dt.max() * 1e3:.1f} ms detectado",
    )

    # SANITY CHECK DE UNIDADES Y ORDEN DE COLUMNAS.
    # El dron arranca en reposo sobre el suelo: el acelerómetro mide la
    # aceleración específica, cuya norma debe ser ~9.81 m/s^2 (la gravedad).
    accel = imu[["ax", "ay", "az"]].to_numpy()
    gyro = imu[["wx", "wy", "wz"]].to_numpy()
    n_ini = min(400, len(imu))  # ~2 s iniciales a 200 Hz)
    norma_accel = np.linalg.norm(accel[:n_ini], axis=1).mean()
    norma_gyro = np.linalg.norm(gyro[:n_ini], axis=1).mean()

    print(f"\n  Primeros {n_ini} samples (dron en reposo):")
    print(f"    |accel| medio = {norma_accel:.3f} m/s^2   (esperado ~9.81)")
    print(f"    |gyro|  medio = {norma_gyro:.4f} rad/s    (esperado ~0.0x)")
    print(f"    accel medio por eje = {accel[:n_ini].mean(axis=0)}")

    ok_unidades = check(
        8.5 < norma_accel < 11.0,
        "la norma del acelerómetro es coherente con la gravedad (m/s^2)",
        "norma del acelerómetro FUERA de rango: ¿columnas gyro/accel intercambiadas, o unidades en g?",
    )
    check(
        norma_gyro < 0.5,
        "el giroscopio está casi a cero en reposo (rad/s)",
        "el giroscopio no está a cero en reposo: ¿columnas intercambiadas?",
    )
    if not ok_unidades:
        fallos.append("unidades IMU")

    # -----------------------------------------------------------------
    # 3. CÁMARAS
    # -----------------------------------------------------------------
    separador("3. CÁMARAS (cam0 / cam1)")

    cam0 = pd.read_csv(
        mav0 / "cam0/data.csv", comment="#", header=0, names=["t_ns", "filename"]
    )
    cam0["t_ns"] = cam0["t_ns"].astype(np.int64)
    t_cam = cam0["t_ns"].to_numpy() / NS

    cam1 = pd.read_csv(
        mav0 / "cam1/data.csv", comment="#", header=0, names=["t_ns", "filename"]
    )
    cam1["t_ns"] = cam1["t_ns"].astype(np.int64)

    dt_cam = np.diff(t_cam)
    hz_cam = 1.0 / np.median(dt_cam)
    print(f"  Frames en cam0.csv: {len(cam0)}")
    print(f"  Frecuencia mediana: {hz_cam:.2f} Hz")
    check(
        19 < hz_cam < 21,
        "frecuencia ~20 Hz (esperado en EuRoC)",
        f"frecuencia inesperada: {hz_cam:.2f} Hz",
    )

    # Los PNG deben existir en disco, no solo estar listados en el CSV.
    dir_img = mav0 / "cam0/data"
    pngs = sorted(dir_img.glob("*.png"))
    print(f"  PNGs en cam0/data/: {len(pngs)}")
    check(
        len(pngs) == len(cam0),
        "cada fila del CSV tiene su PNG en disco",
        f"descuadre: {len(cam0)} filas de CSV vs {len(pngs)} PNGs (¿unzip incompleto?)",
    )

    # El nombre del fichero ES el timestamp en nanosegundos.
    if pngs:
        stem = pngs[0].stem
        check(
            stem.isdigit() and int(stem) == cam0["t_ns"].iloc[0],
            f"el nombre del PNG es el timestamp en ns ({stem})",
            f"el nombre del primer PNG ({stem}) no coincide con el primer timestamp del CSV",
        )

    # Estéreo hardware-sincronizado: cam0 y cam1 comparten timestamp exacto.
    n_comun = min(len(t_cam), len(cam1))
    desfase = np.abs(
        cam0["t_ns"].to_numpy()[:n_comun] - cam1["t_ns"].to_numpy()[:n_comun]
    )
    print(f"  Desfase máximo cam0 vs cam1: {desfase.max()} ns")
    check(
        desfase.max() == 0,
        "cam0 y cam1 están sincronizadas por hardware",
        "cam0 y cam1 NO comparten timestamps",
    )

    # -----------------------------------------------------------------
    # 4. SINCRONIZACIÓN IMU <-> CÁMARA
    # -----------------------------------------------------------------
    separador("4. SINCRONIZACIÓN")

    print(f"  IMU:    [{t_imu[0]:.6f} , {t_imu[-1]:.6f}] s")
    print(f"  Cámara: [{t_cam[0]:.6f} , {t_cam[-1]:.6f}] s")

    solape = min(t_imu[-1], t_cam[-1]) - max(t_imu[0], t_cam[0])
    print(f"  Solape temporal: {solape:.2f} s")
    check(
        solape > 0.9 * (t_cam[-1] - t_cam[0]),
        "IMU y cámara se solapan casi por completo",
        "solape insuficiente",
    )

    # Cuántas muestras de IMU caen entre dos frames consecutivos.
    # A 200 Hz de IMU y 20 Hz de cámara, esperamos ~10.
    idx = np.searchsorted(t_imu, t_cam)
    por_frame = np.diff(idx)
    print(
        f"  Muestras de IMU por frame: mediana={np.median(por_frame):.0f} "
        f"min={por_frame.min()} max={por_frame.max()}"
    )
    check(
        8 <= np.median(por_frame) <= 12,
        "~10 muestras de IMU por frame",
        "ratio IMU/cámara inesperado",
    )

    # -----------------------------------------------------------------
    # 5. GROUND TRUTH
    # -----------------------------------------------------------------
    separador("5. GROUND TRUTH")

    # 17 columnas: t, posición, cuaternión (w PRIMERO), velocidad, bias gyro, bias accel.
    cols_gt = [
        "t_ns",
        "px",
        "py",
        "pz",
        "qw",
        "qx",
        "qy",
        "qz",
        "vx",
        "vy",
        "vz",
        "bwx",
        "bwy",
        "bwz",
        "bax",
        "bay",
        "baz",
    ]
    gt = pd.read_csv(
        mav0 / "state_groundtruth_estimate0/data.csv",
        comment="#",
        header=0,
        names=cols_gt,
    )
    gt["t_ns"] = gt["t_ns"].astype(np.int64)
    t_gt = gt["t_ns"].to_numpy() / NS

    print(f"  Muestras: {len(gt)}  ({1.0 / np.median(np.diff(t_gt)):.0f} Hz)")
    print(f"  Rango:    [{t_gt[0]:.6f} , {t_gt[-1]:.6f}] s")
    print(
        f"  Posición inicial: [{gt.px.iloc[0]:.3f} {gt.py.iloc[0]:.3f} {gt.pz.iloc[0]:.3f}] m"
    )
    recorrido = np.linalg.norm(
        np.diff(gt[["px", "py", "pz"]].to_numpy(), axis=0), axis=1
    ).sum()
    print(f"  Recorrido total:  {recorrido:.1f} m")

    # El cuaternión debe estar normalizado. Si no lo está, el orden de columnas
    # está mal o hay filas corruptas.
    norma_q = np.linalg.norm(gt[["qw", "qx", "qy", "qz"]].to_numpy(), axis=1)
    print(f"  |q|: min={norma_q.min():.6f} max={norma_q.max():.6f} (esperado 1.0)")
    check(
        np.allclose(norma_q, 1.0, atol=1e-3),
        "cuaterniones normalizados (orden w,x,y,z correcto)",
        "cuaterniones NO normalizados: revisa el orden de columnas",
    )

    # El GT no cubre necesariamente todo el recorrido de la cámara.
    print(
        f"  Cobertura GT sobre el tramo de cámara: "
        f"{100 * (min(t_gt[-1], t_cam[-1]) - max(t_gt[0], t_cam[0])) / (t_cam[-1] - t_cam[0]):.1f} %"
    )

    # -----------------------------------------------------------------
    # 6. CALIBRACIÓN
    # -----------------------------------------------------------------
    separador("6. CALIBRACIÓN")

    with open(mav0 / "cam0/sensor.yaml") as f:
        cfg_cam = yaml.safe_load(f)
    with open(mav0 / "imu0/sensor.yaml") as f:
        cfg_imu = yaml.safe_load(f)

    print(
        f"  Modelo de cámara: {cfg_cam['camera_model']} / {cfg_cam['distortion_model']}"
    )
    print(f"  Resolución:       {cfg_cam['resolution']}")
    print(f"  Intrínsecos [fu fv cu cv]: {cfg_cam['intrinsics']}")
    print(f"  Distorsión  [k1 k2 p1 p2]: {cfg_cam['distortion_coefficients']}")

    # T_BS de cam0: transformada del body frame (= IMU) al frame del sensor.
    T_BS_cam = np.array(cfg_cam["T_BS"]["data"]).reshape(4, 4)
    T_BS_imu = np.array(cfg_imu["T_BS"]["data"]).reshape(4, 4)
    print("\n  T_BS de cam0 (body/IMU -> cámara):")
    for fila in T_BS_cam:
        print("    " + "  ".join(f"{v: .6f}" for v in fila))
    print(
        f"  Traslación cámara-IMU: {T_BS_cam[:3, 3]} m "
        f"(norma {np.linalg.norm(T_BS_cam[:3, 3]) * 100:.1f} cm)"
    )

    # Que T_BS del IMU sea la identidad confirma que el body frame ES el IMU.
    check(
        np.allclose(T_BS_imu, np.eye(4)),
        "T_BS del IMU es la identidad -> el body frame es el frame del IMU",
        "T_BS del IMU no es la identidad (inesperado en EuRoC)",
    )
    # Una matriz de rotación válida cumple R^T R = I y det(R) = +1.
    R = T_BS_cam[:3, :3]
    check(
        np.allclose(R.T @ R, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(R), 1.0, atol=1e-6),
        "la rotación de T_BS es ortonormal con det=+1",
        "la rotación de T_BS no es válida (¿mal reshape? EuRoC usa row-major)",
    )

    print("\n  Parámetros de ruido de la IMU (para la Fase 6):")
    for clave in [
        "gyroscope_noise_density",
        "gyroscope_random_walk",
        "accelerometer_noise_density",
        "accelerometer_random_walk",
    ]:
        print(f"    {clave:32s} = {cfg_imu[clave]}")

    # -----------------------------------------------------------------
    # VEREDICTO
    # -----------------------------------------------------------------
    separador("VEREDICTO")
    if fallos:
        print("  Hay comprobaciones en FAIL. Revísalas antes de pasar a la Fase 1.")
        return 1
    print("  Dataset verificado. Puedes pasar a la Fase 1 (carga y sincronización).")
    print("\n  Anota estos tres números, los usarás constantemente:")
    print(f"    - IMU a {hz_imu:.0f} Hz, cámara a {hz_cam:.0f} Hz")
    print(f"    - ~{np.median(por_frame):.0f} muestras de IMU por frame")
    print(f"    - {recorrido:.1f} m de recorrido en {t_gt[-1] - t_gt[0]:.0f} s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
