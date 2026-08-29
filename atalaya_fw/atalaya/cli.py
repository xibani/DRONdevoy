"""
atalaya.cli — línea de comandos.

    python -m atalaya selftest
    python -m atalaya inspeccionar   config.yaml
    python -m atalaya extraer-frames config.yaml
    python -m atalaya offset         config.yaml
    python -m atalaya ejecutar       config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _cargar(ruta_cfg, verbose=True):
    from .config import cargar_config, construir_secuencia
    cfg = cargar_config(ruta_cfg)
    return cfg, construir_secuencia(cfg, verbose=verbose)


def cmd_selftest(_args):
    from .selftest import correr_todo
    correr_todo()


def cmd_inspeccionar(args):
    from .config import cargar_config
    cfg = cargar_config(args.config)
    d = cfg["dataset"]
    if str(d.get("tipo", "")).lower() == "ardupilot":
        from .datasets.ardupilot import inventario, tiempos_trigger
        inv = inventario(d["log"])
        print(f"{'mensaje':10s} {'n':>8s} {'Hz':>8s}")
        for tt, n, hz in inv["filas"]:
            print(f"{tt:10s} {n:8d} {hz:8.1f}")
        trig = tiempos_trigger(d["log"])
        if len(trig):
            print(f"\ntriggers de cámara (CAM/TRIG): {len(trig)}  "
                  f"dt mediana {np.median(np.diff(trig))*1e3:.1f} ms")
        print("\nElige en el YAML: dataset.imu.mensaje (IMU o GYR+ACC), "
              "dataset.imu.instancia, dataset.gt.fuente (att_pos|ninguna).")
    else:
        cfg2, seq = _cargar(args.config)
        print(seq)
        tramo = seq.tramo_estatico()
        print(f"tramo estático inicial: {tramo if tramo else 'NO detectado'}")


def cmd_extraer_frames(args):
    from .config import cargar_config
    from .datasets.video import componer_frames
    cfg = cargar_config(args.config)
    rutas, t = componer_frames(cfg["dataset"], verbose=True)
    print(f"{len(rutas)} frames listos; ya puedes correr `ejecutar`.")


def cmd_offset(args):
    from .time_offset import estimar_offset
    cfg, seq = _cargar(args.config)
    tramo = seq.tramo_estatico()
    bg = None
    if tramo:
        m = (seq.t_imu >= tramo[0]) & (seq.t_imu <= tramo[1])
        bg = seq.gyro[m].mean(0)
    k0 = int(np.searchsorted(seq.t_cam, args.t_ini)) if args.t_ini else 0
    k1 = int(np.searchsorted(seq.t_cam, args.t_fin)) if args.t_fin else min(
        k0 + 600, len(seq.t_cam))
    estimar_offset(seq, k0, k1, bg=bg, rango_s=args.rango)


def cmd_ejecutar(args):
    from .config import config_eskf, config_frontend
    from .eskf import ejecutar_eskf
    from .evaluacion import informe
    from .frontend import construir_medidas
    from .inicializacion import estado_inicial

    cfg, seq = _cargar(args.config)
    if len(seq.t_cam) < 10:
        sys.exit("la secuencia no tiene imágenes: corre antes `extraer-frames` "
                 "o revisa dataset.camara en el YAML")

    c_fe, c_kf = config_frontend(cfg), config_eskf(cfg)
    ini = cfg["inicializacion"]
    tipo_ini = str(ini.get("tipo", "gt" if seq.tiene_gt else "estatica"))

    t_ini = cfg["eskf"].get("t_ini")
    t_fin = cfg["eskf"].get("t_fin")

    # arranque tras el tramo estático si lo hay (que la escala nazca con movimiento)
    tramo = seq.tramo_estatico(t_max=float(ini.get("t_estatico_max", 10.0)))
    t_arranque = tramo[1] if (tramo and tipo_ini == "estatica") else \
        (tramo[1] if tramo else seq.t_cam[0])
    k0 = int(np.searchsorted(seq.t_cam, t_ini if t_ini is not None else t_arranque))
    k1 = int(np.searchsorted(seq.t_cam, t_fin)) if t_fin is not None else len(seq.t_cam) - 1
    k0 = max(k0, 0); k1 = min(k1, len(seq.t_cam) - 1)
    print(f"\nventana: frames [{k0}, {k1}) = "
          f"[{seq.t_cam[k0]:.1f}, {seq.t_cam[k1-1]:.1f}] s")

    print("\n--- front-end ---")
    medidas = construir_medidas(seq, k0, k1, c_fe)
    if not medidas:
        sys.exit("el front-end no produjo medidas: revisa imágenes, "
                 "intrínsecos y paralaje_min")

    print("\n--- inicialización ---")
    estado0, _ = estado_inicial(seq, seq.t_cam[k0], tipo=tipo_ini,
                                t_estatico_max=float(ini.get("t_estatico_max", 10.0)))

    print(f"\n--- ESKF (modo {c_kf.modo}) ---")
    res = ejecutar_eskf(seq, medidas, k0, k1, estado0, c_kf)
    res["init_tipo"] = tipo_ini
    print(f"medidas {res['n_medidas']}   aceptadas "
          f"{100*res['n_aceptadas']/max(res['n_medidas'],1):.1f} %")

    res_dr = None
    if not args.sin_dead_reckoning:
        print("\n--- dead reckoning (IMU sola, misma ventana) ---")
        res_dr = ejecutar_eskf(seq, [], k0, k1, estado0, c_kf, solo_imu=True)

    print("\n--- informe ---")
    dir_salida = Path(cfg["salida"].get("directorio",
                                        f"results/{seq.nombre}")).expanduser()
    informe(seq, res, dir_salida, res_dr=res_dr, medidas=medidas)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="atalaya",
                                 description="Framework VIO del proyecto ATALAYA")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest", help="validación sintética sin datos")

    p = sub.add_parser("inspeccionar", help="qué hay en el dataset/log")
    p.add_argument("config")

    p = sub.add_parser("extraer-frames", help="componer frames desde el vídeo")
    p.add_argument("config")

    p = sub.add_parser("offset", help="estimar offset temporal cámara-IMU")
    p.add_argument("config")
    p.add_argument("--t-ini", type=float, default=None)
    p.add_argument("--t-fin", type=float, default=None)
    p.add_argument("--rango", type=float, default=0.5, help="± s del barrido")

    p = sub.add_parser("ejecutar", help="front-end + ESKF + informe")
    p.add_argument("config")
    p.add_argument("--sin-dead-reckoning", action="store_true")

    args = ap.parse_args(argv)
    {"selftest": cmd_selftest, "inspeccionar": cmd_inspeccionar,
     "extraer-frames": cmd_extraer_frames, "offset": cmd_offset,
     "ejecutar": cmd_ejecutar}[args.cmd](args)


if __name__ == "__main__":
    main()
