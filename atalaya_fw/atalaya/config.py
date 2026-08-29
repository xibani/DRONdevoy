"""
atalaya.config — YAML de configuración y fábrica de secuencias.

El YAML es el único punto de decisión del usuario: qué dataset, qué mensajes
del log, qué cámara, qué modo del filtro. Ver configs/*.yaml de ejemplo.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .eskf import ConfigEskf
from .frontend import ConfigFrontend
from .sensores import ParamsImu, camara_desde_config


def cargar_config(ruta) -> dict:
    with open(Path(ruta).expanduser()) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("dataset", {})
    cfg.setdefault("frontend", {})
    cfg.setdefault("eskf", {})
    cfg.setdefault("inicializacion", {})
    cfg.setdefault("salida", {})
    return cfg


def config_frontend(cfg) -> ConfigFrontend:
    c = ConfigFrontend()
    for k, v in (cfg.get("frontend") or {}).items():
        if hasattr(c, k):
            setattr(c, k, type(getattr(c, k))(v) if not isinstance(v, (list, tuple))
                    else tuple(v))
    return c


def config_eskf(cfg) -> ConfigEskf:
    c = ConfigEskf()
    for k, v in (cfg.get("eskf") or {}).items():
        if hasattr(c, k):
            setattr(c, k, type(getattr(c, k))(v))
    return c


def imu_desde_config(cfg) -> ParamsImu | None:
    b = cfg.get("imu")
    if not b:
        return None
    p = ParamsImu()
    for k, v in b.items():
        if hasattr(p, k):
            setattr(p, k, float(v))
    return p


def construir_secuencia(cfg, verbose=True):
    """Fábrica: devuelve una Secuencia según dataset.tipo."""
    d = cfg["dataset"]
    tipo = str(d.get("tipo", "euroc")).lower()

    if tipo == "euroc":
        from .datasets.euroc import cargar_euroc
        seq = cargar_euroc(d["raiz"], d.get("camara_euroc", "cam0"))
        if verbose:
            print(seq)
        return seq

    camara = camara_desde_config(cfg["camara"])
    imu_p = imu_desde_config(cfg)

    if tipo == "ardupilot":
        from .datasets.ardupilot import cargar_ardupilot
        from .datasets.video import componer_frames
        rutas, t_fr = ([], [])
        if (d.get("camara") or {}).get("video") or \
                Path(str((d.get("camara") or {}).get("directorio_frames", "")),
                     ).joinpath("data.csv").exists():
            rutas, t_fr = componer_frames(d, verbose=verbose)
        seq = cargar_ardupilot(d, camara, imu_p, rutas, t_fr, verbose=verbose)
        if verbose:
            print(seq)
        return seq

    if tipo == "generico":
        from .datasets.generico import cargar_generico
        return cargar_generico(d, camara, imu_p, verbose=verbose)

    raise ValueError(f"dataset.tipo='{tipo}' no reconocido (euroc|ardupilot|generico)")
