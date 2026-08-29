#!/usr/bin/env python3
"""ATALAYA — servidor API local para la interfaz gráfica.

Puente REST entre una interfaz web (generada con Lovable, Google Stitch, etc.)
y el paquete `atalaya`. La interfaz NO ejecuta Python: solo habla HTTP con
este servidor, que corre en la misma máquina que los datos.

Uso:
    pip install fastapi "uvicorn[standard]"
    python servidor_api.py                # escucha en http://localhost:8420

Diseño:
- Los comandos largos (selftest, extraer-frames, offset, ejecutar) se lanzan
  como TRABAJOS: subprocesos `python -m atalaya ...` cuyo stdout se captura en
  vivo. La interfaz crea el trabajo (POST /api/trabajos) y sondea su estado y
  log (GET /api/trabajos/{id}) cada ~1 s. Así el servidor nunca se bloquea y
  el log que ve el usuario es EXACTAMENTE el del CLI (una sola fuente de
  verdad; nada se re-implementa aquí).
- Los YAML de configs/ se leen y escriben como texto; el servidor valida que
  parsean antes de guardar y ofrece un setter puntual por ruta
  ("dataset.camara.offset_temporal_s") para el botón «aplicar offset».
- Los informes de results/ se listan y sirven tal cual (PNG, TUM, npz, txt).

Seguridad: pensado para localhost. CORS abierto porque el frontend de
Lovable/Stitch se sirve desde otro origen durante el desarrollo. No lo
expongas a Internet: /api/explorar lista el sistema de archivos local.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

RAIZ = Path(__file__).resolve().parent          # carpeta que contiene atalaya/
DIR_CONFIGS = RAIZ / "configs"
DIR_RESULTS = RAIZ / "results"
PUERTO = 8420

app = FastAPI(title="ATALAYA API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# ----------------------------------------------------------------- trabajos
TRABAJOS: dict[str, dict] = {}                  # id -> estado en memoria
_LOCK = threading.Lock()

COMANDOS = {"selftest", "inspeccionar", "extraer-frames", "offset", "ejecutar"}


class NuevoTrabajo(BaseModel):
    comando: str                                # uno de COMANDOS
    config: str | None = None                   # nombre de YAML en configs/
    t_ini: float | None = None                  # solo offset
    t_fin: float | None = None                  # solo offset
    rango: float | None = None                  # solo offset (s de búsqueda)
    sin_dead_reckoning: bool = False            # solo ejecutar


def _linea_comando(t: NuevoTrabajo) -> list[str]:
    cmd = [sys.executable, "-u", "-m", "atalaya", t.comando]
    if t.comando != "selftest":
        if not t.config:
            raise HTTPException(400, "este comando necesita 'config'")
        cmd.append(str(DIR_CONFIGS / t.config))
    if t.comando == "offset":
        if t.t_ini is not None:
            cmd += ["--t-ini", str(t.t_ini)]
        if t.t_fin is not None:
            cmd += ["--t-fin", str(t.t_fin)]
        if t.rango is not None:
            cmd += ["--rango", str(t.rango)]
    if t.comando == "ejecutar" and t.sin_dead_reckoning:
        cmd.append("--sin-dead-reckoning")
    return cmd


def _extraer_resultado(comando: str, log: str) -> dict:
    """Datos estructurados que a la interfaz le interesan del log."""
    res: dict = {}
    if comando == "offset":
        m = re.search(r"offset estimado = ([+-]?[\d.]+) ms", log)
        if m:
            res["offset_s"] = round(float(m.group(1)) / 1000.0, 6)
        res["aviso_contraste"] = "correlación casi plana" in log
    if comando == "ejecutar":
        m = re.search(r"informe escrito en (.+)", log)
        if m:
            res["directorio"] = m.group(1).strip()
        for clave, patron in [("ate_se3", r"ATE SE\(3\)[^\d]*([\d.]+)"),
                              ("ate_sim3", r"ATE Sim\(3\)[^\d]*([\d.]+)"),
                              ("aceptadas_pct", r"aceptadas[^\d]*([\d.]+)\s*%")]:
            m = re.search(patron, log)
            if m:
                res[clave] = float(m.group(1))
    if comando == "selftest":
        res["ok"] = "TODO OK" in log
    return res


def _correr(tid: str, cmd: list[str]) -> None:
    tr = TRABAJOS[tid]
    tr.update(estado="corriendo", inicio=time.time())
    try:
        # UTF-8 explícito en ambos lados: en Windows el hijo y el padre no
        # comparten codificación por defecto y los acentos del log se rompen.
        proc = subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace",
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        tr["pid"] = proc.pid
        for linea in proc.stdout:               # streaming al log en memoria
            with _LOCK:
                tr["log"] += linea
        rc = proc.wait()
        tr["estado"] = "ok" if rc == 0 else "error"
        tr["codigo"] = rc
    except Exception as e:                      # p. ej. binario ausente
        tr["log"] += f"\n[servidor] excepción: {e}\n"
        tr["estado"] = "error"
    tr["fin"] = time.time()
    tr["resultado"] = _extraer_resultado(tr["comando"], tr["log"])


@app.post("/api/trabajos")
def crear_trabajo(t: NuevoTrabajo):
    if t.comando not in COMANDOS:
        raise HTTPException(400, f"comando desconocido: {t.comando}")
    cmd = _linea_comando(t)
    tid = uuid.uuid4().hex[:12]
    TRABAJOS[tid] = {"id": tid, "comando": t.comando, "config": t.config,
                     "estado": "en_cola", "log": "", "resultado": {},
                     "inicio": None, "fin": None, "cmd": " ".join(cmd)}
    threading.Thread(target=_correr, args=(tid, cmd), daemon=True).start()
    return {"id": tid}


@app.get("/api/trabajos")
def listar_trabajos():
    with _LOCK:
        return [{k: v for k, v in tr.items() if k != "log"}
                for tr in sorted(TRABAJOS.values(),
                                 key=lambda x: x["inicio"] or 1e18)]


@app.get("/api/trabajos/{tid}")
def ver_trabajo(tid: str, desde: int = 0):
    """`desde` = nº de caracteres de log ya recibidos (sondeo incremental)."""
    tr = TRABAJOS.get(tid)
    if not tr:
        raise HTTPException(404, "trabajo no encontrado")
    with _LOCK:
        log = tr["log"]
    out = {k: v for k, v in tr.items() if k != "log"}
    out["log"] = log[desde:]
    out["log_total"] = len(log)
    return out


# ------------------------------------------------------------------ configs
class TextoYaml(BaseModel):
    texto: str


class CampoYaml(BaseModel):
    ruta: str                                   # "dataset.camara.offset_temporal_s"
    valor: float | int | str | bool | None


@app.get("/api/configs")
def listar_configs():
    DIR_CONFIGS.mkdir(exist_ok=True)
    return sorted(p.name for p in DIR_CONFIGS.glob("*.yaml"))


def _ruta_config(nombre: str) -> Path:
    if "/" in nombre or ".." in nombre or not nombre.endswith(".yaml"):
        raise HTTPException(400, "nombre de config inválido")
    return DIR_CONFIGS / nombre


@app.get("/api/configs/{nombre}")
def leer_config(nombre: str):
    p = _ruta_config(nombre)
    if not p.exists():
        raise HTTPException(404, "no existe")
    texto = p.read_text()
    return {"nombre": nombre, "texto": texto,
            "contenido": yaml.safe_load(texto)}


@app.put("/api/configs/{nombre}")
def guardar_config(nombre: str, body: TextoYaml):
    p = _ruta_config(nombre)
    try:
        yaml.safe_load(body.texto)              # valida antes de escribir
    except yaml.YAMLError as e:
        raise HTTPException(422, f"YAML inválido: {e}")
    p.write_text(body.texto)
    return {"ok": True}


@app.post("/api/configs/{nombre}/campo")
def poner_campo(nombre: str, body: CampoYaml):
    """Setter puntual (p. ej. aplicar el offset estimado sin tocar el resto)."""
    p = _ruta_config(nombre)
    if not p.exists():
        raise HTTPException(404, "no existe")
    cfg = yaml.safe_load(p.read_text()) or {}
    nodo, partes = cfg, body.ruta.split(".")
    for k in partes[:-1]:
        nodo = nodo.setdefault(k, {})
        if not isinstance(nodo, dict):
            raise HTTPException(422, f"'{k}' no es un bloque en el YAML")
    nodo[partes[-1]] = body.valor
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return {"ok": True, "ruta": body.ruta, "valor": body.valor}


# --------------------------------------------------------------- resultados
@app.get("/api/resultados")
def listar_resultados():
    DIR_RESULTS.mkdir(exist_ok=True)
    out = []
    for d in sorted(DIR_RESULTS.iterdir()):
        if not d.is_dir():
            continue
        resumen = (d / "resumen.txt")
        out.append({"nombre": d.name,
                    "archivos": sorted(f.name for f in d.iterdir()
                                       if f.is_file()),
                    "resumen": resumen.read_text() if resumen.exists() else None,
                    "modificado": d.stat().st_mtime})
    return out


@app.get("/api/resultados/{nombre}/archivo/{fname}")
def servir_resultado(nombre: str, fname: str):
    p = (DIR_RESULTS / nombre / fname).resolve()
    if DIR_RESULTS.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "no existe")
    return FileResponse(p)


# ------------------------------------------------------------------- varios
@app.get("/api/estado")
def estado():
    try:
        import pymavlink                        # noqa: F401
        mav = True
    except ImportError:
        mav = False
    from atalaya import __version__
    return {"ok": True, "version": __version__, "raiz": str(RAIZ),
            "pymavlink": mav, "configs": len(list(DIR_CONFIGS.glob("*.yaml")))
            if DIR_CONFIGS.exists() else 0}


@app.get("/api/explorar")
def explorar(ruta: str = "."):
    """Selector de archivos para la interfaz (logs .BIN, vídeos...)."""
    p = (RAIZ / ruta).resolve() if not Path(ruta).is_absolute() \
        else Path(ruta).resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, "directorio no existe")
    dirs, files = [], []
    try:
        for h in sorted(p.iterdir()):
            (dirs if h.is_dir() else files).append(h.name)
    except PermissionError:
        raise HTTPException(403, "sin permiso")
    return {"ruta": str(p), "padre": str(p.parent), "dirs": dirs,
            "archivos": files}


if __name__ == "__main__":
    import uvicorn
    print(f"ATALAYA API en http://localhost:{PUERTO}  (raíz: {RAIZ})")
    uvicorn.run(app, host="127.0.0.1", port=PUERTO, log_level="warning")
