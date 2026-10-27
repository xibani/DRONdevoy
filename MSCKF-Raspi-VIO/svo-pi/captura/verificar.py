#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
captura/verificar.py — Verificación de una grabación de captura/grabar.py.

Se ejecuta en: [Raspberry Pi] o en el PC ([Windows PowerShell] / [WSL/Ubuntu])
tras copiar la carpeta de la grabación.

Comprueba:
  1. Regularidad de la IMU: frecuencia, jitter (desv. típica de Δt), muestras perdidas.
  2. Regularidad de la cámara: frecuencia, frames perdidos, PNG presentes en disco.
  3. Base de reloj común: t_llegada - t_sensor debe ser positivo y pequeño.
  4. Tramo inicial en reposo: dirección de la gravedad, sesgo y ruido del giroscopio.
  5. Criterio de salida de F1 (config/captura.yaml -> verificacion.criterio).

Uso:
  python3 captura/verificar.py datos/20261026_183000
"""

import argparse
import csv
import json
import os

import numpy as np
import yaml

RUTA_CONFIG_DEFECTO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "config", "captura.yaml")


# =====================================================================
# Carga
# =====================================================================
def cargar_grabacion(raiz):
    """Lee una grabación en formato EuRoC + ficheros de diagnóstico.

    Devuelve un dict con arrays numpy:
      t_imu [ns int64], w [rad/s], a [m/s^2], temp [°C]
      t_cam [ns int64], t_llegada [ns int64], archivos (lista)
      meta (dict)
    """
    carpeta_imu = os.path.join(raiz, "mav0", "imu0")
    carpeta_cam = os.path.join(raiz, "mav0", "cam0")

    with open(os.path.join(raiz, "meta.json")) as f:
        meta = json.load(f)

    # IMU (EuRoC: t, wx, wy, wz, ax, ay, az). El timestamp se lee como entero
    # aparte: en float64 se perdería precisión en ns (2^53 ≈ 9e15 ns ≈ 104 días).
    t_imu = []
    filas = []
    with open(os.path.join(carpeta_imu, "data.csv")) as f:
        for linea in f:
            if linea.startswith("#") or not linea.strip():
                continue
            campos = linea.strip().split(",")
            t_imu.append(int(campos[0]))
            filas.append([float(c) for c in campos[1:7]])
    t_imu = np.array(t_imu, dtype=np.int64)
    filas = np.array(filas, dtype=np.float64)
    w = filas[:, 0:3]
    a = filas[:, 3:6]

    temp = None
    ruta_temp = os.path.join(carpeta_imu, "diagnostico.csv")
    if os.path.exists(ruta_temp):
        datos_temp = np.loadtxt(ruta_temp, delimiter=",", skiprows=1, ndmin=2)
        temp = datos_temp[:, 1]

    # Cámara (EuRoC: t, filename)
    t_cam = []
    archivos = []
    with open(os.path.join(carpeta_cam, "data.csv")) as f:
        for linea in f:
            if linea.startswith("#") or not linea.strip():
                continue
            t, nombre = linea.strip().split(",")
            t_cam.append(int(t))
            archivos.append(nombre)
    t_cam = np.array(t_cam, dtype=np.int64)

    # Diagnóstico de cámara: llegada al proceso
    t_llegada = None
    ruta_diag = os.path.join(carpeta_cam, "diagnostico.csv")
    if os.path.exists(ruta_diag):
        lista = []
        with open(ruta_diag) as f:
            for fila in csv.DictReader(f):
                lista.append(int(fila["t_llegada_ns"]))
        t_llegada = np.array(lista, dtype=np.int64)

    return {"meta": meta, "t_imu": t_imu, "w": w, "a": a, "temp": temp,
            "t_cam": t_cam, "t_llegada": t_llegada, "archivos": archivos,
            "carpeta_img": os.path.join(carpeta_cam, "data")}


# =====================================================================
# Análisis
# =====================================================================
def analizar_intervalos(t_ns, frecuencia_nominal_hz, factor_hueco):
    """Estadística de Δt entre marcas consecutivas.

    Muestras perdidas: un Δt ≈ k·T implica k-1 muestras ausentes.
    """
    T = 1.0 / frecuencia_nominal_hz
    dt = np.diff(t_ns).astype(np.float64) / 1e9
    duracion = (t_ns[-1] - t_ns[0]) / 1e9

    perdidas_por_intervalo = np.maximum(np.round(dt / T) - 1, 0)
    perdidas = int(np.sum(perdidas_por_intervalo))

    # Jitter calculado solo sobre intervalos "normales" (sin huecos), para
    # que una pérdida aislada no infle la desviación típica.
    # Ojo: si cada marca tiene ruido independiente σ, σ(Δt) = σ·√2.
    normales = dt[dt <= factor_hueco * T]

    return {
        "n": len(t_ns),
        "duracion_s": duracion,
        "frecuencia_hz": (len(t_ns) - 1) / duracion,
        "dt_medio_ms": dt.mean() * 1e3,
        "jitter_ms": normales.std() * 1e3 if len(normales) > 1 else float("nan"),
        "dt_min_ms": dt.min() * 1e3,
        "dt_max_ms": dt.max() * 1e3,
        "dt_p99_ms": np.percentile(dt, 99) * 1e3,
        "huecos": int(np.sum(dt > factor_hueco * T)),
        "perdidas": perdidas,
        "perdidas_pct": 100.0 * perdidas / (len(t_ns) + perdidas),
        "no_crecientes": int(np.sum(dt <= 0)),
        "dt_s": dt,
    }


def analizar_reloj(t_cam, t_llegada, t_imu, latencia_max_ms):
    """Comprueba que cámara e IMU comparten base de tiempos.

    Con el mismo reloj, t_llegada - t_sensor es la latencia de entrega del
    frame: positiva y de decenas de ms. Con relojes distintos saldrían
    valores negativos o de segundos/horas.
    """
    res = {"latencia_ms": None, "reloj_ok": None}
    if t_llegada is not None:
        lat = (t_llegada - t_cam).astype(np.float64) / 1e6
        res["latencia_ms"] = {"media": lat.mean(), "min": lat.min(), "max": lat.max()}
        res["reloj_ok"] = bool(lat.min() > 0 and lat.max() < latencia_max_ms)

    res["margen_ini_s"] = (t_cam[0] - t_imu[0]) / 1e9
    res["margen_fin_s"] = (t_imu[-1] - t_cam[-1]) / 1e9
    res["imu_cubre_cam"] = bool(res["margen_ini_s"] > 0 and res["margen_fin_s"] > 0)
    return res


def analizar_reposo(t_imu, w, a, t_reposo_s, g):
    """Primer tramo en reposo: f_medida = -R_w_i^T · g_w + b_a  ⇒  ‖ā‖ ≈ g.

    En reposo el acelerómetro mide la fuerza específica, que apunta hacia
    ARRIBA; la media del giroscopio es una estimación del sesgo b_g.
    """
    mascara = (t_imu - t_imu[0]) / 1e9 < t_reposo_s
    a_r = a[mascara]
    w_r = w[mascara]

    a_media = a_r.mean(axis=0)
    eje = int(np.argmax(np.abs(a_media)))
    return {
        "n": int(mascara.sum()),
        "a_media": a_media,
        "norma_a": float(np.linalg.norm(a_media)),
        "error_norma": float(np.linalg.norm(a_media) - g),
        "eje_arriba": ("+" if a_media[eje] > 0 else "-") + "xyz"[eje],
        "sesgo_gir_dps": np.degrees(w_r.mean(axis=0)),
        "ruido_gir_dps": np.degrees(w_r.std(axis=0)),
        "ruido_acc": a_r.std(axis=0),
    }


def evaluar_criterio(res_imu, res_cam, res_reloj, n_faltan_png, criterio):
    """Criterio de salida de F1. Devuelve lista de (descripción, ok)."""
    return [
        (f"duración ≥ {criterio['duracion_min_s']:.0f} s",
         res_cam["duracion_s"] >= criterio["duracion_min_s"]),
        (f"frames perdidos ≤ {criterio['imagenes_perdidas_max']}",
         res_cam["perdidas"] + n_faltan_png <= criterio["imagenes_perdidas_max"]),
        (f"IMU perdidas ≤ {criterio['imu_perdidas_max_pct']} %",
         res_imu["perdidas_pct"] <= criterio["imu_perdidas_max_pct"]),
        (f"jitter IMU ≤ {criterio['imu_jitter_max_ms']} ms",
         res_imu["jitter_ms"] <= criterio["imu_jitter_max_ms"]),
        ("timestamps estrictamente crecientes",
         res_imu["no_crecientes"] == 0 and res_cam["no_crecientes"] == 0),
        ("reloj común verificado", bool(res_reloj["reloj_ok"])),
        ("IMU cubre todo el tramo de imagen", res_reloj["imu_cubre_cam"]),
    ]


# =====================================================================
# Informe
# =====================================================================
def imprimir_intervalos(nombre, r, frecuencia_nominal_hz):
    print(f"\n=== {nombre} ===")
    print(f"  muestras:          {r['n']}")
    print(f"  duración:          {r['duracion_s']:.2f} s")
    print(f"  frecuencia media:  {r['frecuencia_hz']:.3f} Hz (nominal {frecuencia_nominal_hz:g})")
    print(f"  Δt medio:          {r['dt_medio_ms']:.3f} ms")
    print(f"  jitter (σ Δt):     {r['jitter_ms']:.3f} ms")
    print(f"  Δt mín/máx/p99:    {r['dt_min_ms']:.3f} / {r['dt_max_ms']:.3f} / {r['dt_p99_ms']:.3f} ms")
    print(f"  huecos:            {r['huecos']}")
    print(f"  perdidas:          {r['perdidas']} ({r['perdidas_pct']:.3f} %)")
    if r["no_crecientes"]:
        print(f"  ¡ATENCIÓN! {r['no_crecientes']} marcas no crecientes")


def guardar_figura(datos, res_imu, res_cam, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_imu = datos["t_imu"]
    t_rel = (t_imu - t_imu[0]) / 1e9
    fig, ejes = plt.subplots(2, 2, figsize=(13, 8))

    ejes[0, 0].hist(res_imu["dt_s"] * 1e3, bins=100)
    ejes[0, 0].set_title("Δt IMU")
    ejes[0, 0].set_xlabel("ms")
    ejes[0, 0].set_yscale("log")

    ejes[0, 1].hist(res_cam["dt_s"] * 1e3, bins=50)
    ejes[0, 1].set_title("Δt cámara")
    ejes[0, 1].set_xlabel("ms")
    ejes[0, 1].set_yscale("log")

    for i, eje in enumerate("xyz"):
        ejes[1, 0].plot(t_rel, datos["a"][:, i], lw=0.6, label=eje)
        ejes[1, 1].plot(t_rel, np.degrees(datos["w"][:, i]), lw=0.6, label=eje)
    t0 = (datos["t_cam"][0] - t_imu[0]) / 1e9
    t1 = (datos["t_cam"][-1] - t_imu[0]) / 1e9
    for ej, titulo, unidad in [(ejes[1, 0], "Acelerómetro", "m/s²"),
                               (ejes[1, 1], "Giroscopio", "°/s")]:
        ej.axvline(t0, color="k", ls="--", lw=0.8)
        ej.axvline(t1, color="k", ls="--", lw=0.8)
        ej.set_title(f"{titulo} (discontinuas: primera/última imagen)")
        ej.set_xlabel("s desde la primera muestra de IMU")
        ej.set_ylabel(unidad)
        ej.legend()

    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raiz", help="carpeta de la grabación (la que contiene mav0/)")
    parser.add_argument("--config", default=RUTA_CONFIG_DEFECTO)
    parser.add_argument("--sin-figura", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_ver = yaml.safe_load(f)["verificacion"]

    datos = cargar_grabacion(args.raiz)
    f_imu = datos["meta"]["imu"]["frecuencia_hz"]
    f_cam = datos["meta"]["camara"]["fps"]

    # 1-2. Regularidad
    res_imu = analizar_intervalos(datos["t_imu"], f_imu, cfg_ver["factor_hueco"])
    res_cam = analizar_intervalos(datos["t_cam"], f_cam, cfg_ver["factor_hueco"])
    imprimir_intervalos("IMU", res_imu, f_imu)
    imprimir_intervalos("CÁMARA", res_cam, f_cam)

    faltan = [n for n in datos["archivos"]
              if not os.path.exists(os.path.join(datos["carpeta_img"], n))]
    print(f"  PNG en disco:      {len(datos['archivos']) - len(faltan)} de {len(datos['archivos'])}")
    print(f"  descartadas cola:  {datos['meta']['camara'].get('descartadas_por_disco', 'n/d')}")

    # 3. Reloj
    res_reloj = analizar_reloj(datos["t_cam"], datos["t_llegada"], datos["t_imu"],
                               cfg_ver["latencia_max_ms"])
    print("\n=== RELOJ COMÚN ===")
    if res_reloj["latencia_ms"] is not None:
        lat = res_reloj["latencia_ms"]
        print(f"  latencia llegada - sensor: media {lat['media']:.1f} ms "
              f"(mín {lat['min']:.1f}, máx {lat['max']:.1f})")
        print("  OK: misma base de tiempos" if res_reloj["reloj_ok"]
              else "  ¡ATENCIÓN! Latencias fuera de rango: relojes distintos")
    print(f"  IMU empieza {res_reloj['margen_ini_s']:.2f} s antes de la 1ª imagen y "
          f"acaba {res_reloj['margen_fin_s']:.2f} s después de la última")

    # 4. Reposo
    rep = analizar_reposo(datos["t_imu"], datos["w"], datos["a"],
                          cfg_ver["t_reposo_s"], cfg_ver["g"])
    print(f"\n=== REPOSO (primeros {cfg_ver['t_reposo_s']:.1f} s, {rep['n']} muestras) ===")
    am = rep["a_media"]
    print(f"  ā [m/s²]:           x={am[0]:+.4f} y={am[1]:+.4f} z={am[2]:+.4f}")
    print(f"  ‖ā‖:                {rep['norma_a']:.4f} (error {rep['error_norma']:+.4f} m/s²)")
    print(f"  eje hacia arriba:   {rep['eje_arriba']}")
    b = rep["sesgo_gir_dps"]
    s = rep["ruido_gir_dps"]
    sa = rep["ruido_acc"]
    print(f"  sesgo gir [°/s]:    x={b[0]:+.3f} y={b[1]:+.3f} z={b[2]:+.3f}")
    print(f"  σ gir [°/s]:        x={s[0]:.4f} y={s[1]:.4f} z={s[2]:.4f}")
    print(f"  σ acc [m/s²]:       x={sa[0]:.4f} y={sa[1]:.4f} z={sa[2]:.4f}")
    if datos["temp"] is not None:
        print(f"  temperatura:        {datos['temp'][:rep['n']].mean():.1f} °C")
    if s.max() > cfg_ver["ruido_gir_max_dps"]:
        print("  ¡ATENCIÓN! σ del giroscopio alta: ¿estaba de verdad quieto?")

    # 5. Criterio F1
    print("\n=== CRITERIO DE SALIDA F1 ===")
    resultados = evaluar_criterio(res_imu, res_cam, res_reloj, len(faltan), cfg_ver["criterio"])
    for texto, ok in resultados:
        print(f"  [{'OK' if ok else 'NO'}] {texto}")
    print("  >>> F1 SUPERADA" if all(ok for _, ok in resultados) else "  >>> F1 NO superada")

    if not args.sin_figura:
        ruta = os.path.join(args.raiz, "verificacion.png")
        guardar_figura(datos, res_imu, res_cam, ruta)
        print(f"\nFigura: {ruta}")


if __name__ == "__main__":
    main()
