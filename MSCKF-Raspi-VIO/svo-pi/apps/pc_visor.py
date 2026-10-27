#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apps/pc_visor.py — Visor en el PC de la orientación que envía la Raspberry Pi.

Se ejecuta en: [Windows PowerShell]  (o [WSL/Ubuntu] con entorno gráfico)
  python apps\\pc_visor.py
La primera vez, Windows preguntará si permite a Python aceptar conexiones en
redes privadas: hay que aceptar o no llegará ningún paquete.

Paneles:
  arriba-izq.   placa con el sesgo del giroscopio corregido (R_corr)
  arriba-der.   placa SIN corregir el sesgo (R_bruto): deriva a la vista
  abajo-izq.    yaw de ambas en el tiempo: la diferencia crece ≈ b_g,z · t
  abajo-der.    roll/pitch integrados frente a los que da la gravedad
                (la gravedad corrige roll/pitch; el yaw no tiene referencia)

Ejes de la placa: x rojo, y verde, z azul = ejes impresos en el módulo MPU-6050.
Mundo: z hacia arriba, yaw inicial = 0 por convenio.

Prueba sin pantalla (guarda una imagen tras escuchar unos segundos):
  python apps/pc_visor.py --png visor.png --segundos 5
"""

import argparse
import os
import socket
import sys
import threading
import time
from collections import deque

import numpy as np
import yaml

RAIZ_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(RAIZ_REPO, "vio"))

import protocolo          # noqa: E402
from so3 import euler_zyx  # noqa: E402

TIMEOUT_SOCKET_S = 0.2     # para poder comprobar periódicamente si hay que parar


# =====================================================================
# Recepción en segundo plano
# =====================================================================
class Receptor:
    """Hilo que escucha UDP y guarda el último mensaje y la historia de ángulos."""

    def __init__(self, puerto, ventana_s):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", puerto))      # escuchar en todas las interfaces
        self.sock.settimeout(TIMEOUT_SOCKET_S)
        self.ventana_s = ventana_s

        self.cerrojo = threading.Lock()
        self.ultimo = None
        self.t_ultimo_reloj = None    # cuándo llegó (reloj del PC), para detectar silencio
        self.recibidos = 0
        self.perdidos = 0
        self.seq_prev = None
        self.t0_ns = None
        # Historia: t [s], yaw_c, yaw_b, roll_c, pitch_c, roll_acc, pitch_acc
        self.historia = deque()

        self.parar = threading.Event()
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self):
        while not self.parar.is_set():
            try:
                datos, _origen = self.sock.recvfrom(protocolo.TAM_MAX_DATAGRAMA)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                m = protocolo.decodificar(datos)
            except Exception:
                continue          # paquete corrupto o ajeno: ignorar
            self._registrar(m)

    def _registrar(self, m):
        with self.cerrojo:
            # Pérdidas por saltos en el número de secuencia
            if self.seq_prev is not None:
                if m["seq"] <= self.seq_prev:
                    # La Pi se reinició: empezar de cero
                    self.historia.clear()
                    self.t0_ns = None
                    self.perdidos = 0
                else:
                    self.perdidos += m["seq"] - self.seq_prev - 1
            self.seq_prev = m["seq"]
            self.recibidos += 1
            self.ultimo = m
            self.t_ultimo_reloj = time.monotonic()

            if m["fase"] != "integrando":
                return
            if self.t0_ns is None:
                self.t0_ns = m["t_ns"]
            t = (m["t_ns"] - self.t0_ns) / 1e9
            yaw_c, pitch_c, roll_c = euler_zyx(m["R_corr"])
            yaw_b, _, _ = euler_zyx(m["R_bruto"])
            self.historia.append((t, yaw_c, yaw_b, roll_c, pitch_c, m["rp_acc"][0], m["rp_acc"][1]))
            while self.historia and t - self.historia[0][0] > self.ventana_s:
                self.historia.popleft()

    def instantanea(self):
        """Copia coherente del estado para dibujar."""
        with self.cerrojo:
            return (self.ultimo, self.t_ultimo_reloj, self.recibidos, self.perdidos,
                    np.array(self.historia) if self.historia else None)

    def cerrar(self):
        self.parar.set()
        self.hilo.join(timeout=1.0)
        self.sock.close()


# =====================================================================
# Dibujo
# =====================================================================
def vertices_caja(semiejes):
    sx, sy, sz = semiejes
    return np.array([[-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
                     [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]])


CARAS = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
COLORES_EJES = ["tab:red", "tab:green", "tab:blue"]


class Visor:

    def __init__(self, cfg_visor, receptor, plt):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.cfg = cfg_visor
        self.rx = receptor
        self.plt = plt
        self.v_cuerpo = vertices_caja(cfg_visor["semiejes_caja"])
        L = cfg_visor["longitud_ejes"]

        self.fig = plt.figure(figsize=(14, 9))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.3, 1])

        # --- Paneles 3D ---
        self.ax3d = []
        self.cajas = []
        self.ejes = []
        titulos = ["Sesgo corregido (R_corr)", "Sin corregir (R_bruto)"]
        for col in range(2):
            ax = self.fig.add_subplot(gs[0, col], projection="3d")
            ax.set_title(titulos[col])
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            ax.set_zlim(-L, L)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel("x mundo")
            ax.set_ylabel("y mundo")
            ax.set_zlabel("z mundo (arriba)")
            # Ejes del mundo como referencia fija
            for k in range(3):
                d = np.zeros(3)
                d[k] = L
                ax.plot([0, d[0]], [0, d[1]], [0, d[2]], color="0.75", lw=1, ls="--")
            caja = Poly3DCollection([], facecolor="tab:cyan", edgecolor="k", alpha=0.35)
            ax.add_collection3d(caja)
            lineas = [ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=3)[0] for c in COLORES_EJES]
            self.ax3d.append(ax)
            self.cajas.append(caja)
            self.ejes.append(lineas)

        # --- Yaw en el tiempo ---
        self.ax_yaw = self.fig.add_subplot(gs[1, 0])
        self.l_yaw_c, = self.ax_yaw.plot([], [], label="corregido", color="tab:blue")
        self.l_yaw_b, = self.ax_yaw.plot([], [], label="sin corregir", color="tab:red")
        self.ax_yaw.set_xlabel("t [s]")
        self.ax_yaw.set_ylabel("yaw [°]")
        self.ax_yaw.set_title("Yaw: sin referencia externa, la deriva se acumula")
        self.ax_yaw.legend(loc="upper left")
        self.ax_yaw.grid(alpha=0.3)

        # --- Roll/pitch frente a la gravedad ---
        self.ax_rp = self.fig.add_subplot(gs[1, 1])
        self.l_roll, = self.ax_rp.plot([], [], color="tab:blue", label="roll giroscopio")
        self.l_roll_a, = self.ax_rp.plot([], [], color="tab:blue", ls=":", label="roll gravedad")
        self.l_pitch, = self.ax_rp.plot([], [], color="tab:orange", label="pitch giroscopio")
        self.l_pitch_a, = self.ax_rp.plot([], [], color="tab:orange", ls=":", label="pitch gravedad")
        self.ax_rp.set_xlabel("t [s]")
        self.ax_rp.set_ylabel("[°]")
        self.ax_rp.set_title("Roll/pitch: la gravedad da una referencia")
        self.ax_rp.legend(loc="upper left", fontsize=8)
        self.ax_rp.grid(alpha=0.3)

        self.texto = self.fig.suptitle("Esperando datos...", fontsize=12)
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))

    def _dibujar_placa(self, i, R):
        v_mundo = (R @ self.v_cuerpo.T).T          # vértices del cuerpo al mundo
        self.cajas[i].set_verts([[v_mundo[j] for j in cara] for cara in CARAS])
        L = self.cfg["longitud_ejes"]
        for k in range(3):
            e = R[:, k] * L                       # columna k de R_w_i = eje k de la IMU en el mundo
            self.ejes[i][k].set_data_3d([0, e[0]], [0, e[1]], [0, e[2]])

    def actualizar(self, _frame=None):
        m, t_reloj, recibidos, perdidos, hist = self.rx.instantanea()

        if m is None:
            self.texto.set_text("Esperando datos de la Pi... (¿IP/puerto? ¿cortafuegos?)")
            return
        silencio = time.monotonic() - t_reloj > self.cfg["timeout_sin_datos_s"]

        # Cabecera
        if m["fase"] == "inicializando":
            estado = f"INICIALIZANDO {100 * m['progreso']:.0f} % — no muevas la placa"
        else:
            b = np.degrees(m["b_g"])
            estado = f"Integrando | b_g = [{b[0]:+.3f}, {b[1]:+.3f}, {b[2]:+.3f}] °/s"
        if m.get("aviso"):
            estado += f" | aviso: {m['aviso']}"
        if silencio:
            estado = "SIN DATOS — " + estado
        self.texto.set_text(f"{estado}\nrecibidos {recibidos} | perdidos {perdidos}")

        # Placas
        self._dibujar_placa(0, m["R_corr"])
        self._dibujar_placa(1, m["R_bruto"])

        # Series temporales
        if hist is not None and len(hist) > 1:
            t = hist[:, 0]
            grados = lambda x: np.degrees(np.unwrap(x))   # unwrap: sin saltos de ±180°
            self.l_yaw_c.set_data(t, grados(hist[:, 1]))
            self.l_yaw_b.set_data(t, grados(hist[:, 2]))
            self.l_roll.set_data(t, grados(hist[:, 3]))
            self.l_pitch.set_data(t, grados(hist[:, 4]))
            self.l_roll_a.set_data(t, grados(hist[:, 5]))
            self.l_pitch_a.set_data(t, grados(hist[:, 6]))
            for ax in (self.ax_yaw, self.ax_rp):
                ax.relim()
                ax.autoscale_view()


# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Visor de orientación (PC)")
    parser.add_argument("--config", default=os.path.join(RAIZ_REPO, "config", "hito_giro.yaml"))
    parser.add_argument("--puerto", type=int, default=None)
    parser.add_argument("--png", default=None, help="modo sin pantalla: guardar imagen y salir")
    parser.add_argument("--segundos", type=float, default=5.0, help="con --png: tiempo escuchando")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    puerto = args.puerto or cfg["red"]["puerto"]

    import matplotlib
    if args.png:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    receptor = Receptor(puerto, cfg["visor"]["ventana_historia_s"])
    print(f"Escuchando UDP en el puerto {puerto}...")
    visor = Visor(cfg["visor"], receptor, plt)

    try:
        if args.png:
            time.sleep(args.segundos)
            visor.actualizar()
            visor.fig.savefig(args.png, dpi=90)
            _, _, recibidos, perdidos, _ = receptor.instantanea()
            print(f"Imagen guardada en {args.png} | recibidos {recibidos} | perdidos {perdidos}")
        else:
            # La referencia a la animación debe mantenerse viva mientras se muestra
            anim = FuncAnimation(visor.fig, visor.actualizar,
                                 interval=cfg["visor"]["intervalo_refresco_ms"],
                                 cache_frame_data=False)
            plt.show()
            del anim
    finally:
        receptor.cerrar()


if __name__ == "__main__":
    main()
