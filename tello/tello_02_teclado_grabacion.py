# -*- coding: utf-8 -*-
"""
tello_02_teclado_grabacion.py

Control por teclado + grabacion sincronizada (video + estado + comandos)
con djitellopy. Sirve para usar el Tello como capturador de datasets de
interiores y como base del futuro bucle de control autonomo.

Requisitos:
    pip install djitellopy opencv-python

Salida en ./sesiones/AAAAMMDD_HHMMSS/ :
    video.mp4      frames tal como llegan (960x720)
    frames.csv     idx_frame, t_host
    estado.csv     t_host + todos los campos del estado
    comandos.csv   t_host, lr, fb, ud, yaw  (lo que enviamos con rc)

Todas las marcas de tiempo son time.monotonic() del PC. Ni el video ni el
estado traen timestamp del dron, asi que la sincronizacion es la de llegada
(incluye la latencia de red y de decodificacion, que es mayor en el video).

Teclas (con la ventana de OpenCV enfocada):
    t        despegar               l      aterrizar
    ESPACIO  EMERGENCIA (corta motores, el dron cae)
    w / s    adelante / atras       a / d  izquierda / derecha
    r / f    subir / bajar          q / e  girar izquierda / derecha
    g        grabar on/off          ESC    salir (aterriza si vuela)

Limitacion: cv2.waitKey no da eventos de soltar tecla. Cada pulsacion mantiene
el comando DURACION_PULSO_S segundos; al mantener pulsada, la autorrepeticion
del sistema operativo lo prolonga. Para control fino, usa pygame o un mando.
"""

import csv
import os
import threading
import time
from datetime import datetime

import cv2
from djitellopy import Tello

# ----------------------------------------------------------------------
# Parametros
# ----------------------------------------------------------------------
VELOCIDAD_RC = 40  # magnitud de los comandos rc (-100..100)
DURACION_PULSO_S = 0.30  # cuanto dura un comando tras pulsar la tecla
PERIODO_RC_S = 0.05  # enviar rc a 20 Hz
BATERIA_MINIMA = 15  # % por debajo del cual aterriza solo
FPS_ESCRITURA = 30  # fps nominal del mp4 (el real esta en frames.csv)

# En djitellopy 2.5 los frames llegan en RGB. Si ves los colores
# intercambiados (caras azules), pon esto a False.
FRAME_ES_RGB = True

# ----------------------------------------------------------------------
# Paso 1: carpeta de la sesion
# ----------------------------------------------------------------------
nombre_sesion = datetime.now().strftime("%Y%m%d_%H%M%S")
carpeta = os.path.join("sesiones", nombre_sesion)
os.makedirs(carpeta, exist_ok=True)

# ----------------------------------------------------------------------
# Paso 2: conexion
# ----------------------------------------------------------------------
tello = Tello()
tello.connect()  # envia "command" y espera "ok"
print(f"Bateria: {tello.get_battery()} %")

tello.streamon()
lector_frames = tello.get_frame_read()  # hilo interno que decodifica video

# Esperar al primer frame (puede tardar unos segundos hasta el keyframe)
print("Esperando video...")
t_espera = time.monotonic()
while lector_frames.frame is None and time.monotonic() - t_espera < 10.0:
    time.sleep(0.05)
if lector_frames.frame is None:
    print("No llega video. Revisa el firewall (UDP 11111 entrante).")

# ----------------------------------------------------------------------
# Paso 3: grabacion del estado en un hilo propio
# ----------------------------------------------------------------------
parar = threading.Event()
grabando = threading.Event()  # compartido con el hilo de estado


def hilo_grabar_estado():
    """
    Sondea el estado de djitellopy cada 5 ms y escribe una fila cada vez
    que llega un paquete nuevo. Un paquete se considera nuevo si cambia
    el objeto diccionario o su contenido.
    """
    ruta = os.path.join(carpeta, "estado.csv")
    fichero = None
    escritor = None
    ref_anterior = None
    copia_anterior = None

    while not parar.is_set():
        estado = tello.get_current_state()
        if estado and (estado is not ref_anterior or estado != copia_anterior):
            ref_anterior = estado
            copia_anterior = dict(estado)

            if grabando.is_set():
                if escritor is None:
                    # La cabecera se fija con el primer paquete grabado
                    fichero = open(ruta, "w", newline="")
                    campos = ["t_host"] + list(estado.keys())
                    escritor = csv.DictWriter(
                        fichero, fieldnames=campos, extrasaction="ignore"
                    )
                    escritor.writeheader()
                fila = dict(estado)
                fila["t_host"] = time.monotonic()
                escritor.writerow(fila)
        time.sleep(0.005)

    if fichero is not None:
        fichero.close()


t_estado = threading.Thread(target=hilo_grabar_estado, daemon=True)
t_estado.start()

# ----------------------------------------------------------------------
# Paso 4: ficheros de frames y comandos (se abren al empezar a grabar)
# ----------------------------------------------------------------------
escritor_video = None
fichero_frames = None
escritor_frames = None
fichero_cmd = None
escritor_cmd = None
idx_frame = 0


def abrir_grabacion(ancho, alto):
    global escritor_video, fichero_frames, escritor_frames
    global fichero_cmd, escritor_cmd
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    escritor_video = cv2.VideoWriter(
        os.path.join(carpeta, "video.mp4"), fourcc, FPS_ESCRITURA, (ancho, alto)
    )
    fichero_frames = open(os.path.join(carpeta, "frames.csv"), "w", newline="")
    escritor_frames = csv.writer(fichero_frames)
    escritor_frames.writerow(["idx_frame", "t_host"])
    fichero_cmd = open(os.path.join(carpeta, "comandos.csv"), "w", newline="")
    escritor_cmd = csv.writer(fichero_cmd)
    escritor_cmd.writerow(["t_host", "lr", "fb", "ud", "yaw"])


def cerrar_grabacion():
    if escritor_video is not None:
        escritor_video.release()
    if fichero_frames is not None:
        fichero_frames.close()
    if fichero_cmd is not None:
        fichero_cmd.close()


# ----------------------------------------------------------------------
# Paso 5: bucle principal
# ----------------------------------------------------------------------
volando = False
comando = [0, 0, 0, 0]  # lr, fb, ud, yaw
t_fin_pulso = 0.0
t_ultimo_rc = 0.0
frame_anterior = None

try:
    while True:
        ahora = time.monotonic()

        # --- 5.1 Video: detectar frame nuevo por identidad del array ---
        frame = lector_frames.frame
        if frame is not None and frame is not frame_anterior:
            frame_anterior = frame
            if FRAME_ES_RGB:
                imagen_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                imagen_bgr = frame

            if grabando.is_set():
                if escritor_video is None:
                    alto, ancho = imagen_bgr.shape[:2]
                    abrir_grabacion(ancho, alto)
                escritor_video.write(imagen_bgr)
                escritor_frames.writerow([idx_frame, ahora])
                idx_frame += 1

            # Superponer informacion solo en la copia que se muestra
            vista = imagen_bgr.copy()
            est = tello.get_current_state()
            texto = (
                f"bat {est.get('bat', '?')}%  h {est.get('h', '?')}  "
                f"tof {est.get('tof', '?')}  yaw {est.get('yaw', '?')}"
            )
            cv2.putText(
                vista, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            if grabando.is_set():
                cv2.circle(vista, (vista.shape[1] - 30, 30), 12, (0, 0, 255), -1)
            cv2.imshow("Tello", vista)

        # --- 5.2 Teclado ---
        tecla = cv2.waitKey(1) & 0xFF
        nuevo = None
        if tecla == 27:  # ESC
            break
        elif tecla == ord(" "):
            # Emergencia: corta motores inmediatamente
            try:
                tello.emergency()
            except Exception as e:
                print(f"emergency: {e}")
            volando = False
        elif tecla == ord("t") and not volando:
            tello.takeoff()  # bloquea hasta "ok"
            volando = True
        elif tecla == ord("l") and volando:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            volando = False
        elif tecla == ord("g"):
            if grabando.is_set():
                grabando.clear()
                print("Grabacion PAUSADA")
            else:
                grabando.set()
                print(f"Grabando en {carpeta}")
        elif tecla == ord("w"):
            nuevo = [0, VELOCIDAD_RC, 0, 0]
        elif tecla == ord("s"):
            nuevo = [0, -VELOCIDAD_RC, 0, 0]
        elif tecla == ord("a"):
            nuevo = [-VELOCIDAD_RC, 0, 0, 0]
        elif tecla == ord("d"):
            nuevo = [VELOCIDAD_RC, 0, 0, 0]
        elif tecla == ord("r"):
            nuevo = [0, 0, VELOCIDAD_RC, 0]
        elif tecla == ord("f"):
            nuevo = [0, 0, -VELOCIDAD_RC, 0]
        elif tecla == ord("q"):
            nuevo = [0, 0, 0, -VELOCIDAD_RC]
        elif tecla == ord("e"):
            nuevo = [0, 0, 0, VELOCIDAD_RC]

        if nuevo is not None:
            comando = nuevo
            t_fin_pulso = ahora + DURACION_PULSO_S
        elif ahora > t_fin_pulso:
            comando = [0, 0, 0, 0]  # sin tecla: hover

        # --- 5.3 Envio de rc a frecuencia fija (tambien hace de keep-alive) ---
        if volando and ahora - t_ultimo_rc >= PERIODO_RC_S:
            tello.send_rc_control(*comando)
            t_ultimo_rc = ahora
            if grabando.is_set() and escritor_cmd is not None:
                escritor_cmd.writerow([ahora] + comando)

        # --- 5.4 Seguridad por bateria ---
        bat = tello.get_current_state().get("bat")
        if volando and bat is not None and float(bat) < BATERIA_MINIMA:
            print("Bateria baja: aterrizando")
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
            volando = False

finally:
    # ------------------------------------------------------------------
    # Paso 6: cierre ordenado
    # ------------------------------------------------------------------
    if volando:
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except Exception as e:
            print(f"land: {e}")
    parar.set()
    time.sleep(0.2)
    cerrar_grabacion()
    try:
        tello.streamoff()
    except Exception:
        pass
    tello.end()
    cv2.destroyAllWindows()
    print(f"Sesion guardada en {carpeta} ({idx_frame} frames)")
