# -*- coding: utf-8 -*-
"""
tello_01_udp_crudo.py

Conexion con el Tello SIN librerias de alto nivel: sockets UDP a mano.
Objetivo: entender el protocolo del SDK antes de usar djitellopy.

Protocolo (SDK oficial, Tello en modo punto de acceso):
    - El Tello crea una WiFi "TELLO-XXXXXX" y tiene IP 192.168.10.1
    - Comandos  : texto por UDP al puerto 8889 del Tello. Responde "ok"/"error"
                  o el valor pedido (p.ej. "battery?" -> "87")
    - Estado    : el Tello envia texto por UDP al puerto 8890 del PC
                  "pitch:0;roll:0;yaw:0;vgx:0;...;agz:-998.00;"
    - Video     : tras "streamon", H.264 crudo por UDP al puerto 11111 del PC

Requisitos:
    pip install av opencv-python

Antes de ejecutar:
    1. Enciende el Tello y conecta el PC a su WiFi (TELLO-XXXXXX).
       Perderas internet en ese adaptador; es normal.
    2. En Windows, permite a Python trafico UDP entrante en el firewall.
       Si no, los comandos responden pero NO llegan ni estado ni video.

Por defecto NO despega (VOLAR = False).
"""

import socket
import threading
import time

import av
import cv2

# ----------------------------------------------------------------------
# Parametros
# ----------------------------------------------------------------------
IP_TELLO = "192.168.10.1"
PUERTO_COMANDOS = 8889  # comandos y respuestas
PUERTO_ESTADO = 8890  # el Tello envia aqui su estado (texto)
PUERTO_VIDEO = 11111  # el Tello envia aqui el flujo H.264

VOLAR = False  # True para probar despegue, hover y aterrizaje
DURACION_PRUEBA_S = 20.0  # segundos que dura la prueba
PERIODO_KEEPALIVE_S = 5.0  # el Tello aterriza solo si pasan 15 s sin comandos

# ----------------------------------------------------------------------
# Variables compartidas entre hilos
# ----------------------------------------------------------------------
parar = threading.Event()
estado_actual = {}
contador_paquetes_estado = 0
ultimo_frame = None
contador_frames = 0

# ----------------------------------------------------------------------
# Paso 1: socket de comandos
# ----------------------------------------------------------------------
sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_cmd.bind(("", PUERTO_COMANDOS))
sock_cmd.settimeout(7.0)


def enviar_comando(texto):
    """Envia un comando de texto y espera la respuesta (bloqueante)."""
    print(f"> {texto}")
    sock_cmd.sendto(texto.encode("utf-8"), (IP_TELLO, PUERTO_COMANDOS))
    try:
        datos, _ = sock_cmd.recvfrom(1024)
        respuesta = datos.decode("utf-8", errors="ignore").strip()
    except socket.timeout:
        respuesta = "TIMEOUT"
    print(f"< {respuesta}")
    return respuesta


# ----------------------------------------------------------------------
# Paso 2: recepcion y parseo del estado
# ----------------------------------------------------------------------
def parsear_estado(texto):
    """
    Convierte "pitch:0;roll:1;...;agz:-998.00;" en un diccionario.
    Campos tipicos del Tello normal (unidades segun SDK, verificalas):
        pitch, roll, yaw : grados (enteros)
        vgx, vgy, vgz    : velocidad (el SDK dice cm/s; en la practica
                           parece dm/s, compruebalo con un vuelo conocido)
        templ, temph     : temperatura min/max (C)
        tof              : distancia ToF inferior (cm)
        h                : altura relativa al despegue (cm)
        bat              : bateria (%)
        baro             : altura barometrica (m)
        time             : tiempo de motores encendidos (s)
        agx, agy, agz    : aceleracion en milesimas de g (agz ~ -1000 en reposo)
    El Tello EDU anade mid, x, y, z, mpry (mission pads).
    """
    estado = {}
    for campo in texto.strip().split(";"):
        if ":" not in campo:
            continue
        clave, valor = campo.split(":", 1)
        try:
            estado[clave] = float(valor)
        except ValueError:
            # p.ej. "mpry:0,0,0" no es un float; se guarda como texto
            estado[clave] = valor
    return estado


def hilo_estado():
    global estado_actual, contador_paquetes_estado
    sock_est = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_est.bind(("", PUERTO_ESTADO))
    sock_est.settimeout(1.0)
    while not parar.is_set():
        try:
            datos, _ = sock_est.recvfrom(1024)
        except socket.timeout:
            continue
        nuevo = parsear_estado(datos.decode("utf-8", errors="ignore"))
        # Marca de tiempo del PC: el estado NO trae timestamp propio
        nuevo["t_host"] = time.monotonic()
        estado_actual = nuevo
        contador_paquetes_estado += 1
    sock_est.close()


# ----------------------------------------------------------------------
# Paso 3: recepcion y decodificacion del video
# ----------------------------------------------------------------------
def hilo_video():
    global ultimo_frame, contador_frames
    try:
        # PyAV abre el flujo UDP y decodifica el H.264.
        # timeout = (apertura, lectura) en segundos
        contenedor = av.open(f"udp://@0.0.0.0:{PUERTO_VIDEO}", timeout=(5.0, 2.0))
    except Exception as e:
        print(f"[video] No se pudo abrir el flujo: {e}")
        return

    try:
        for frame in contenedor.decode(video=0):
            if parar.is_set():
                break
            # PyAV entrega frames YUV; los pasamos a BGR para OpenCV
            ultimo_frame = frame.to_ndarray(format="bgr24")
            contador_frames += 1
    except Exception as e:
        # Al principio suelen llegar errores de decodificacion hasta el
        # primer keyframe; si el hilo muere aqui, revisa firewall/WiFi
        print(f"[video] Flujo interrumpido: {e}")
    finally:
        contenedor.close()


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    # Entrar en modo SDK: sin esto el Tello ignora el resto de comandos
    if enviar_comando("command") != "ok":
        print("El Tello no responde. Revisa la WiFi y el firewall.")
        return

    # Consultas basicas
    enviar_comando("battery?")
    enviar_comando("sn?")
    enviar_comando("sdk?")  # en el Tello normal puede responder error

    # Arrancar recepcion de estado
    t_est = threading.Thread(target=hilo_estado, daemon=True)
    t_est.start()

    # Arrancar video
    enviar_comando("streamon")
    t_vid = threading.Thread(target=hilo_video, daemon=True)
    t_vid.start()

    # Vuelo opcional
    volando = False
    if VOLAR:
        if enviar_comando("takeoff") == "ok":
            volando = True

    t_inicio = time.monotonic()
    t_ultimo_keepalive = t_inicio

    try:
        while time.monotonic() - t_inicio < DURACION_PRUEBA_S:
            # Mostrar el ultimo frame con algunos datos superpuestos
            if ultimo_frame is not None:
                imagen = ultimo_frame.copy()
                est = estado_actual
                texto = (
                    f"bat {est.get('bat', '?')}%  h {est.get('h', '?')} cm  "
                    f"tof {est.get('tof', '?')} cm  yaw {est.get('yaw', '?')}"
                )
                cv2.putText(
                    imagen,
                    texto,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Tello - UDP crudo", imagen)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break

            # Keep-alive: cualquier comando sirve
            ahora = time.monotonic()
            if ahora - t_ultimo_keepalive > PERIODO_KEEPALIVE_S:
                enviar_comando("battery?")
                t_ultimo_keepalive = ahora
    finally:
        if volando:
            enviar_comando("land")
        enviar_comando("streamoff")
        parar.set()
        time.sleep(1.0)
        cv2.destroyAllWindows()
        sock_cmd.close()

        # Frecuencias reales medidas (dato util para tu estimador)
        duracion = time.monotonic() - t_inicio
        print(f"\nDuracion: {duracion:.1f} s")
        print(
            f"Paquetes de estado: {contador_paquetes_estado} "
            f"-> {contador_paquetes_estado / duracion:.1f} Hz"
        )
        print(
            f"Frames de video:    {contador_frames} "
            f"-> {contador_frames / duracion:.1f} fps"
        )


if __name__ == "__main__":
    main()
