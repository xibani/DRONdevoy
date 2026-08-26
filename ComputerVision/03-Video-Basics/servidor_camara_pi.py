#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
SERVIDOR MJPEG PARA RASPBERRY PI  ->  se ejecuta EN LA RASPBERRY
================================================================================

Convierte la Raspberry Pi + Camera Module en una "webcam de red".
Publica el video en:   http://<IP_DE_LA_PI>:8000/stream.mjpg

Desde el PC, en OpenCV, basta con hacer:
    cap = cv2.VideoCapture("http://192.168.1.50:8000/stream.mjpg")

REQUISITOS EN LA PI (Raspberry Pi OS Bookworm o posterior):
    sudo apt update
    sudo apt install -y python3-picamera2
    # comprobar que la camara se detecta:
    rpicam-hello --list-cameras

USO:
    python3 servidor_camara_pi.py
    (Ctrl+C para parar)
================================================================================
"""

import io
import socket
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder, Quality
from picamera2.outputs import FileOutput

# ------------------------------------------------------------------------------
# 1) PARAMETROS DE CONFIGURACION
# ------------------------------------------------------------------------------
ANCHO = 640  # resolucion baja = menos latencia. Sube a 1280x720 si va fino.
ALTO = 480
FPS = 30
PUERTO = 8000

# ------------------------------------------------------------------------------
# 2) PAGINA HTML DE PRUEBA (para verificar desde el navegador)
# ------------------------------------------------------------------------------
PAGINA_HTML = f"""<!DOCTYPE html>
<html>
<head><title>Camara Raspberry Pi</title></head>
<body style="background:#111; color:#eee; font-family:sans-serif; text-align:center">
  <h2>Stream MJPEG de la Raspberry Pi</h2>
  <img src="stream.mjpg" width="{ANCHO}" height="{ALTO}" />
  <p>URL para OpenCV: <code>/stream.mjpg</code></p>
</body>
</html>
"""


# ------------------------------------------------------------------------------
# 3) BUFFER COMPARTIDO: guarda siempre el ULTIMO frame JPEG generado
# ------------------------------------------------------------------------------
class SalidaStreaming(io.BufferedIOBase):
    """El encoder MJPEG escribe aqui cada frame comprimido en JPEG."""

    def __init__(self):
        self.frame = None
        self.condicion = Condition()  # para avisar a los clientes de frame nuevo

    def write(self, buf):
        with self.condicion:
            self.frame = buf
            self.condicion.notify_all()


# ------------------------------------------------------------------------------
# 4) SERVIDOR HTTP: atiende las peticiones de los clientes (PC, navegador...)
# ------------------------------------------------------------------------------
class ManejadorStreaming(server.BaseHTTPRequestHandler):

    def do_GET(self):

        # --- Redireccion de la raiz a la pagina de prueba ---
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()

        # --- Pagina HTML de prueba ---
        elif self.path == "/index.html":
            contenido = PAGINA_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(contenido))
            self.end_headers()
            self.wfile.write(contenido)

        # --- EL STREAM: esto es lo que consume OpenCV ---
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", 0)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()
            try:
                while True:
                    # Esperamos a que haya un frame nuevo
                    with salida.condicion:
                        salida.condicion.wait()
                        frame = salida.frame

                    # Enviamos el frame como una "parte" del multipart
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception as e:
                print(f"[INFO] Cliente {self.client_address} desconectado: {e}")

        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, formato, *args):
        pass  # silenciamos el log por peticion para no ensuciar la consola


class ServidorStreaming(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# ------------------------------------------------------------------------------
# 5) UTILIDAD: averiguar la IP local de la Pi (para mostrarla por pantalla)
# ------------------------------------------------------------------------------
def obtener_ip_local():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no envia nada, solo elige la interfaz
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# ------------------------------------------------------------------------------
# 6) PROGRAMA PRINCIPAL
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    print("[1/3] Inicializando la camara...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (ANCHO, ALTO)},
        controls={"FrameRate": FPS},
    )
    picam2.configure(config)

    print("[2/3] Arrancando el encoder MJPEG...")
    salida = SalidaStreaming()
    picam2.start_recording(MJPEGEncoder(), FileOutput(salida), quality=Quality.MEDIUM)

    ip = obtener_ip_local()
    print("[3/3] Servidor en marcha.\n")
    print("=" * 70)
    print(f"  Navegador (prueba):  http://{ip}:{PUERTO}/")
    print(f"  OpenCV (URL):        http://{ip}:{PUERTO}/stream.mjpg")
    print("=" * 70)
    print("\nCtrl+C para parar.\n")

    try:
        direccion = ("", PUERTO)
        servidor = ServidorStreaming(direccion, ManejadorStreaming)
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Parando...")
    finally:
        picam2.stop_recording()
        print("[INFO] Camara liberada. Adios.")
