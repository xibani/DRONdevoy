import cv2
from threading import Thread
import queue


class WriterPi:
    """Escribe a disco en un hilo aparte para no frenar el bucle principal."""

    def __init__(self, fichero, fourcc, fps, tam):
        self.writer = cv2.VideoWriter(fichero, fourcc, fps, tam)
        self.cola = queue.Queue(maxsize=64)
        self.parado = False
        self.hilo = Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self):
        while not self.parado or not self.cola.empty():
            try:
                frame = self.cola.get(timeout=0.1)
            except queue.Empty:
                continue
            self.writer.write(frame)

    def write(self, frame):
        if not self.cola.full():  # si se llena, mejor perder un frame que congelar
            self.cola.put(frame)

    def close(self):
        self.parado = True
        self.hilo.join()
        self.writer.release()
