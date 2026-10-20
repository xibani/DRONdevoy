"""Extrae los fotogramas de una sesión Tello y conserva sus timestamps.

Uso:
    python extraer_frames.py sesiones/20261017_183000
    python extraer_frames.py sesiones/20261017_183000 --formato jpg
"""

import argparse
import csv
from pathlib import Path

import cv2


def extraer(carpeta: Path, formato: str) -> None:
    video = carpeta / "video.mp4"
    origen_csv = carpeta / "frames.csv"
    destino = carpeta / "frames"
    manifiesto = carpeta / "frames_extraidos.csv"

    if not video.is_file():
        raise FileNotFoundError(f"No existe el vídeo: {video}")
    if not origen_csv.is_file():
        raise FileNotFoundError(f"No existe el CSV de timestamps: {origen_csv}")

    with origen_csv.open(newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))

    if not filas:
        raise ValueError(f"El fichero {origen_csv} no contiene frames")

    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        raise RuntimeError(f"No se puede abrir el vídeo: {video}")

    destino.mkdir(parents=True, exist_ok=True)
    inicio = float(filas[0]["t_host"])
    cantidad = 0
    parametros = [cv2.IMWRITE_JPEG_QUALITY, 95] if formato == "jpg" else []

    try:
        with manifiesto.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["idx_frame", "t_host", "t_rel_s", "archivo"])

            while True:
                ok, imagen = captura.read()
                if not ok:
                    break
                if cantidad >= len(filas):
                    print(
                        "AVISO: el vídeo tiene más imágenes que las registradas en frames.csv"
                    )
                    break

                fila = filas[cantidad]
                idx = int(fila["idx_frame"])
                t_host = float(fila["t_host"])
                archivo = destino / f"frame_{idx:06d}.{formato}"

                if not cv2.imwrite(str(archivo), imagen, parametros):
                    raise OSError(f"Error al guardar {archivo}")

                writer.writerow(
                    [
                        idx,
                        fila["t_host"],
                        f"{t_host - inicio:.6f}",
                        f"frames/{archivo.name}",
                    ]
                )
                cantidad += 1
    finally:
        captura.release()

    if cantidad != len(filas):
        print(
            f"AVISO: video: {cantidad} imágenes extraídas; CSV: {len(filas)} registros"
        )
    print(f"Frames extraídos: {cantidad}")
    print(f"Imágenes: {destino}")
    print(f"Timestamps: {manifiesto}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sesion", type=Path, help="Carpeta que contiene video.mp4 y frames.csv"
    )
    parser.add_argument("--formato", choices=("png", "jpg"), default="png")
    args = parser.parse_args()
    extraer(args.sesion, args.formato)
