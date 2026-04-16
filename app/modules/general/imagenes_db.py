"""Mueve imágenes que tienen un archivo .txt asociado a otra carpeta.

Este módulo se utiliza para construir conjuntos de datos YOLO a partir de
carpetas que contienen pares imagen/etiqueta.
"""

import argparse
import os
import shutil

from app.core.task import Task


def mover_imagenes_con_txt(source_folder, destination_folder):
    """Copia imágenes con TXT asociado a la carpeta de destino.

    Parameters
    ----------
    source_folder : str
        Carpeta de origen que contiene imágenes y archivos .txt.
    destination_folder : str
        Carpeta de destino donde se copiarán las imágenes válidas.
    """
    os.makedirs(destination_folder, exist_ok=True)
    archivos = os.listdir(source_folder)
    archivos_txt = {os.path.splitext(f)[0] for f in archivos if f.lower().endswith('.txt')}
    extensiones_imagen = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    for archivo in sorted(archivos):
        nombre, extension = os.path.splitext(archivo)
        if extension.lower() in extensiones_imagen and nombre in archivos_txt:
            origen = os.path.join(source_folder, archivo)
            destino = os.path.join(destination_folder, archivo)
            shutil.copy2(origen, destino)
            print(f"Movido: {archivo}")


class ImagenesDbTask(Task):
    """Tarea para mover imágenes que tienen etiquetas TXT asociadas."""

    name = "imagenes_db"

    def __init__(self, params):
        """Inicializa la tarea con sus parámetros.

        Parameters
        ----------
        params : dict
            Debe contener 'source_folder' y 'destination_folder'.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        """Ejecuta la copia de imágenes pareadas hacia la carpeta destino."""
        mover_imagenes_con_txt(
            self.params.get("source_folder"),
            self.params.get("destination_folder"),
        )


def main():
    parser = argparse.ArgumentParser(description="Mueve imágenes con TXT asociado")
    parser.add_argument("--source", required=True, help="Carpeta de origen")
    parser.add_argument("--dest", required=True, help="Carpeta de destino")
    args = parser.parse_args()

    task = ImagenesDbTask(
        {"source_folder": args.source, "destination_folder": args.dest}
    )
    task.run()


if __name__ == "__main__":
    main()

