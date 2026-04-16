"""Analiza y opcionalmente redimensiona imágenes dentro de una carpeta."""

import argparse
import os

import cv2

from app.core.task import Task


def analyze_image_sizes(folder_path, extensions=None):
    """Devuelve las resoluciones únicas encontradas en la carpeta.

    Parameters
    ----------
    folder_path : str
        Carpeta con las imágenes.
    extensions : tuple[str, ...], optional
        Extensiones de imagen válidas.

    Returns
    -------
    set[str]
        Resoluciones únicas en formato 'altoXancho'.
    """
    if extensions is None:
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    resolutions = set()
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(extensions):
            continue

        input_path = os.path.join(folder_path, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {filename}")
            continue

        height, width = image.shape[:2]
        resolutions.add(f"{height}X{width}")

    return resolutions


def resize_images(source_folder, destination_folder, target_width, target_height, extensions=None):
    """Redimensiona imágenes a la resolución objetivo y las guarda en el destino."""
    if extensions is None:
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    os.makedirs(destination_folder, exist_ok=True)
    for filename in sorted(os.listdir(source_folder)):
        if not filename.lower().endswith(extensions):
            continue

        input_path = os.path.join(source_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"No se pudo cargar la imagen: {filename}")
            continue

        resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        output_path = os.path.join(destination_folder, filename)
        cv2.imwrite(output_path, resized_image)
        print(f"Guardada redimensionada: {output_path}")


class ResizeFolderTask(Task):
    """Tarea para analizar y redimensionar imágenes en una carpeta."""

    name = "resize_folder"

    def __init__(self, params):
        """Inicializa la tarea.

        Parameters
        ----------
        params : dict
            Debe contener 'source_folder'. Si 'do_resize' es True, también se
            requieren 'destination_folder', 'width' y 'height'.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        """Ejecuta el análisis o redimensionado según los parámetros."""
        source_folder = self.params.get("source_folder")
        if not source_folder:
            raise ValueError("Se requiere 'source_folder'")

        do_resize = bool(self.params.get("do_resize", False))

        if do_resize:
            destination_folder = self.params.get("destination_folder")
            width = int(self.params.get("width", 0))
            height = int(self.params.get("height", 0))
            if not destination_folder or width <= 0 or height <= 0:
                raise ValueError("Para redimensionar se requiere 'destination_folder', 'width' y 'height'")
            resize_images(source_folder, destination_folder, width, height)
        else:
            sizes = analyze_image_sizes(source_folder)
            print(f"Resoluciones encontradas: {sorted(sizes)}")


def main():
    parser = argparse.ArgumentParser(description="Analiza y redimensiona imágenes en una carpeta")
    parser.add_argument("--source", required=True, help="Carpeta de imágenes de entrada")
    parser.add_argument("--resize", action="store_true", help="Realizar redimensionado")
    parser.add_argument("--output", help="Carpeta de salida cuando se redimensiona")
    parser.add_argument("--width", type=int, help="Ancho objetivo")
    parser.add_argument("--height", type=int, help="Alto objetivo")
    args = parser.parse_args()

    params = {"source_folder": args.source, "do_resize": args.resize}
    if args.resize:
        params.update({"destination_folder": args.output, "width": args.width, "height": args.height})

    task = ResizeFolderTask(params)
    task.run()


if __name__ == "__main__":
    main()