"""Verifica la consistencia de tamaños de las imágenes en una carpeta."""

import argparse
import os

import cv2

from app.core.task import Task


class CheckSizesTask(Task):
    """Tarea para comprobar el tamaño de imágenes en un directorio."""

    name = "check_sizes"

    def __init__(self, params):
        """Inicializa la tarea con los parámetros de configuración.

        Parameters
        ----------
        params : dict
            Debe contener 'folder_path'.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def verify_sizes(self, folder_path, extensions=None):
        """Verifica que todas las imágenes tengan la misma resolución."""
        if extensions is None:
            extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        image_files = [
            f for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(extensions)
        ]

        if not image_files:
            print("No se encontraron imágenes en la carpeta.")
            return

        reference_size = None
        inconsistencies = False

        for filename in image_files:
            path = os.path.join(folder_path, filename)
            image = cv2.imread(path)
            if image is None:
                print(f"Error al cargar la imagen: {filename}")
                continue

            h, w = image.shape[:2]
            size = (w, h)

            if reference_size is None:
                reference_size = size
                print(f"Tamaño de referencia establecido en: {reference_size}")
            elif size != reference_size:
                print(
                    f"La imagen '{filename}' tiene un tamaño diferente: {size} "
                    f"(se esperaba {reference_size})"
                )
                inconsistencies = True

        if not inconsistencies:
            print(f"Todas las imágenes tienen el mismo tamaño: {reference_size}")
        else:
            print("Se encontraron inconsistencias en el tamaño de las imágenes.")

    def run(self):
        """Ejecuta la verificación de tamaños del dataset."""
        self.verify_sizes(
            folder_path=self.params.get("folder_path"),
            extensions=self.params.get("extensions"),
        )


def main():
    parser = argparse.ArgumentParser(description="Verifica tamaños de imágenes en una carpeta")
    parser.add_argument("--folder", required=True, help="Carpeta con imágenes")
    args = parser.parse_args()

    task = CheckSizesTask({"folder_path": args.folder})
    task.run()


if __name__ == "__main__":
    main()
