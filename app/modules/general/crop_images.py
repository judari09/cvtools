"""Recorta un conjunto de imágenes en una región fija definida por coordenadas.

Este módulo transforma imágenes de una carpeta de entrada en una carpeta de salida
aplicando un recorte rectangular con coordenadas definidas.
"""

import argparse
import os

import cv2

from app.core.task import Task


class CropImagesTask(Task):
    """Tarea para recortar imágenes en un área fija.

Example YAML:
```yaml
- name: crop_images
  params:
    input_folder: <value>
    output_folder: <value>
    x1: <value>
    y1: <value>
    x2: <value>
    y2: <value>
```

Example YAML:
```yaml
- name: crop_images
  params:
    input_folder: <value>
    output_folder: <value>
    x1: <value>
    y1: <value>
    x2: <value>
    y2: <value>
```

Example YAML:
```yaml
- name: crop_images
  params:
    input_folder: <value>
    output_folder: <value>
    x1: <value>
    y1: <value>
    x2: <value>
    y2: <value>
```"""

    name = "crop_images"

    def __init__(self, params):
        """Inicializa la tarea.

        Parameters
        ----------
        params : dict
            Debe incluir 'input_folder', 'output_folder', 'x1', 'y1', 'x2', 'y2'.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        """Ejecuta el recorte de imágenes usando los parámetros configurados."""
        input_folder = self.params.get("input_folder")
        output_folder = self.params.get("output_folder")
        x1 = int(self.params.get("x1", 0))
        y1 = int(self.params.get("y1", 0))
        x2 = int(self.params.get("x2", 0))
        y2 = int(self.params.get("y2", 0))

        if not input_folder or not output_folder:
            raise ValueError("Se requieren 'input_folder' y 'output_folder'")

        os.makedirs(output_folder, exist_ok=True)
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        for filename in sorted(os.listdir(input_folder)):
            if not filename.lower().endswith(valid_extensions):
                continue

            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"No se pudo leer {img_path}")
                continue

            height, width = img.shape[:2]
            if x2 - x1 > width or y2 - y1 > height:
                print(f"La imagen {filename} es demasiado pequeña para la región definida.")
                continue

            cropped_img = img[y1:y2, x1:x2]
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_img)
            print(f"Imagen recortada guardada en {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Recorta imágenes en una región fija")
    parser.add_argument("--input", required=True, help="Carpeta de imágenes de entrada")
    parser.add_argument("--output", required=True, help="Carpeta de salida para las imágenes recortadas")
    parser.add_argument("--x1", type=int, required=True, help="Coordenada x inicial")
    parser.add_argument("--y1", type=int, required=True, help="Coordenada y inicial")
    parser.add_argument("--x2", type=int, required=True, help="Coordenada x final")
    parser.add_argument("--y2", type=int, required=True, help="Coordenada y final")
    args = parser.parse_args()

    task = CropImagesTask(
        {
            "input_folder": args.input,
            "output_folder": args.output,
            "x1": args.x1,
            "y1": args.y1,
            "x2": args.x2,
            "y2": args.y2,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
