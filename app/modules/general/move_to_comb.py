"""Copia imágenes y etiquetas TXT pareadas a una carpeta combinada.

Esta utilidad construye un directorio 'comb' con imágenes y sus etiquetas
asociadas, omitiendo etiquetas vacías o pares incompletos.
"""

import argparse
import os
import shutil

from app.core.task import Task
from tqdm import tqdm


def mover_imagenes_y_etiquetas(image_folder, label_folder, destination_folder):
    """Copia imágenes y sus etiquetas asociadas a la carpeta destino.

    Parameters
    ----------
    image_folder : str
        Carpeta con imágenes de entrada.
    label_folder : str
        Carpeta con etiquetas TXT de entrada.
    destination_folder : str
        Carpeta destino donde se copiarán las imágenes y etiquetas.
    """
    os.makedirs(destination_folder, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png"}
    copied = 0
    empty_labels = 0
    missing_images = 0

    available_images = {
        os.path.splitext(f)[0]: os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    }

    labels = sorted(os.listdir(label_folder))
    for label_filename in tqdm(labels, desc="Copiando archivos"):
        if not label_filename.lower().endswith(".txt"):
            continue

        label_path = os.path.join(label_folder, label_filename)
        if os.path.getsize(label_path) == 0:
            print(f"⚠️ Etiqueta vacía, omitida: {label_filename}")
            empty_labels += 1
            continue

        base_name, _ = os.path.splitext(label_filename)
        image_path = available_images.get(base_name)

        if image_path:
            shutil.copy(image_path, os.path.join(destination_folder, os.path.basename(image_path)))
            shutil.copy(label_path, os.path.join(destination_folder, label_filename))
            copied += 1
        else:
            print(f"❌ Sin imagen para: {label_filename}")
            missing_images += 1

    print(f"\nResumen: {copied} copiados | {empty_labels} vacíos omitidos | {missing_images} sin imagen")


class MoveToCombTask(Task):
    """Tarea para copiar imágenes y etiquetas emparejadas a una carpeta combinada.

Example YAML:
```yaml
- name: move_to_comb
  params:
    image_folder: <value>
    label_folder: <value>
    destination_folder: <value>
```

Example YAML:
```yaml
- name: move_to_comb
  params:
    image_folder: <value>
    label_folder: <value>
    destination_folder: <value>
```

Example YAML:
```yaml
- name: move_to_comb
  params:
    image_folder: <value>
    label_folder: <value>
    destination_folder: <value>
```"""

    name = "move_to_comb"

    def __init__(self, params):
        """Inicializa la tarea.

        Parameters
        ----------
        params : dict
            Debe incluir 'image_folder', 'label_folder' y 'destination_folder'.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        """Ejecuta la copia de imágenes y etiquetas a la carpeta destino."""
        mover_imagenes_y_etiquetas(
            self.params.get("image_folder"),
            self.params.get("label_folder"),
            self.params.get("destination_folder"),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Copia imágenes y etiquetas TXT pareadas a una carpeta destino"
    )
    parser.add_argument("--images", required=True, help="Carpeta con las imágenes de entrada")
    parser.add_argument("--labels", required=True, help="Carpeta con las etiquetas TXT de entrada")
    parser.add_argument("--output", required=True, help="Carpeta destino (comb)")
    args = parser.parse_args()

    task = MoveToCombTask(
        {
            "image_folder": args.images,
            "label_folder": args.labels,
            "destination_folder": args.output,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
