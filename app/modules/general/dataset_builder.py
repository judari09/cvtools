"""Construye un dataset emparejando imágenes y etiquetas TXT en una carpeta destino.

Consolida la funcionalidad de imagenes_db.py y move_to_comb.py en un único módulo
parametrizable: con filter_empty=False copia solo imágenes que tienen etiqueta;
con filter_empty=True omite además las etiquetas vacías.
"""

import argparse
import os
import shutil

from tqdm import tqdm
from app.core.task import Task
from app.utils.file_utils import pair_images_labels, filter_empty_labels


def build_dataset(source_images: str, source_labels: str, destination: str, filter_empty: bool = True):
    """Copia pares imagen+etiqueta a la carpeta destino.

    Parameters
    ----------
    source_images : str
        Carpeta con las imágenes de entrada.
    source_labels : str
        Carpeta con los archivos .txt de etiquetas.
    destination : str
        Carpeta de destino.
    filter_empty : bool, optional
        Si True, omite etiquetas vacías (comportamiento move_to_comb).
        Si False, incluye todas las etiquetas existentes (comportamiento imagenes_db).
        Default es True.
    """
    os.makedirs(destination, exist_ok=True)

    pairs = pair_images_labels(source_images, source_labels)

    if not pairs:
        print(f"No se encontraron pares imagen-etiqueta en {source_images} / {source_labels}")
        return

    empty_count = 0
    if filter_empty:
        original_count = len(pairs)
        pairs = filter_empty_labels(pairs)
        empty_count = original_count - len(pairs)
        if empty_count > 0:
            print(f"⚠ {empty_count} etiquetas vacías omitidas")

    copied = 0
    missing = 0

    for img_path, lbl_path in tqdm(pairs, desc="Copiando archivos"):
        try:
            shutil.copy2(img_path, os.path.join(destination, os.path.basename(img_path)))
            shutil.copy2(lbl_path, os.path.join(destination, os.path.basename(lbl_path)))
            copied += 1
        except Exception as e:
            print(f"❌ Error copiando {os.path.basename(img_path)}: {e}")
            missing += 1

    print(f"\nResumen: {copied} pares copiados | {empty_count} vacíos omitidos | {missing} errores")
    print(f"Archivos en: {destination}")


class DatasetBuilderTask(Task):
    """Tarea para construir dataset emparejando imágenes y etiquetas.

Example YAML:
```yaml
- name: dataset_builder
  params:
    source_images: /ruta/imagenes
    source_labels: /ruta/etiquetas
    destination: /ruta/destino
    filter_empty: true
```"""

    name = "dataset_builder"

    def __init__(self, params):
        """Inicializa la tarea.

        Parameters
        ----------
        params : dict
            Debe incluir 'source_images', 'source_labels', 'destination'.
            Opcionalmente 'filter_empty' (bool, default True).
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        """Ejecuta la construcción del dataset."""
        build_dataset(
            source_images=self.params.get("source_images"),
            source_labels=self.params.get("source_labels"),
            destination=self.params.get("destination"),
            filter_empty=bool(self.params.get("filter_empty", True)),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Construye un dataset copiando pares imagen+etiqueta a una carpeta destino"
    )
    parser.add_argument("--source-images", required=True, help="Carpeta con las imágenes")
    parser.add_argument("--source-labels", required=True, help="Carpeta con las etiquetas TXT")
    parser.add_argument("--destination", required=True, help="Carpeta de destino")
    parser.add_argument(
        "--no-filter-empty",
        action="store_true",
        help="Incluir etiquetas vacías (por defecto se omiten)",
    )
    args = parser.parse_args()

    task = DatasetBuilderTask({
        "source_images": args.source_images,
        "source_labels": args.source_labels,
        "destination": args.destination,
        "filter_empty": not args.no_filter_empty,
    })
    task.run()


if __name__ == "__main__":
    main()
