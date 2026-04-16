"""Separa un conjunto de imágenes y etiquetas TXT en train y val."""

import argparse
import os
import random
import shutil

from app.core.task import Task


def mover_archivo(source_folder, filename, destination_folder):
    """Copia la imagen y su etiqueta correspondiente a la carpeta destino."""
    base_name, _ = os.path.splitext(filename)
    image_source = os.path.join(source_folder, filename)
    label_source = os.path.join(source_folder, base_name + ".txt")

    shutil.copy2(image_source, os.path.join(destination_folder, "images", filename))
    if os.path.exists(label_source):
        shutil.copy2(label_source, os.path.join(destination_folder, "labels", base_name + ".txt"))


def separar_datos(source_folder, train_folder, val_folder, split_ratio=0.8):
    """Separa imágenes en conjuntos de entrenamiento y validación.

    Parameters
    ----------
    source_folder : str
        Carpeta con imágenes y etiquetas TXT.
    train_folder : str
        Carpeta de salida para train.
    val_folder : str
        Carpeta de salida para val.
    split_ratio : float, optional
        Proporción de imágenes para train. Default es 0.8.
    """
    for folder in [train_folder, val_folder]:
        os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

    archivos = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    random.shuffle(archivos)

    num_train = int(len(archivos) * split_ratio)
    train_files = archivos[:num_train]
    val_files = archivos[num_train:]

    for archivo in train_files:
        mover_archivo(source_folder, archivo, train_folder)
    for archivo in val_files:
        mover_archivo(source_folder, archivo, val_folder)

    print(f"Datos separados: {len(train_files)} en entrenamiento, {len(val_files)} en validación")


class SepararTrainValTask(Task):
    """Tarea para separar un conjunto en train y validation."""

    name = "separar_train_val"

    def __init__(self, params):
        """Inicializa la tarea.

        Parameters
        ----------
        params : dict
            Debe incluir 'source_folder', 'train_folder', 'val_folder' y
            opcionalmente 'split_ratio'.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        """Ejecuta la separación del dataset."""
        separar_datos(
            self.params.get("source_folder"),
            self.params.get("train_folder"),
            self.params.get("val_folder"),
            split_ratio=float(self.params.get("split_ratio", 0.8)),
        )


def main():
    parser = argparse.ArgumentParser(description="Separa imágenes y etiquetas en train/val para YOLO")
    parser.add_argument("--origen", required=True, help="Carpeta origen con imágenes y TXTs (comb)")
    parser.add_argument("--train", required=True, help="Carpeta de salida train")
    parser.add_argument("--val", required=True, help="Carpeta de salida val")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Fracción para train (default: 0.8)",
    )
    args = parser.parse_args()

    task = SepararTrainValTask(
        {
            "source_folder": args.origen,
            "train_folder": args.train,
            "val_folder": args.val,
            "split_ratio": args.ratio,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
