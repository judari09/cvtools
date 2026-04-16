"""Verifica la integridad de un dataset de segmentación YOLO.

Este módulo revisa que cada imagen tenga una etiqueta .txt asociada y permite
visualizar las máscaras de segmentación para inspección manual.
"""

import argparse
import os

import cv2
import numpy as np

from app.core.task import Task


def draw_segments_on_image(img_path, label_path, output_window=True):
    """Dibuja los segmentos de un archivo de etiquetas YOLO sobre la imagen.

    Parameters
    ----------
    img_path : str
        Ruta de la imagen.
    label_path : str
        Ruta del archivo de etiquetas YOLO.
    output_window : bool, optional
        Si True, muestra la imagen en una ventana.

    Returns
    -------
    numpy.ndarray
        Imagen resultante con las etiquetas dibujadas.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")

    h, w = img.shape[:2]

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = [l.strip().split() for l in f if l.strip()]

    for parts in lines:
        cls_id = int(parts[0])
        values = list(map(float, parts[1:]))

        coords = np.array(values).reshape(-1, 2)
        abs_coords = (coords * [w, h]).astype(int)

        cv2.polylines(img, [abs_coords], isClosed=True, color=(0, 255, 0), thickness=2)
        M = cv2.moments(abs_coords)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(img, str(cls_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if output_window:
        cv2.imshow("Verificación de segmentación", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


class CheckIntegrityDatasetTask(Task):
    """Tarea para verificar integridad de imágenes y etiquetas YOLO.

    Esta tarea revisa que cada imagen en un directorio tenga su etiqueta .txt
    correspondiente en el directorio de etiquetas.
    """

    name = "check_integrity_dataset"

    def __init__(self, params):
        """Inicializa la tarea con sus parámetros.

        Parameters
        ----------
        params : dict
            Parámetros de la tarea.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def verify_dataset(self, images_dir, labels_dir, output_window=True):
        """Verifica pares imagen/etiqueta en el dataset.

        Parameters
        ----------
        images_dir : str
            Carpeta con imágenes.
        labels_dir : str
            Carpeta con etiquetas .txt.
        output_window : bool, optional
            Si True, muestra cada imagen con su segmentación.
        """
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            img_path = os.path.join(images_dir, fname)
            label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + '.txt')

            if os.path.exists(label_path):
                print(f"Procesando: {fname}")
                draw_segments_on_image(img_path, label_path, output_window=output_window)
            else:
                print(f"Etiqueta ausente para: {fname}")

    def run(self):
        """Ejecuta la tarea con los parámetros configurados."""
        self.verify_dataset(
            images_dir=self.params.get("images_dir"),
            labels_dir=self.params.get("labels_dir"),
            output_window=self.params.get("output_window", True),
        )


def main():
    parser = argparse.ArgumentParser(description="Verifica integridad de dataset YOLO")
    parser.add_argument("--images", required=True, help="Carpeta con las imágenes")
    parser.add_argument("--labels", required=True, help="Carpeta con las etiquetas TXT")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="No mostrar las imágenes en ventana",
    )
    args = parser.parse_args()

    task = CheckIntegrityDatasetTask(
        {
            "images_dir": args.images,
            "labels_dir": args.labels,
            "output_window": not args.no_window,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
