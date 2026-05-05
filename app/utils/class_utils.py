"""Utilidades compartidas para conteo y análisis de clases en datasets."""

import os
import json
from collections import Counter


def count_classes_json(folder: str) -> Counter:
    """Cuenta apariciones de cada etiqueta en archivos JSON de LabelMe.

    Parameters
    ----------
    folder : str
        Carpeta que contiene los archivos JSON.

    Returns
    -------
    Counter
        Contador con nombre de clase como clave y número de apariciones como valor.
    """
    counter = Counter()
    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(folder, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get('shapes', []):
                label = shape.get('label', '<sin_etiqueta>')
                counter[label] += 1
        except Exception:
            pass
    return counter


def count_classes_per_image_json(label_folder: str, image_files: list) -> Counter:
    """Cuenta cuántas imágenes contienen cada clase (al menos una vez).

    Diseñado para análisis de balanceo en augmentación de datasets YOLO-seg
    con etiquetas en formato LabelMe JSON.

    Parameters
    ----------
    label_folder : str
        Carpeta con los archivos JSON de etiquetas.
    image_files : list of str
        Lista de rutas a las imágenes del dataset.

    Returns
    -------
    Counter
        Contador con clase como clave y número de imágenes que la contienen como valor.
    """
    counter = Counter()
    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(label_folder, base_name + '.json')
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            clases = {shape['label'] for shape in data.get('shapes', [])}
            for clase in clases:
                counter[clase] += 1
        except Exception:
            pass
    return counter
