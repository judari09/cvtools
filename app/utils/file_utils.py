"""Utilidades compartidas para manejo de archivos de imágenes y etiquetas."""

import os
import shutil

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def get_image_files(folder: str) -> list:
    """Devuelve lista de nombres de archivo de imagen en una carpeta."""
    return [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]


def pair_images_labels(images_folder: str, labels_folder: str) -> list:
    """Empareja imágenes con sus etiquetas TXT correspondientes.

    Parameters
    ----------
    images_folder : str
        Carpeta que contiene las imágenes.
    labels_folder : str
        Carpeta que contiene los archivos .txt de etiquetas.

    Returns
    -------
    list of tuple
        Lista de (image_path, label_path) para cada par encontrado.
    """
    available_images = {}
    for f in os.listdir(images_folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTENSIONS:
            available_images[name] = os.path.join(images_folder, f)

    pairs = []
    for f in sorted(os.listdir(labels_folder)):
        if not f.lower().endswith('.txt'):
            continue
        name = os.path.splitext(f)[0]
        if name in available_images:
            pairs.append((available_images[name], os.path.join(labels_folder, f)))
    return pairs


def filter_empty_labels(pairs: list) -> list:
    """Filtra pares donde el archivo de etiqueta está vacío.

    Parameters
    ----------
    pairs : list of tuple
        Lista de (image_path, label_path).

    Returns
    -------
    list of tuple
        Solo los pares con etiqueta no vacía.
    """
    return [(img, lbl) for img, lbl in pairs if os.path.getsize(lbl) > 0]


def copy_pair(image_path: str, label_path: str, dest_folder: str) -> None:
    """Copia imagen y etiqueta a la carpeta destino.

    Parameters
    ----------
    image_path : str
        Ruta a la imagen origen.
    label_path : str
        Ruta al archivo de etiqueta origen.
    dest_folder : str
        Carpeta de destino donde se copiarán ambos archivos.
    """
    shutil.copy2(image_path, os.path.join(dest_folder, os.path.basename(image_path)))
    shutil.copy2(label_path, os.path.join(dest_folder, os.path.basename(label_path)))
