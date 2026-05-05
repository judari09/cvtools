"""Utilidades compartidas para lectura y validación de imágenes."""

import os
import cv2

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
FALLBACK_WIDTH = 640
FALLBACK_HEIGHT = 360


def validate_image(path: str):
    """Valida que la imagen sea completamente legible y decodificable.

    Parameters
    ----------
    path : str
        Ruta al archivo de imagen.

    Returns
    -------
    numpy.ndarray or None
        Array BGR de la imagen si es válida, None si está corrupta.
    """
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        ok, _ = cv2.imencode('.png', img)
        if not ok:
            return None
        return img
    except Exception:
        return None


def get_image_dimensions(image_path: str, json_data: dict = None) -> tuple:
    """Obtiene las dimensiones de una imagen con fallback a metadatos JSON.

    Prioridad:
    1. Imagen real (cv2)
    2. Metadatos del JSON (imageWidth / imageHeight)
    3. Fallback hardcodeado (640x360)

    Parameters
    ----------
    image_path : str
        Ruta base para buscar la imagen (sin extensión o con extensión).
    json_data : dict, optional
        Datos JSON con campos imageWidth / imageHeight como fallback.

    Returns
    -------
    tuple
        (width, height, source) donde source es 'imagen', 'json' o 'fallback'.
    """
    base = os.path.splitext(image_path)[0]
    folder = os.path.dirname(image_path)
    name = os.path.splitext(os.path.basename(image_path))[0]

    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(folder, name + ext)
        if os.path.exists(candidate):
            img = cv2.imread(candidate)
            if img is not None:
                h, w = img.shape[:2]
                return w, h, 'imagen'

    if json_data:
        w = json_data.get('imageWidth')
        h = json_data.get('imageHeight')
        if w and h:
            return int(w), int(h), 'json'

    return FALLBACK_WIDTH, FALLBACK_HEIGHT, 'fallback'
