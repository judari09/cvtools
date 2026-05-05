"""Utilidades compartidas para simplificación y optimización de polígonos."""

import numpy as np
import cv2


def remove_close_points(points: list, min_dist: float = 2.0) -> list:
    """Elimina puntos consecutivos más cercanos que min_dist píxeles.

    Parameters
    ----------
    points : list
        Lista de puntos [x, y].
    min_dist : float, optional
        Distancia mínima entre puntos consecutivos. Default es 2.0.

    Returns
    -------
    list
        Lista filtrada de puntos.
    """
    if len(points) <= 3:
        return points

    filtered = [points[0]]
    for pt in points[1:]:
        dx = pt[0] - filtered[-1][0]
        dy = pt[1] - filtered[-1][1]
        if (dx * dx + dy * dy) >= min_dist * min_dist:
            filtered.append(pt)

    return filtered if len(filtered) >= 3 else points


def simplify_polygon(points: list, epsilon: float = 2.0, min_points: int = 5) -> list:
    """Simplifica un polígono usando el algoritmo Douglas-Peucker (cv2.approxPolyDP).

    Parameters
    ----------
    points : list
        Lista de puntos [x, y].
    epsilon : float, optional
        Tolerancia en píxeles (mayor = más simplificación). Default es 2.0.
    min_points : int, optional
        Número mínimo de puntos a preservar. Default es 5.

    Returns
    -------
    list
        Lista simplificada de puntos.
    """
    if len(points) <= min_points:
        return points

    pts = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(pts, epsilon, True)
    result = [[float(x), float(y)] for [x, y] in approx[:, 0, :]]

    if len(result) < min_points:
        step = max(1, len(points) // min_points)
        result = points[::step][:min_points]

    return result


def smooth_contour(points: list, window: int = 5) -> list:
    """Suaviza el contorno aplicando una media móvil circular sobre las coordenadas.

    Parameters
    ----------
    points : list
        Lista de puntos [x, y].
    window : int, optional
        Tamaño de la ventana de suavizado (impar recomendado). Default es 5.

    Returns
    -------
    list
        Lista de puntos suavizados.
    """
    if len(points) <= window:
        return points

    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    half = window // 2
    smoothed = np.zeros_like(pts)

    for i in range(n):
        indices = [(i + j) % n for j in range(-half, half + 1)]
        smoothed[i] = pts[indices].mean(axis=0)

    return [[round(float(x), 1), round(float(y), 1)] for x, y in smoothed]


def optimize_shape(
    points: list,
    epsilon: float = 3.0,
    min_dist: float = 2.0,
    min_points: int = 5,
    smooth: bool = False,
    smooth_window: int = 5,
) -> list:
    """Pipeline completo de optimización de polígono.

    Pasos: eliminar puntos cercanos → (opcional) suavizar → Douglas-Peucker.

    Parameters
    ----------
    points : list
        Lista de puntos [x, y].
    epsilon : float, optional
        Tolerancia Douglas-Peucker. Default es 3.0.
    min_dist : float, optional
        Distancia mínima entre puntos consecutivos. Default es 2.0.
    min_points : int, optional
        Mínimo de puntos a preservar. Default es 5.
    smooth : bool, optional
        Activar suavizado de contorno. Default es False.
    smooth_window : int, optional
        Tamaño de ventana de suavizado. Default es 5.

    Returns
    -------
    list
        Lista optimizada de puntos.
    """
    points = remove_close_points(points, min_dist=min_dist)

    if smooth and len(points) > smooth_window:
        points = smooth_contour(points, window=smooth_window)

    points = simplify_polygon(points, epsilon=epsilon, min_points=min_points)

    return points
