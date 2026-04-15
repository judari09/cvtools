import os
import json
import shutil
import numpy as np
import cv2
from core.task import Task

class OptimizePolygonsJsonTask(Task):
    name = "optimize_polygons_json"
    def __init__(self, params):
        super().__init__(name="optimize_polygons_json", params=params)
        self.params = params

    def remove_close_points(self, points, min_dist=2.0):
        """
        Elimina puntos consecutivos que esten a menos de min_dist pixeles.
        Esto remueve puntos duplicados o casi duplicados que generan ruido visual.
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


    def simplify_polygon(self, points, epsilon=2.0, min_points=5):
        """
        Simplifica un poligono usando Douglas-Peucker (cv2.approxPolyDP).
        - epsilon: tolerancia en pixeles (mayor = mas simplificacion)
        - min_points: minimo de puntos que debe conservar
        """
        if len(points) <= min_points:
            return points

        pts = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
        approx = cv2.approxPolyDP(pts, epsilon, True)
        result = [[float(x), float(y)] for [x, y] in approx[:, 0, :]]

        # Si se redujo demasiado, tomar puntos distribuidos del original
        if len(result) < min_points:
            step = max(1, len(points) // min_points)
            result = points[::step][:min_points]

        return result


    def smooth_contour(self,points, window=5):
        """
        Suaviza el contorno aplicando media movil circular sobre las coordenadas.
        Reduce irregularidades sin cambiar drasticamente la forma.
        - window: tamaño de la ventana de suavizado (impar recomendado)
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


    def optimize_shape(self,points, epsilon=3.0, min_dist=2.0, min_points=5, smooth=False, smooth_window=5):
        """
        Pipeline completo de optimizacion de un poligono:
        1. Eliminar puntos cercanos/duplicados
        2. (Opcional) Suavizar contorno
        3. Simplificar con Douglas-Peucker
        """
        # Paso 1: eliminar puntos cercanos
        points = self.remove_close_points(points, min_dist=min_dist)

        # Paso 2: suavizado opcional
        if smooth and len(points) > smooth_window:
            points = self.smooth_contour(points, window=smooth_window)

        # Paso 3: simplificacion Douglas-Peucker
        points = self.simplify_polygon(points, epsilon=epsilon, min_points=min_points)

        return points


    def optimize_jsons(self, folder_path, epsilon=3.0, min_dist=2.0, min_points=5,
                    smooth=False, smooth_window=5, target_label=None):
        """
        Recorre todos los JSON (formato LabelMe) en la carpeta y optimiza los poligonos.
        - epsilon: tolerancia Douglas-Peucker (mayor = mas agresivo). Recomendado: 2.0-5.0
        - min_dist: distancia minima entre puntos consecutivos en px
        - min_points: minimo de puntos a conservar por poligono
        - smooth: activar suavizado de contorno
        - smooth_window: ventana del suavizado
        - target_label: si se especifica, solo optimiza shapes con ese label (None = todos)
        """
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

        if not json_files:
            print(f"No se encontraron archivos JSON en {folder_path}")
            return

        # Crear backup de los JSON originales antes de modificar
        backup_dir = os.path.join(folder_path, '_backup_antes_optimizar')
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            for file_name in json_files:
                src = os.path.join(folder_path, file_name)
                dst = os.path.join(backup_dir, file_name)
                shutil.copy2(src, dst)
            print(f"  Backup creado en: {backup_dir} ({len(json_files)} archivos)")
        else:
            print(f"  Backup ya existe en: {backup_dir} (no se sobreescribe)")

        total_before = 0
        total_after = 0
        files_modified = 0

        for file_name in json_files:
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            shapes = data.get('shapes', [])
            modified = False

            for shape in shapes:
                if shape.get('shape_type') != 'polygon':
                    continue

                if target_label and shape['label'] != target_label:
                    continue

                original_count = len(shape['points'])
                optimized = self.optimize_shape(
                    shape['points'],
                    epsilon=epsilon,
                    min_dist=min_dist,
                    min_points=min_points,
                    smooth=smooth,
                    smooth_window=smooth_window,
                )
                new_count = len(optimized)

                if new_count < original_count:
                    shape['points'] = optimized
                    total_before += original_count
                    total_after += new_count
                    modified = True

            if modified:
                files_modified += 1
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

        reduction = ((total_before - total_after) / total_before * 100) if total_before > 0 else 0
        print("\n Resultados de optimizacion:")
        print(f"  Archivos modificados: {files_modified}/{len(json_files)}")
        print(f"  Puntos antes:  {total_before}")
        print(f"  Puntos despues: {total_after}")
        print(f"  Reduccion: {reduction:.1f}%")

    def run(self):
        self.optimize_jsons(
            folder_path=self.params.folder_path,
            epsilon=self.params.epsilon,
            min_dist=self.params.min_dist,
            min_points=self.params.min_points,
            smooth=self.params.smooth,
            smooth_window=self.params.smooth_window,
            target_label=self.params.target_label,
        )
