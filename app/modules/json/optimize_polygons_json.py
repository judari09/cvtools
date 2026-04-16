import os
import json
import shutil
import numpy as np
import cv2
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task


class OptimizePolygonsJsonTask(Task):
    name = "optimize_polygons_json"
    def __init__(self, params):
        """Initialize the OptimizePolygonsJsonTask.

        Parameters
        ----------
        params : object
            Parameters object containing configuration.
        """
        super().__init__(name="optimize_polygons_json", params=params)
        self.params = params

    def remove_close_points(self, points, min_dist=2.0):
        """Remove consecutive points that are closer than min_dist pixels.

        This removes duplicate or nearly duplicate points that generate visual noise.

        Parameters
        ----------
        points : list
            List of [x, y] points.
        min_dist : float, optional
            Minimum distance between consecutive points in pixels. Default is 2.0.

        Returns
        -------
        list
            Filtered list of points.
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
        """Simplify a polygon using Douglas-Peucker algorithm (cv2.approxPolyDP).

        Parameters
        ----------
        points : list
            List of [x, y] points.
        epsilon : float, optional
            Tolerance in pixels (higher = more simplification). Default is 2.0.
        min_points : int, optional
            Minimum number of points to preserve. Default is 5.

        Returns
        -------
        list
            Simplified list of points.
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
        """Smooth the contour by applying circular moving average on coordinates.

        Reduces irregularities without drastically changing the shape.

        Parameters
        ----------
        points : list
            List of [x, y] points.
        window : int, optional
            Size of the smoothing window (odd recommended). Default is 5.

        Returns
        -------
        list
            Smoothed list of points.
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
        """Complete polygon optimization pipeline.

        1. Remove close/duplicate points
        2. (Optional) Smooth contour
        3. Simplify with Douglas-Peucker

        Parameters
        ----------
        points : list
            List of [x, y] points.
        epsilon : float, optional
            Douglas-Peucker tolerance. Default is 3.0.
        min_dist : float, optional
            Minimum distance between consecutive points. Default is 2.0.
        min_points : int, optional
            Minimum points to preserve. Default is 5.
        smooth : bool, optional
            Enable contour smoothing. Default is False.
        smooth_window : int, optional
            Smoothing window size. Default is 5.

        Returns
        -------
        list
            Optimized list of points.
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
        """Optimize polygons in all JSON files (LabelMe format) in the folder.

        Parameters
        ----------
        folder_path : str
            Path to folder containing JSON files.
        epsilon : float, optional
            Douglas-Peucker tolerance (higher = more aggressive). Recommended: 2.0-5.0. Default is 3.0.
        min_dist : float, optional
            Minimum distance between consecutive points in pixels. Default is 2.0.
        min_points : int, optional
            Minimum points to preserve per polygon. Default is 5.
        smooth : bool, optional
            Enable contour smoothing. Default is False.
        smooth_window : int, optional
            Smoothing window size. Default is 5.
        target_label : str, optional
            If specified, only optimize shapes with this label (None = all). Default is None.
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
        """Run the polygon optimization task.
        """
        self.optimize_jsons(
            folder_path=self.params.folder_path,
            epsilon=self.params.epsilon,
            min_dist=self.params.min_dist,
            min_points=self.params.min_points,
            smooth=self.params.smooth,
            smooth_window=self.params.smooth_window,
            target_label=self.params.target_label,
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize polygon annotations in LabelMe JSON files."
    )
    parser.add_argument(
        "--folder-path",
        required=True,
        help="Folder containing JSON files to optimize.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=3.0,
        help="Douglas-Peucker epsilon for polygon simplification.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=2.0,
        help="Minimum distance for duplicate point removal.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum points to preserve per polygon.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply smoothing before simplification.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window size used by smoothing.",
    )
    parser.add_argument(
        "--target-label",
        default=None,
        help="Only optimize polygons with this label.",
    )
    args = parser.parse_args()
    params = argparse.Namespace(
        folder_path=args.folder_path,
        epsilon=args.epsilon,
        min_dist=args.min_dist,
        min_points=args.min_points,
        smooth=args.smooth,
        smooth_window=args.smooth_window,
        target_label=args.target_label,
    )
    OptimizePolygonsJsonTask(params).run()
