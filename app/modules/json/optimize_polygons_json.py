import os
import json
import shutil
try:
    from app.core.task import Task
    from app.utils.polygon_utils import remove_close_points, simplify_polygon, smooth_contour, optimize_shape
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task
    from app.utils.polygon_utils import remove_close_points, simplify_polygon, smooth_contour, optimize_shape


class OptimizePolygonsJsonTask(Task):
    """Tarea para optimize_polygons_json.

Example YAML:
```yaml
- name: optimize_polygons_json
  params:
    folder_path: <value>
    epsilon: <value>
    min_points: <value>
    min_dist: <value>
    smooth: <value>
    smooth_window: <value>
    target_label: <value>
```"""

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
                optimized = optimize_shape(
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
        """Run the polygon optimization task."""
        self.optimize_jsons(
            folder_path=self.params.get("folder_path"),
            epsilon=float(self.params.get("epsilon", 3.0)),
            min_dist=float(self.params.get("min_dist", 2.0)),
            min_points=int(self.params.get("min_points", 5)),
            smooth=bool(self.params.get("smooth", False)),
            smooth_window=int(self.params.get("smooth_window", 5)),
            target_label=self.params.get("target_label"),
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
    params = {
        "folder_path": args.folder_path,
        "epsilon": args.epsilon,
        "min_dist": args.min_dist,
        "min_points": args.min_points,
        "smooth": args.smooth,
        "smooth_window": args.smooth_window,
        "target_label": args.target_label,
    }
    OptimizePolygonsJsonTask(params).run()
