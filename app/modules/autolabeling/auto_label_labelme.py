import os
import argparse
import numpy as np
import json
import tempfile
import signal
import contextlib
import cv2
from ultralytics import YOLO
from tqdm import tqdm
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task



class _TimeoutError(Exception):
    pass

class AutoLabelLabelMeTask(Task):
    """
    Task for auto-labeling images using YOLO models and generating LabelMe JSON annotations.

    This class extends the base Task class to perform automatic labeling of images
    using segmentation and detection YOLO models, optionally refining with SAM.
    Generates LabelMe-compatible JSON files with polygon annotations.

    Attributes
    ----------
    params : object
        Configuration parameters for the task.
    """

    name = "auto_label_labelme"
    def __init__(self, params):
        """
        Initialize the AutoLabelLabelMeTask with given parameters.

        Parameters
        ----------
        params : object
            Configuration parameters including model paths, input/output directories,
            confidence thresholds, and other settings.
        """
        super().__init__(name="auto_label_labelme", params=params)
        self.params = params


    @contextlib.contextmanager
    def timeout(self, seconds):
        """
        Context manager that raises _TimeoutError if the block exceeds 'seconds' seconds.

        Parameters
        ----------
        seconds : int
            Timeout duration in seconds.

        Yields
        ------
        None
        """
        def _handler(signum, frame):
            raise _TimeoutError(f"Timeout tras {seconds}s")

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


    def validate_image(self, image_path):
        """
        Validate that the image is fully readable and decodable.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        numpy.ndarray or None
            BGR image array if valid, None if corrupted.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            # Forzar decodificación completa: codificar de vuelta a PNG en memoria
            ok, _ = cv2.imencode(".png", img)
            if not ok:
                return None
            return img
        except Exception:
            return None


    def simplify_polygon(self, points, epsilon=2.0, min_points=5):
        """
        Simplify a polygon using Douglas-Peucker algorithm.

        Parameters
        ----------
        points : list of list
            List of [x, y] points.
        epsilon : float, optional
            Tolerance for simplification in pixels. Default is 2.0.
        min_points : int, optional
            Minimum number of points the polygon must have. Default is 5.

        Returns
        -------
        list of list
            Simplified polygon points.
        """
        if len(points) <= min_points:
            return points  # ya cumple el mínimo

        pts = np.array(points, dtype=np.float32)
        contour = pts.reshape((-1, 1, 2))
        approx = cv2.approxPolyDP(contour, epsilon, True)

        approx_points = [[float(x), float(y)] for [x, y] in approx[:, 0, :]]

        # Si se redujo demasiado, tomar puntos distribuidos del original
        if len(approx_points) < min_points:
            step = max(1, len(points) // min_points)
            approx_points = points[::step][:min_points]

        return approx_points

    def load_models(self, seg_paths, det_paths=None, sam_path=None):
        """
        Load YOLO segmentation, detection, and optional SAM models.

        Parameters
        ----------
        seg_paths : list of str
            Paths to segmentation model files.
        det_paths : list of str, optional
            Paths to detection model files.
        sam_path : str, optional
            Path to SAM model file.

        Returns
        -------
        tuple
            (list of YOLO seg models, list of YOLO det models, SAM model or None).
        """
        seg_models = []
        for path in seg_paths:
            print(f"Cargando modelo YOLO-seg: {path}")
            seg_models.append(YOLO(path))

        det_models = []
        if det_paths:
            for path in det_paths:
                print(f"Cargando modelo YOLO-det: {path}")
                det_models.append(YOLO(path))

        sam_model = None
        if sam_path:
            from ultralytics import SAM

            print(f"Cargando modelo SAM: {sam_path}")
            sam_model = SAM(sam_path)

        return seg_models, det_models, sam_model


    def parse_class_map(self, class_map_list):
        """
        Parse the list of 'original:new' mappings to a dictionary.

        Parameters
        ----------
        class_map_list : list of str
            List of 'original:new' strings. If 'new' is 'null', the class is discarded.

        Returns
        -------
        dict or None
            Dictionary {original: new_or_None}, or None if no mappings.
        """
        if not class_map_list:
            return None

        mapping = {}
        for entry in class_map_list:
            parts = entry.split(":", 1)
            if len(parts) != 2:
                print(f"Advertencia: formato inválido en class-map '{entry}', se ignora")
                continue
            original, nuevo = parts
            mapping[original] = None if nuevo.lower() == "null" else nuevo
        return mapping


    def run_multi_model_inference(self, models, image_path, conf, class_map, infer_timeout=60):
        """
        Run each YOLO model on the image and collect all detections.

        Applies class mapping: renames or discards according to class_map.

        Parameters
        ----------
        models : list
            List of YOLO models.
        image_path : str
            Path to the image file.
        conf : float
            Confidence threshold.
        class_map : dict or None
            Class mapping dictionary.
        infer_timeout : int, optional
            Inference timeout in seconds. Default is 60.

        Returns
        -------
        list of dict
            List of detections with keys: label, points (pixels), box (xyxy).
        """
        detections = []

        image_bgr = self.validate_image(image_path)
        if image_bgr is None:
            print(f"Advertencia: imagen corrupta o ilegible '{image_path}', se omite.")
            return detections

        for model in models:
            try:
                with self.timeout(infer_timeout):
                    results = model(image_bgr, task="segment", conf=conf, verbose=False, save=False)
            except _TimeoutError:
                print(f"Advertencia: timeout en inferencia seg para '{image_path}', se omite.")
                return detections
            except Exception as e:
                print(f"Advertencia: error en inferencia seg para '{image_path}': {e}")
                return detections
            result = results[0]

            if result.masks is None:
                continue

            for mask, box in zip(result.masks.xy, result.boxes):
                cls_id = int(box.cls[0].item())
                label = result.names[cls_id]

                # Aplicar mapeo de clases
                if class_map is not None:
                    if label not in class_map:
                        continue  # clase no está en el mapeo, se ignora
                    mapped = class_map[label]
                    if mapped is None:
                        continue  # clase descartada explícitamente
                    label = mapped

                points = [[float(x), float(y)] for x, y in mask]
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                detections.append(
                    {
                        "label": label,
                        "points": points,
                        "box": bbox,
                    }
                )

        return detections


    def run_detection_inference(self, models, image_path, conf, class_map, infer_timeout=60):
        """
        Run YOLO detection models (bboxes only, no masks).

        Returns list of dicts with keys: label, box (xyxy), points=None.
        These detections REQUIRE SAM to generate polygons.

        Parameters
        ----------
        models : list
            List of YOLO detection models.
        image_path : str
            Path to the image file.
        conf : float
            Confidence threshold.
        class_map : dict or None
            Class mapping dictionary.
        infer_timeout : int, optional
            Inference timeout in seconds. Default is 60.

        Returns
        -------
        list of dict
            List of detections with keys: label, box (xyxy), points=None.
        """
        detections = []

        image_bgr = self.validate_image(image_path)
        if image_bgr is None:
            print(f"Advertencia: imagen corrupta o ilegible '{image_path}', se omite.")
            return detections

        for model in models:
            try:
                with self.timeout(infer_timeout):
                    results = model(image_bgr, conf=conf, verbose=False, save=False)
            except _TimeoutError:
                print(f"Advertencia: timeout en inferencia det para '{image_path}', se omite.")
                return detections
            except Exception as e:
                print(f"Advertencia: error en inferencia det para '{image_path}': {e}")
                return detections
            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = result.names[cls_id]

                # Aplicar mapeo de clases
                if class_map is not None:
                    if label not in class_map:
                        continue  # clase no está en el mapeo, se ignora
                    mapped = class_map[label]
                    if mapped is None:
                        continue  # clase descartada explícitamente
                    label = mapped

                bbox = box.xyxy[0].tolist()

                detections.append(
                    {
                        "label": label,
                        "points": None,  # sin máscara, SAM la generará
                        "box": bbox,
                    }
                )

        return detections


    def refine_with_sam(self, sam_model, image_path, detections):
        """
        Use SAM to refine masks using YOLO bboxes as prompts.

        Replaces YOLO polygons with SAM's more precise ones.

        Parameters
        ----------
        sam_model : SAM
            SAM model instance.
        image_path : str
            Path to the image file.
        detections : list of dict
            List of detections to refine.

        Returns
        -------
        list of dict
            Refined detections with updated points.
        """
        if not detections:
            return detections

        boxes = [d["box"] for d in detections]
        boxes_array = np.array(boxes, dtype=np.float32)

        results = sam_model.predict(image_path, bboxes=boxes_array, verbose=False, save=False)
        result = results[0]

        if result.masks is None:
            return detections

        refined = []
        for i, det in enumerate(detections):
            if i < len(result.masks.xy) and len(result.masks.xy[i]) > 0:
                sam_points = [[float(x), float(y)] for x, y in result.masks.xy[i]]
                det["points"] = sam_points
            refined.append(det)

        return refined


    def build_labelme_json(self, shapes, image_path, output_dir):
        """
        Build the LabelMe JSON dict from shapes.

        Calculates imagePath as relative path from output_dir to the image.

        Parameters
        ----------
        shapes : list
            List of shape dictionaries.
        image_path : str
            Path to the image file.
        output_dir : str
            Output directory for JSON files.

        Returns
        -------
        dict or None
            LabelMe JSON dict, or None if image cannot be read.
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Advertencia: no se pudo leer imagen para dimensiones '{image_path}'")
                return None
            height, width = img.shape[:2]

            # Ruta relativa desde el directorio de salida (donde vive el JSON)
            # hasta la imagen, igual que LabelMe lo genera manualmente.
            # Se usan rutas absolutas para evitar errores si alguna es relativa.
            rel_path = os.path.relpath(os.path.abspath(image_path), os.path.abspath(output_dir))

            return {
                "version": "5.10.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": rel_path,
                "imageData": None,
                "imageHeight": int(height),
                "imageWidth": int(width),
            }
        except Exception as e:
            print(f"etiqueta no genera para {image_path} por {e}")


    def auto_label_images(
        self,
        seg_model_paths,
        input_dir,
        output_dir,
        conf=0.3,
        det_model_paths=None,
        use_sam=False,
        sam_model_path=None,
        class_map=None,
        epsilon=2.0,
    ):
        """
        Generate LabelMe JSON label files using predictions from multiple YOLO models + optional SAM.

        Parameters
        ----------
        seg_model_paths : list of str
            Segmentation model paths (generate masks directly).
        input_dir : str
            Input directory with images.
        output_dir : str
            Output directory for JSON files.
        conf : float, optional
            Confidence threshold. Default is 0.3.
        det_model_paths : list of str, optional
            Detection model paths (bboxes only, require SAM).
        use_sam : bool, optional
            Whether to use SAM for refinement. Default is False.
        sam_model_path : str, optional
            Path to SAM model.
        class_map : dict, optional
            Class mapping dictionary.
        epsilon : float, optional
            Polygon simplification epsilon. Default is 2.0.

        Notes
        -----
        If JSON already exists, it does not overwrite.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Validación: modelos de detección requieren SAM
        if det_model_paths and not use_sam:
            print(
                "ERROR: --det-models requiere --use-sam. "
                "Los modelos de detección solo generan bboxes, "
                "SAM es necesario para obtener las máscaras de segmentación."
            )
            return

        # Cargar modelos
        seg_models, det_models, sam_model = self.load_models(
            seg_model_paths,
            det_paths=det_model_paths,
            sam_path=sam_model_path if use_sam else None,
        )

        # Listar imágenes
        try:
            image_files = [
                f
                for f in os.listdir(input_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        except OSError as e:
            print(f"ERROR: no se puede acceder al directorio '{input_dir}': {e}")
            raise

        if not image_files:
            print(f"No se encontraron imágenes en {input_dir}")
            return

        for filename in tqdm(image_files, desc="Etiquetando imágenes"):
            image_path = os.path.join(input_dir, filename)

            # Nombre del archivo JSON correspondiente
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(output_dir, json_filename)

            # Cargar shapes existentes si el JSON ya existe (anotaciones manuales)
            existing_shapes = []
            existing_labels = set()
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    existing_shapes = existing_data.get("shapes", [])
                    existing_labels = {s["label"] for s in existing_shapes}
                except (json.JSONDecodeError, OSError) as e:
                    print(f"Advertencia: JSON corrupto '{json_path}', se regenerará: {e}")

            # Inferencia con modelos de segmentación
            seg_detections = self.run_multi_model_inference(
                seg_models, image_path, conf, class_map
            )

            # Refinamiento con SAM para modelos de segmentación (opcional)
            if sam_model and seg_detections:
                seg_detections = self.refine_with_sam(sam_model, image_path, seg_detections)

            # Inferencia con modelos de detección + SAM obligatorio
            det_detections = []
            if det_models:
                det_detections = self.run_detection_inference(
                    det_models, image_path, conf, class_map
                )
                if sam_model and det_detections:
                    det_detections = self.refine_with_sam(sam_model, image_path, det_detections)
                    # Filtrar detecciones donde SAM no pudo generar máscara
                    det_detections = [d for d in det_detections if d["points"] is not None]

            # Combinar todas las detecciones
            all_detections = seg_detections + det_detections

            # Construir shapes nuevas, solo para labels que NO existan ya
            new_shapes = []
            for det in all_detections:
                if det["label"] in existing_labels:
                    continue  # ya existe una anotación manual para este label
                points = det["points"]
                # points = simplify_polygon(det["points"], epsilon=epsilon, min_points=5)
                new_shapes.append(
                    {
                        "label": det["label"],
                        "points": points,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                        "mask": None,
                    }
                )

            # Si el JSON ya existía y no hay shapes nuevas, no modificar
            if existing_shapes and not new_shapes:
                continue

            # Combinar: primero las manuales existentes, luego las nuevas
            all_shapes = existing_shapes + new_shapes

            try:
                # Generar y guardar JSON
                labelme_data = self.build_labelme_json(all_shapes, image_path, output_dir)
                if labelme_data is None:
                    print(f"Advertencia: no se generó JSON para '{filename}', se omite.")
                    continue

                # Escritura atómica: escribir en temporal y luego renombrar
                dir_salida = os.path.dirname(json_path)
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=dir_salida, suffix=".tmp", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                    json.dump(labelme_data, tmp, ensure_ascii=False, indent=2)
                os.replace(tmp_path, json_path)
            except Exception as e:
                print(f"Error al guardar JSON para '{filename}': {e}")
            
        print(f"Etiquetado completo. JSONs guardados en: {output_dir}")


    def run(self):
        """
        Execute the auto-labeling process using configured parameters.

        Parses class map and calls auto_label_images with task parameters.
        """
        class_map = self.parse_class_map(self.params.class_map)
        self.auto_label_images(
            seg_model_paths=self.params.models,
            input_dir=self.params.input,
            output_dir=self.params.output,
            conf=self.params.conf,
            det_model_paths=self.params.det_models,
            use_sam=self.params.use_sam,
            sam_model_path=self.params.sam_model,
            class_map=class_map,
            epsilon=self.params.epsilon,
        )

if __name__ == "__main__":
    import argparse

    def parse_class_map(entries):
        values = []
        for entry in entries or []:
            if ":" in entry:
                values.append(entry)
            else:
                print(f"Advertencia: formato inválido para class-map '{entry}', se ignora.")
        return values

    parser = argparse.ArgumentParser(
        description="Auto-label images using YOLO models and generate LabelMe JSONs."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to one or more YOLO segmentation models.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input images folder.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for generated JSON files.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--det-models",
        nargs="*",
        default=[],
        help="Optional YOLO detection models.",
    )
    parser.add_argument(
        "--use-sam",
        action="store_true",
        help="Enable SAM refinement when available.",
    )
    parser.add_argument(
        "--sam-model",
        default=None,
        help="Optional SAM model path.",
    )
    parser.add_argument(
        "--class-map",
        nargs="*",
        default=[],
        help="Optional class renaming map entries as original:new or original:null.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="Polygon simplification epsilon for mask shapes.",
    )
    args = parser.parse_args()
    params = argparse.Namespace(
        models=args.models,
        input=args.input,
        output=args.output,
        conf=args.conf,
        det_models=args.det_models,
        use_sam=args.use_sam,
        sam_model=args.sam_model,
        class_map=parse_class_map(args.class_map),
        epsilon=args.epsilon,
    )
    AutoLabelLabelMeTask(params).run()
