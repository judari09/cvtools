import os
import json
import glob
import cv2
from tqdm import tqdm
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task



# Mapeo de clases
class Json2TxtTask(Task):
    """Tarea para json2txt.

Example YAML:
```yaml
- name: json2txt
  params:
    input_dir: <value>
    output_dir: <value>
    carpeta_imagenes: <value>
    polygon_4pt_as_bbox: false
```"""

    
    name = "json2txt"
    
    def __init__(self, params):
        """Initialize the Json2TxtTask.

        Parameters
        ----------
        params : object
            Parameters object containing configuration.
        """
        super().__init__(name="json2txt", params=params)
        self.params = params

        # Dimensiones de fallback si no se puede leer imagen ni JSON
        self.FALLBACK_WIDTH = 640
        self.FALLBACK_HEIGHT = 360

        self.EXTENSIONES_IMG = [".jpg", ".jpeg", ".png"]


    def obtener_dimensiones(self,json_file, data, carpeta_imagenes):
        """Get the real dimensions of the image associated with the JSON.

        Priority:
        1. Real image (cv2)
        2. JSON metadata (imageWidth / imageHeight)
        3. Hardcoded fallback (640x360)

        Parameters
        ----------
        json_file : str
            Path to the JSON file.
        data : dict
            Loaded JSON data.
        carpeta_imagenes : str
            Directory containing images.

        Returns
        -------
        tuple
            (width, height, source) where source is 'imagen', 'json', or 'fallback'.
        """
        nombre_base = os.path.splitext(os.path.basename(json_file))[0]

        # 1. Buscar imagen en carpeta_imagenes
        for ext in self.EXTENSIONES_IMG:
            img_path = os.path.join(carpeta_imagenes, nombre_base + ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    return w, h, "imagen"

        # 2. Metadatos del JSON
        w = data.get("imageWidth")
        h = data.get("imageHeight")
        if w and h:
            return w, h, "json"

        # 3. Fallback
        return self.FALLBACK_WIDTH, self.FALLBACK_HEIGHT, "fallback"


    def convert_polygon_to_yolo(self, points, img_width, img_height):
        """Convierte puntos de polígono a formato YOLO segmentación/OBB normalizado."""
        coords = []
        for point in points:
            x = point[0] / img_width
            y = point[1] / img_height
            coords.append(f"{x:.6f} {y:.6f}")
        return " ".join(coords)

    def convert_rectangle_to_yolo(self, points, img_width, img_height):
        """Convierte dos puntos [[x1,y1],[x2,y2]] a formato YOLO detección cx cy w h."""
        x1, y1 = points[0]
        x2, y2 = points[1]
        cx = ((x1 + x2) / 2) / img_width
        cy = ((y1 + y2) / 2) / img_height
        w = abs(x2 - x1) / img_width
        h = abs(y2 - y1) / img_height
        return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def convert_4pt_polygon_to_bbox(self, points, img_width, img_height):
        """Convierte un polígono de 4 puntos a bbox YOLO (cx cy w h) usando el AABB."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = ((min(xs) + max(xs)) / 2) / img_width
        cy = ((min(ys) + max(ys)) / 2) / img_height
        w = (max(xs) - min(xs)) / img_width
        h = (max(ys) - min(ys)) / img_height
        return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def convert_shape_to_yolo(self, shape, img_width, img_height, polygon_4pt_as_bbox=False):
        """Despacha al conversor correcto según shape_type.

        Parameters
        ----------
        polygon_4pt_as_bbox : bool
            Si True, convierte polígonos de exactamente 4 puntos a bbox (detección).
            Si False, los trata como polígono normalizado (segmentación / OBB).

        Returns
        -------
        str or None
            Línea de coordenadas YOLO, o None si el shape_type no es soportado.
        """
        shape_type = shape.get("shape_type", "polygon")
        points = shape["points"]
        if shape_type == "rectangle":
            return self.convert_rectangle_to_yolo(points, img_width, img_height)
        elif shape_type == "polygon":
            if polygon_4pt_as_bbox and len(points) == 4:
                return self.convert_4pt_polygon_to_bbox(points, img_width, img_height)
            return self.convert_polygon_to_yolo(points, img_width, img_height)
        else:
            return None


    def convertir_json_a_txt(self, input_dir, output_dir, carpeta_imagenes=None, polygon_4pt_as_bbox=False):
        """Convert LabelMe JSON files to YOLO .txt format.

        Parameters
        ----------
        input_dir : str
            Directory with JSON files.
        output_dir : str
            Output directory for .txt files.
        carpeta_imagenes : str, optional
            Directory with images (default same as input_dir).
        polygon_4pt_as_bbox : bool
            If True, 4-point polygons are converted to axis-aligned bounding boxes
            (cx cy w h) suitable for detection. If False, they are kept as normalized
            polygon coordinates, useful for segmentation or OBB tasks.
        """
        if carpeta_imagenes is None:
            carpeta_imagenes = input_dir

        os.makedirs(output_dir, exist_ok=True)

        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        if not json_files:
            print(f"No se encontraron archivos JSON en: {input_dir}")
            return

        convertidos, fallidos = 0, 0
        fuentes = {"imagen": 0, "json": 0, "fallback": 0}

        for json_file in tqdm(json_files, desc="Convirtiendo etiquetas"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ Error al cargar {os.path.basename(json_file)}: {e}")
                fallidos += 1
                continue

            img_width, img_height, fuente = self.obtener_dimensiones(json_file, data, carpeta_imagenes)
            fuentes[fuente] += 1

            txt_filename = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
            txt_filepath = os.path.join(output_dir, txt_filename)

            with open(txt_filepath, "w") as txt_file:
                for shape in data.get("shapes", []):
                    label = shape["label"]
                    class_id = self.params.get("class_map", {}).get(label, self.params.get("default_class_id", 0))
                    yolo_coords = self.convert_shape_to_yolo(
                        shape, img_width, img_height,
                        polygon_4pt_as_bbox=polygon_4pt_as_bbox,
                    )
                    if yolo_coords is None:
                        print(f"Advertencia: shape_type '{shape.get('shape_type')}' no soportado en {os.path.basename(json_file)}, se omite.")
                        continue
                    txt_file.write(f"{class_id} {yolo_coords}\n")

            convertidos += 1

        print(f"\nResumen: {convertidos} convertidos | {fallidos} con error")
        print(f"Fuente de dimensiones — imagen: {fuentes['imagen']} | json: {fuentes['json']} | fallback: {fuentes['fallback']}")
        print(f"Archivos guardados en: {output_dir}")
    
    def run(self):
        """Run the JSON to TXT conversion task."""
        self.convertir_json_a_txt(
            input_dir=self.params.get("input_dir"),
            output_dir=self.params.get("output_dir"),
            carpeta_imagenes=self.params.get("carpeta_imagenes"),
            polygon_4pt_as_bbox=self.params.get("polygon_4pt_as_bbox", False),
        )


if __name__ == "__main__":
    import argparse

    def parse_class_map(entries):
        mapping = {}
        for entry in entries or []:
            if ":" in entry:
                label, value = entry.split(":", 1)
                try:
                    mapping[label] = int(value)
                except ValueError:
                    print(f"Advertencia: valor de clase inválido para '{label}': {value}")
        return mapping

    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to YOLO .txt format."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where YOLO .txt files will be written.",
    )
    parser.add_argument(
        "--carpeta-imagenes",
        default=None,
        help="Optional folder containing associated images. Defaults to input directory.",
    )
    parser.add_argument(
        "--class-map",
        nargs="*",
        default=[],
        help="Optional class map entries in label:id format.",
    )
    parser.add_argument(
        "--default-class-id",
        type=int,
        default=0,
        help="Default class ID used for labels not found in class map.",
    )
    parser.add_argument(
        "--polygon-4pt-as-bbox",
        action="store_true",
        default=False,
        help=(
            "Convert 4-point polygons to axis-aligned bounding boxes (cx cy w h). "
            "Use this for detection tasks. Without this flag, 4-point polygons are "
            "kept as normalized polygon coordinates (segmentation / OBB)."
        ),
    )
    args = parser.parse_args()
    params = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "carpeta_imagenes": args.carpeta_imagenes or args.input_dir,
        "class_map": parse_class_map(args.class_map),
        "default_class_id": args.default_class_id,
        "polygon_4pt_as_bbox": args.polygon_4pt_as_bbox,
    }
    Json2TxtTask(params).run()
