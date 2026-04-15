import os
import json
import glob
import cv2
from tqdm import tqdm
from core.task import Task


# Mapeo de clases
class Json2TxtTask(Task):
    
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


    def convert_to_yolo_format(self, points, img_width, img_height):
        """Convert list of points to normalized string for YOLO segmentation.

        Parameters
        ----------
        points : list
            List of [x, y] points.
        img_width : int
            Image width.
        img_height : int
            Image height.

        Returns
        -------
        str
            Space-separated normalized coordinates.
        """
        coords = []
        for point in points:
            x = point[0] / img_width
            y = point[1] / img_height
            coords.append(f"{x:.6f} {y:.6f}")
        return " ".join(coords)


    def convertir_json_a_txt(self,input_dir, output_dir, carpeta_imagenes=None):
        """Convert LabelMe JSON files to YOLO segmentation format (.txt).

        Parameters
        ----------
        input_dir : str
            Directory with JSON files.
        output_dir : str
            Output directory for .txt files.
        carpeta_imagenes : str, optional
            Directory with images (default same as input_dir).
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
                    points = shape["points"]
                    class_id = self.params.CLASS_MAP.get(label, self.params.DEFAULT_CLASS_ID)
                    yolo_points = self.convert_to_yolo_format(points, img_width, img_height)
                    txt_file.write(f"{class_id} {yolo_points}\n")

            convertidos += 1

        print(f"\nResumen: {convertidos} convertidos | {fallidos} con error")
        print(f"Fuente de dimensiones — imagen: {fuentes['imagen']} | json: {fuentes['json']} | fallback: {fuentes['fallback']}")
        print(f"Archivos guardados en: {output_dir}")
    
    def run(self):
        """Run the JSON to TXT conversion task.
        """
        self.convertir_json_a_txt(
            input_dir=self.params.input_dir,
            output_dir=self.params.output_dir,
            carpeta_imagenes=self.params.carpeta_imagenes,
        )

