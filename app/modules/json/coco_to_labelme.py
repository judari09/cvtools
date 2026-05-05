"""Convierte anotaciones en formato COCO a formato LabelMe JSON.

Este módulo proporciona funcionalidad para convertir datasets anotados en formato
COCO (con bounding boxes) al formato LabelMe JSON. Soporta la conversión de cajas
delimitadoras en coordenadas de polígonos rectangulares.

Ejemplo YAML:
```yaml
- name: coco_to_labelme
  params:
    coco_json_path: /ruta/a/_annotations.coco.json
    output_dir: /ruta/al/output
    images_dir: /ruta/a/imagenes
```
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    from app.core.task import Task
except ImportError:
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    )
    from app.core.task import Task


class CocoToLabelmeTask(Task):
    """Tarea para convertir anotaciones COCO a formato LabelMe JSON.

    Convierte un archivo COCO JSON que contiene bounding boxes rectangulares
    en múltiples archivos JSON en formato LabelMe, uno por imagen anotada.

    Los bounding boxes COCO son convertidos a polígonos rectangulares en
    LabelMe (4 puntos en las esquinas del rectángulo). Las imágenes no se
    copian; en su lugar, los archivos JSON hacen referencia a la ruta
    absoluta de las imágenes originales.

    Ejemplo YAML:
    ```yaml
    - name: coco_to_labelme
      params:
        coco_json_path: <value>
        output_dir: <value>
        images_dir: <value>
    ```
    """

    name = "coco_to_labelme"

    def __init__(self, params):
        """Inicializa la tarea.

        Parameters
        ----------
        params : dict
            Parámetros con:
            - coco_json_path: ruta al archivo COCO JSON
            - output_dir: directorio donde guardar archivos LabelMe
            - images_dir: directorio con las imágenes (para obtener dimensiones)
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    @staticmethod
    def bbox_to_polygon(
        bbox: List[float], img_width: int, img_height: int
    ) -> List[List[int]]:
        """Convierte un bounding box COCO a polígono LabelMe.

        Parameters
        ----------
        bbox : list
            Bounding box en formato COCO [x, y, width, height]
        img_width : int
            Ancho de la imagen
        img_height : int
            Alto de la imagen

        Returns
        -------
        list
            Coordenadas de los 4 puntos del rectángulo [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        x, y, w, h = bbox

        # Asegurar que los valores están dentro de los límites de la imagen
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(img_width, int(x + w))
        y2 = min(img_height, int(y + h))

        # Retornar los 4 puntos del rectángulo (en orden: arriba-izq, arriba-der, abajo-der, abajo-izq)
        return [
            [x1, y1],  # arriba-izquierda
            [x2, y1],  # arriba-derecha
            [x2, y2],  # abajo-derecha
            [x1, y2],  # abajo-izquierda
        ]

    @staticmethod
    def create_labelme_json(
        image_path: str, img_width: int, img_height: int, shapes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Crea la estructura JSON de LabelMe.

        Parameters
        ----------
        image_path : str
            Nombre de archivo de la imagen
        img_width : int
            Ancho de la imagen en píxeles
        img_height : int
            Alto de la imagen en píxeles
        shapes : list
            Lista de anotaciones (shapes) en formato LabelMe

        Returns
        -------
        dict
            Diccionario con la estructura completa de LabelMe JSON
        """
        return {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path,
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width,
        }

    def convert_coco_to_labelme(
        self, coco_json_path: str, output_dir: str, images_dir: str
    ) -> None:
        """Realiza la conversión de COCO a LabelMe.

        Parameters
        ----------
        coco_json_path : str
            Ruta al archivo COCO JSON
        output_dir : str
            Directorio de salida para archivos LabelMe JSON
        images_dir : str
            Directorio con las imágenes
        """
        # Crear directorio de salida
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Cargar archivo COCO
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # Crear diccionarios de búsqueda rápida
        images_by_id = {img["id"]: img for img in coco_data["images"]}
        categories_by_id = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

        # Agrupar anotaciones por imagen
        annotations_by_image_id = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image_id:
                annotations_by_image_id[img_id] = []
            annotations_by_image_id[img_id].append(ann)

        # Convertir cada imagen
        converted_count = 0
        skipped_count = 0

        for image_id, annotations in annotations_by_image_id.items():
            if image_id not in images_by_id:
                print(f"⚠ Imagen ID {image_id} no encontrada en lista de imágenes")
                skipped_count += 1
                continue

            image_info = images_by_id[image_id]
            image_name = image_info["file_name"]
            img_width = image_info["width"]
            img_height = image_info["height"]

            # Ruta absoluta de la imagen
            image_full_path = str(Path(images_dir) / image_name)

            # Crear lista de shapes
            shapes = []
            for ann in annotations:
                category_id = ann["category_id"]
                bbox = ann["bbox"]

                # Convertir bbox a polígono
                polygon = self.bbox_to_polygon(bbox, img_width, img_height)

                shape = {
                    "label": categories_by_id.get(category_id, "unknown"),
                    "points": polygon,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                }
                shapes.append(shape)

            # Crear JSON de LabelMe
            labelme_json = self.create_labelme_json(
                image_name, img_width, img_height, shapes
            )

            # Guardar JSON
            output_filename = Path(image_name).stem + ".json"
            output_path = Path(output_dir) / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(labelme_json, f, indent=2, ensure_ascii=False)

            converted_count += 1
            print(
                f"✓ Convertido: {image_name} → {output_filename} ({len(shapes)} objetos)"
            )

        print(f"\n✓ Conversión completada: {converted_count} imágenes convertidas")
        if skipped_count > 0:
            print(f"⚠ {skipped_count} imágenes omitidas")

    def run(self):
        """Ejecuta la tarea de conversión."""
        self.convert_coco_to_labelme(
            coco_json_path=self.params.get("coco_json_path"),
            output_dir=self.params.get("output_dir"),
            images_dir=self.params.get("images_dir"),
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convierte anotaciones COCO JSON a formato LabelMe JSON"
    )
    parser.add_argument(
        "--coco-json",
        required=True,
        help="Ruta al archivo COCO JSON (_annotations.coco.json)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde guardar archivos LabelMe JSON",
    )
    parser.add_argument(
        "--images-dir",
        default=".",
        help="Directorio con las imágenes (opcional, por defecto: directorio actual)",
    )

    args = parser.parse_args()

    task = CocoToLabelmeTask(
        {
            "coco_json_path": args.coco_json,
            "output_dir": args.output_dir,
            "images_dir": args.images_dir,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
