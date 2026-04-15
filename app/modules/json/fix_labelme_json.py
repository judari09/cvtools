"""
Corrige archivos JSON generados por auto_label_labelme.py para que sean
completamente válidos y abribles en LabelMe.

Correcciones aplicadas:
  1. imagePath  → ruta relativa desde el directorio del JSON hasta la imagen
  2. version    → se actualiza a la versión indicada (default 5.10.1)
  3. imageHeight / imageWidth → int nativo de Python (no numpy)
  4. imageData  → null (evita archivos enormes con base64)
  5. Cada shape  → garantiza que tenga todos los campos requeridos por LabelMe
"""

import os
import argparse
import json
import glob

from core.task import Task

class FixLabelMeJsonTask(Task):
    
    name = "fix_labelme_json"
    
    def __init__(self, params):
        super().__init__(name="fix_labelme_json", params=params)
        self.params = params
        
        self.SHAPE_DEFAULTS = {
            "label": "",
            "points": [],
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        }

        # Extensiones de imagen soportadas
        self.IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def run(self):
        self.fix_all_jsons(
            labels_dir=self.params.labels_dir,
            images_dir=self.params.images_dir,
            version=self.params.version,
            dry_run=self.params.dry_run,
        )




    def find_image_for_json(self,json_stem, images_dir):
        """
        Busca la imagen correspondiente al JSON probando cada extensión.
        Retorna la ruta absoluta de la imagen o None si no existe.
        """
        for ext in self.IMG_EXTENSIONS:
            candidate = os.path.join(images_dir, json_stem + ext)
            if os.path.isfile(candidate):
                return candidate
        return None


    def fix_shape(self,shape):
        """
        Garantiza que cada shape tenga todos los campos que LabelMe espera.
        Solo agrega campos faltantes con valores por defecto.
        NO modifica campos que ya existen (preserva anotaciones manuales).
        """
        fixed = dict(shape)  # copia del original
        changed = False
        for key, default_val in self.SHAPE_DEFAULTS.items():
            if key not in fixed:
                fixed[key] = default_val
                changed = True
        return fixed, changed


    def fix_json_file(self,json_path, images_dir, version, dry_run=False):
        """
        Lee un JSON, aplica todas las correcciones y lo sobreescribe.
        Retorna (corregido: bool, mensaje: str).
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_dir = os.path.dirname(os.path.abspath(json_path))
        json_stem = os.path.splitext(os.path.basename(json_path))[0]

        changes = []

        # --- 1. Corregir imagePath ---
        image_file = self.find_image_for_json(json_stem, images_dir)
        if image_file:
            rel_path = os.path.relpath(image_file, json_dir)
            if data.get("imagePath") != rel_path:
                changes.append(f"imagePath: '{data.get('imagePath')}' → '{rel_path}'")
                data["imagePath"] = rel_path
        else:
            # Si no encuentra la imagen, intentar corregir solo si es ruta absoluta
            current = data.get("imagePath", "")
            if os.path.isabs(current):
                rel = os.path.relpath(current, json_dir)
                changes.append(f"imagePath (abs→rel): '{current}' → '{rel}'")
                data["imagePath"] = rel

        # --- 2. Corregir version ---
        if data.get("version") != version:
            changes.append(f"version: '{data.get('version')}' → '{version}'")
            data["version"] = version

        # --- 3. Asegurar flags ---
        if "flags" not in data:
            data["flags"] = {}
            changes.append("flags: añadido")

        # --- 4. Corregir imageData ---
        if data.get("imageData") is not None:
            changes.append("imageData: eliminado (era base64)")
            data["imageData"] = None

        # --- 5. Corregir imageHeight / imageWidth ---
        for dim in ("imageHeight", "imageWidth"):
            val = data.get(dim)
            if val is not None and not isinstance(val, int):
                data[dim] = int(val)
                changes.append(f"{dim}: convertido a int")

        # --- 6. Corregir shapes (solo agregar campos faltantes, no modificar existentes) ---
        if "shapes" not in data:
            data["shapes"] = []
            changes.append("shapes: añadido vacío")
        else:
            for i, shape in enumerate(data["shapes"]):
                fixed, changed = self.fix_shape(shape)
                if changed:
                    data["shapes"][i] = fixed
                    changes.append(f"shape[{i}]: campos faltantes añadidos")

        if not changes:
            return False, "sin cambios"

        if not dry_run:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return True, "; ".join(changes)


    def fix_all_jsons(self,labels_dir, images_dir, version, dry_run=False):
        """
        Corrige todos los archivos JSON en labels_dir.
        """
        json_files = sorted(glob.glob(os.path.join(labels_dir, "*.json")))

        if not json_files:
            print(f"No se encontraron archivos JSON en: {labels_dir}")
            return

        total = len(json_files)
        fixed_count = 0

        for json_path in json_files:
            fixed, msg = self.fix_json_file(json_path, images_dir, version, dry_run)
            if fixed:
                fixed_count += 1
                prefix = "[DRY-RUN] " if dry_run else ""
                print(f"{prefix}{os.path.basename(json_path)}: {msg}")

        action = "se corregirían" if dry_run else "corregidos"
        print(f"\nResumen: {fixed_count}/{total} archivos {action}")

