import os
import json
import cv2
import albumentations as A
import glob
from collections import Counter
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task


# ---------------------------------------------------------------------------
# Perfiles de augmentación
# ---------------------------------------------------------------------------
# Se definen 3 perfiles base. El número de copias aplicado a cada imagen se
# ajusta dinámicamente según la frecuencia de sus clases en el dataset:
# las imágenes que contienen clases poco frecuentes reciben más copias para
# balancear la distribución y evitar sesgo hacia las clases dominantes.
#
# Escala de copias por perfil según frecuencia relativa de la clase:
#   Tier ALTO    (clase frecuente):  leve×2  mod×2  agr×1  → 5  copias
#   Tier MEDIO   (clase moderada):   leve×3  mod×3  agr×2  → 8  copias
#   Tier BAJO    (clase escasa):     leve×5  mod×4  agr×3  → 12 copias
# ---------------------------------------------------------------------------
class AlbumentationsForYolosegTask(Task):
    """
    Task for applying data augmentation to YOLO segmentation datasets using Albumentations.

    This class extends the base Task class to perform image and polygon augmentation
    specifically tailored for YOLO segmentation models. It applies tiered augmentation
    strategies based on class frequency to balance the dataset, generating multiple
    augmented versions of each image with updated polygon coordinates.

    Attributes
    ----------
    KP_PARAMS : albumentations.KeypointParams
        Parameters for keypoint handling in augmentations.
    """

    name = "albumentations_for_yoloseg"
    def __init__(self, params):
        """
        Initialize the AlbumentationsForYolosegTask with given parameters.

        Parameters
        ----------
        params : object
            Configuration parameters for the task, including input/output directories,
            augmentation thresholds (UMBRAL_BAJO, UMBRAL_MEDIO), and other settings.
        """
        super().__init__(name="albumentations_for_yoloseg", params=params)

        self.KP_PARAMS = A.KeypointParams(format="xy", remove_invisible=False)

        # Umbrales para clasificar la frecuencia de una clase (fracción del total).
        # Si la clase más rara tiene ≤ UMBRAL_BAJO del total de muestras → Tier BAJO.
        #self.params.UMBRAL_BAJO  = 0.20   # menos del 20 % del total → escasa
        #self.params.UMBRAL_MEDIO = 0.40   # entre 20 % y 40 %        → moderada
        # por encima del 40 %                             → frecuente


    def build_transforms(self):
        """
        Build and return a dictionary of reusable augmentation transforms.

        Returns
        -------
        dict
            Dictionary containing three augmentation profiles:
            - 'ligero': Light augmentations
            - 'moderado': Moderate augmentations
            - 'agresivo': Aggressive augmentations
            Each is an albumentations.Compose object with keypoint parameters.
        """
        # -- PERFIL LIGERO -------------------------------------------------------
        ligero = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.6),
            A.CLAHE(clip_limit=3.0, p=0.4),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.0), p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
            A.Blur(blur_limit=3, p=0.4),
            A.ImageCompression(quality_range=(60, 90), p=0.5),
        ], keypoint_params=self.KP_PARAMS)

        # -- PERFIL MODERADO -----------------------------------------------------
        moderado = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=12, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=8,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.4),
            A.RandomBrightnessContrast(p=0.6),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2,
                        shadow_roi=(0, 0.3, 1, 1), p=0.5),
            A.RandomFog(fog_coef_range=(0.05, 0.25), p=0.3),
            A.HueSaturationValue(p=0.5),
            A.MotionBlur(blur_limit=5, p=0.4),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
            A.ImageCompression(quality_range=(50, 80), p=0.4),
        ], keypoint_params=self.KP_PARAMS)

        # -- PERFIL AGRESIVO -----------------------------------------------------
        agresivo = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.Affine(shear=(-6, 6), p=0.4),
            A.Perspective(scale=(0.03, 0.07), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3,
                        shadow_roi=(0, 0.2, 1, 1), p=0.6),
            A.RandomFog(fog_coef_range=(0.1, 0.35), p=0.35),
            A.RandomRain(p=0.2),
            A.RandomSunFlare(src_radius=80, p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.4),
            A.MotionBlur(blur_limit=7, p=0.5),
            A.Defocus(radius=(1, 3), p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=35, val_shift_limit=25, p=0.6),
            A.GaussNoise(var_limit=(20.0, 60.0), p=0.5),
            A.ImageCompression(quality_range=(40, 70), p=0.5),
            A.CoarseDropout(num_holes_range=(2, 6),
                            hole_height_range=(10, 45),
                            hole_width_range=(10, 45), p=0.4),
        ], keypoint_params=self.KP_PARAMS)

        return {"ligero": ligero, "moderado": moderado, "agresivo": agresivo}


    # ---------------------------------------------------------------------------
    # Análisis de frecuencia de clases
    # ---------------------------------------------------------------------------

    def contar_clases(self, label_folder, image_files):
        """
        Count how many images contain each class at least once.

        Parameters
        ----------
        label_folder : str
            Path to the folder containing JSON label files.
        image_files : list of str
            List of paths to image files.

        Returns
        -------
        collections.Counter
            Counter object with class labels as keys and the number of images
            containing each class as values.
        """
        conteo = Counter()
        for image_path in image_files:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(label_folder, base_name + ".json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                clases_en_imagen = {shape["label"] for shape in data.get("shapes", [])}
                for clase in clases_en_imagen:
                    conteo[clase] += 1
            except Exception:
                pass
        return conteo


    def calcular_tier(self, clases_imagen, conteo_global, total_imagenes):
        """
        Determine the augmentation tier based on the rarest class in the image.

        Parameters
        ----------
        clases_imagen : set
            Set of class labels present in the image.
        conteo_global : collections.Counter
            Global count of images per class.
        total_imagenes : int
            Total number of images in the dataset.

        Returns
        -------
        str
            Augmentation tier: "bajo" (low), "medio" (medium), or "alto" (high).
        """
        if not clases_imagen or total_imagenes == 0:
            return "alto"

        # Fracción más baja entre las clases de esta imagen
        fraccion_min = min(
            conteo_global.get(c, 0) / total_imagenes
            for c in clases_imagen
        )

        if fraccion_min <= self.params.UMBRAL_BAJO:
            return "bajo"
        elif fraccion_min <= self.params.UMBRAL_MEDIO:
            return "medio"
        else:
            return "alto"


    def copias_por_tier(self, tier, transforms):
        """
        Return the list of (transform, suffix, num_copies) for the given tier.

        Parameters
        ----------
        tier : str
            Augmentation tier: "alto", "medio", or "bajo".
        transforms : dict
            Dictionary of transform objects from build_transforms().

        Returns
        -------
        list of tuple
            List of (transform, suffix, num_copies) for each augmentation level.
        """
        planes = {
            "alto":  [(transforms["ligero"],   "_aug_leve", 2),
                    (transforms["moderado"], "_aug_mod",  2),
                    (transforms["agresivo"], "_aug_agr",  1)],
            "medio": [(transforms["ligero"],   "_aug_leve", 3),
                    (transforms["moderado"], "_aug_mod",  3),
                    (transforms["agresivo"], "_aug_agr",  2)],
            "bajo":  [(transforms["ligero"],   "_aug_leve", 5),
                    (transforms["moderado"], "_aug_mod",  4),
                    (transforms["agresivo"], "_aug_agr",  3)],
        }
        return planes[tier]


    # ---------------------------------------------------------------------------
    # I/O helpers
    # ---------------------------------------------------------------------------

    def load_labelme(self, json_path):
        """
        Load LabelMe JSON data from a file.

        Parameters
        ----------
        json_path : str
            Path to the JSON file.

        Returns
        -------
        dict
            Parsed JSON data from the file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_labelme(self, data, json_path):
        """
        Save LabelMe JSON data to a file.

        Parameters
        ----------
        data : dict
            JSON data to save.
        json_path : str
            Path where the JSON file will be saved.
        """
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def clip_keypoints(self, keypoints, width, height):
        """
        Clip keypoints to the image boundaries after geometric transformations.

        Parameters
        ----------
        keypoints : list of tuple
            List of (x, y) keypoints.
        width : int
            Image width.
        height : int
            Image height.

        Returns
        -------
        list of list
            Clipped keypoints as [[x, y], ...].
        """
        clipped = []
        for x, y in keypoints:
            x = max(0.0, min(float(x), width - 1))
            y = max(0.0, min(float(y), height - 1))
            clipped.append([x, y])
        return clipped


    # ---------------------------------------------------------------------------
    # Núcleo de augmentación
    # ---------------------------------------------------------------------------

    def augment_image_and_labels(self, image_path, json_path, output_image_dir, output_label_dir,
                                transform, suffix, num_copias):
        """
        Apply the given transform exactly num_copias times to the image and labels.

        Saves each result as PNG + JSON in the specified output directories.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        json_path : str
            Path to the input JSON label file.
        output_image_dir : str
            Directory to save augmented images.
        output_label_dir : str
            Directory to save augmented labels.
        transform : albumentations.Compose
            Augmentation transform to apply.
        suffix : str
            Suffix for naming augmented files.
        num_copias : int
            Number of augmented copies to generate.

        Returns
        -------
        int
            Number of successfully generated augmentations.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] No se pudo leer la imagen: {image_path}")
                return 0

            data = self.load_labelme(json_path)

            if not data.get("shapes"):
                print(f"[ADVERTENCIA] Sin shapes en {json_path}, se omite.")
                return 0

            polygons           = [shape["points"] for shape in data["shapes"]]
            labels             = [shape["label"]  for shape in data["shapes"]]
            keypoints          = [tuple(p) for poly in polygons for p in poly]
            keypoints_per_poly = [len(poly) for poly in polygons]

            base = os.path.splitext(os.path.basename(image_path))[0]

            for i in range(num_copias):
                aug     = transform(image=image, keypoints=keypoints)
                aug_img = aug["image"]
                aug_kps = self.clip_keypoints(aug["keypoints"], aug_img.shape[1], aug_img.shape[0])

                new_polygons, idx = [], 0
                for n in keypoints_per_poly:
                    new_polygons.append([[float(x), float(y)] for x, y in aug_kps[idx:idx + n]])
                    idx += n

                new_shapes = []
                for label, poly in zip(labels, new_polygons):
                    new_shapes.append({
                        "label":       label,
                        "points":      poly,
                        "group_id":    None,
                        "description": "",
                        "shape_type":  "polygon",
                        "flags":       {},
                        "mask":        None,
                    })

                new_data                    = data.copy()
                new_data["shapes"]          = new_shapes
                new_data["imageHeight"], new_data["imageWidth"] = aug_img.shape[:2]
                new_data["imageData"]       = None

                nombre        = f"{base}{suffix}_{i}"
                new_img_path  = os.path.join(output_image_dir, nombre + ".png")
                new_json_path = os.path.join(output_label_dir, nombre + ".json")
                new_data["imagePath"] = os.path.relpath(new_img_path, output_label_dir)

                cv2.imwrite(new_img_path, aug_img)
                self.save_labelme(new_data, new_json_path)
                print(f"  [OK] {nombre}")

            return num_copias

        except Exception as e:
            print(f"[ERROR] Procesando {json_path}: {e}")
            return 0


    # ---------------------------------------------------------------------------
    # Procesamiento por carpeta
    # ---------------------------------------------------------------------------

    def process_folder(self, image_folder, label_folder, output_image_dir, output_label_dir):
        """
        Process all images in the folder, analyze class frequency, and assign more
        augmentation copies to images containing rare classes.

        Augmentation tiers:
        HIGH (class ≥ 40% of total) → 5 copies per image
        MEDIUM (class 20–40%)       → 8 copies per image
        LOW (class ≤ 20%)           → 12 copies per image

        Parameters
        ----------
        image_folder : str
            Path to the input images directory.
        label_folder : str
            Path to the input labels directory.
        output_image_dir : str
            Path to the output images directory.
        output_label_dir : str
            Path to the output labels directory.
        """
        os.makedirs(output_image_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)

        extensiones = ("*.png", "*.jpg", "*.jpeg")
        image_files = []
        for ext in extensiones:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))

        if not image_files:
            print(f"[ADVERTENCIA] No se encontraron imágenes en {image_folder}")
            return

        # --- Paso 1: analizar frecuencia de clases en todo el dataset -----------
        print("Analizando distribución de clases...")
        conteo_global  = self.contar_clases(label_folder, image_files)
        total_imagenes = len(image_files)

        print(f"\nDistribución de clases ({total_imagenes} imágenes totales):")
        for clase, cnt in sorted(conteo_global.items(), key=lambda x: x[1]):
            fraccion = cnt / total_imagenes
            tier_cls = ("BAJO" if fraccion <= self.params.UMBRAL_BAJO
                        else "MEDIO" if fraccion <= self.params.UMBRAL_MEDIO
                        else "ALTO")
            print(f"  {clase:<25} {cnt:>4} imágenes ({fraccion*100:.1f}%)  → tier {tier_cls}")
        print()

        # --- Paso 2: construir transforms una sola vez --------------------------
        transforms = self.build_transforms()

        # --- Paso 3: augmentar cada imagen con su tier correspondiente ----------
        total_ok, total_err = 0, 0
        tier_stats = Counter()

        for image_path in image_files:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(label_folder, base_name + ".json")

            if not os.path.exists(json_path):
                print(f"[ADVERTENCIA] Sin JSON para {base_name}, se omite.")
                continue

            # Determinar tier según las clases presentes en esta imagen
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                clases_imagen = {shape["label"] for shape in data.get("shapes", [])}
            except Exception as e:
                print(f"[ERROR] Leyendo {json_path}: {e}")
                continue

            tier = self.calcular_tier(clases_imagen, conteo_global, total_imagenes)
            plan = self.copias_por_tier(tier, transforms)
            tier_stats[tier] += 1

            copias_totales = sum(n for _, _, n in plan)
            print(f"Procesando [{tier.upper():>5} | {copias_totales} copias]: {base_name}"
                f"  clases={sorted(clases_imagen)}")

            for transform, suffix, num_copias in plan:
                generadas = self.augment_image_and_labels(
                    image_path, json_path,
                    output_image_dir, output_label_dir,
                    transform, suffix, num_copias
                )
                total_ok  += generadas
                total_err += (num_copias - generadas)

        # --- Resumen final ------------------------------------------------------
        print(f"\n{'='*55}")
        print("Augmentación completada")
        print(f"  Imágenes procesadas : {sum(tier_stats.values())}")
        print(f"  Tier ALTO  (×5)     : {tier_stats['alto']} imágenes")
        print(f"  Tier MEDIO (×8)     : {tier_stats['medio']} imágenes")
        print(f"  Tier BAJO  (×12)    : {tier_stats['bajo']} imágenes")
        print(f"  Total generadas     : {total_ok}")
        print(f"  Errores             : {total_err}")
        print(f"{'='*55}")

    def run(self):
        """
        Execute the augmentation process for YOLO segmentation datasets.

        This method calls process_folder with the input and output directories
        specified in the task parameters.
        """
        self.process_folder(
            self.params.input_images_dir,
            self.params.input_labels_dir,
            self.params.output_images_dir,
            self.params.output_labels_dir
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply Albumentations augmentations to a YOLO segmentation dataset."
    )
    parser.add_argument(
        "--input-images-dir",
        required=True,
        help="Input images folder.",
    )
    parser.add_argument(
        "--input-labels-dir",
        required=True,
        help="Input LabelMe JSON folder.",
    )
    parser.add_argument(
        "--output-images-dir",
        required=True,
        help="Output folder for augmented images.",
    )
    parser.add_argument(
        "--output-labels-dir",
        required=True,
        help="Output folder for generated JSON label files.",
    )
    args = parser.parse_args()
    params = argparse.Namespace(
        input_images_dir=args.input_images_dir,
        input_labels_dir=args.input_labels_dir,
        output_images_dir=args.output_images_dir,
        output_labels_dir=args.output_labels_dir,
    )
    AlbumentationsForYolosegTask(params).run()
