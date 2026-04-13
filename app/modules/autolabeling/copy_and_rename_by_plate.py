from fast_alpr import ALPR
from ultralytics import YOLO
import os
import shutil
import sys
import cv2
from core.task import Task
# Agregar el path del código del proyecto para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from src.ocr_plate import process_image

class CopyAndRenameByPlateTask(Task):
    """
    Task for copying and renaming images based on detected license plates.

    This class extends the base Task class to process images, detect license plates
    using YOLO and ALPR OCR, and copy/rename images based on the detected plate text.
    It also crops and resizes detected plates to a standard size.

    Attributes
    ----------
    plate_detector_model : YOLO
        YOLO model for license plate detection.
    reader_alpr_ocr : ALPR
        ALPR instance for OCR on detected plates.
    valid_ext : tuple
        Valid image file extensions.
    """

    name = "copy_and_rename_by_plate"
    def __init__(self, params):
        """
        Initialize the CopyAndRenameByPlateTask with given parameters.

        Parameters
        ----------
        params : object
            Configuration parameters including OUTPUT_DIR, CROPPED_PLATES_DIR,
            IMAGES_DIR, PLATE_MODEL_PATH, and OUTPUT_DIR_NOREC.
        """
        super().__init__(name="copy_and_rename_by_plate", params=params)
        self.params = params


        os.makedirs(self.params.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.params.CROPPED_PLATES_DIR, exist_ok=True)

        self.valid_ext = (".jpg", ".jpeg", ".png")

        # Inicializar modelos
        print("Cargando modelos...")
        self.plate_detector_model = YOLO(self.params.PLATE_MODEL_PATH)
        self.reader_alpr_ocr = ALPR(
            ocr_model="global-plates-mobile-vit-v2-model",
            detector_model="yolo-v9-t-256-license-plate-end2end",
        )
        print("Modelos cargados correctamente.")


    def get_image_paths(self):
        """
        Get all image paths from the input directory recursively.

        Returns
        -------
        list of str
            List of paths to valid image files.
        """
        image_paths = []
        for root, _, files in os.walk(self.params.IMAGES_DIR):
            for f in files:
                if f.lower().endswith(self.valid_ext):
                    image_paths.append(os.path.join(root, f))
        return image_paths


    def print_statistics(self, stats):
        """
        Print a detailed report of processing statistics.

        Parameters
        ----------
        stats : dict
            Dictionary containing statistics from the processing run.
        """
        print(f"\n{'='*70}")
        print(f"{'ESTADÍSTICAS DEL PROCESAMIENTO':^70}")
        print(f"{'='*70}\n")

        # Resumen general
        print("📊 RESUMEN GENERAL")
        print(f"  Total de imágenes procesadas: {stats['total_images']}")
        print(
            f"  ✓ Detecciones válidas:         {stats['valid_detections']} ({stats['valid_detections']/max(stats['total_images'], 1)*100:.1f}%)"
        )
        print(
            f"  ✗ Detecciones inválidas:       {stats['invalid_detections']} ({stats['invalid_detections']/max(stats['total_images'], 1)*100:.1f}%)"
        )
        print(
            f"  ⚠ Errores de procesamiento:    {stats['errors']} ({stats['errors']/max(stats['total_images'], 1)*100:.1f}%)"
        )

        # Estadísticas de confianza
        if stats["confidences"]:
            avg_conf = sum(stats["confidences"]) / len(stats["confidences"])
            min_conf = min(stats["confidences"])
            max_conf = max(stats["confidences"])
            print(f"\n📈 CONFIANZA DE DETECCIONES")
            print(f"  Promedio:  {avg_conf:.2f}%")
            print(f"  Mínima:    {min_conf}%")
            print(f"  Máxima:    {max_conf}%")

        # Estadísticas de altura
        if stats["heights"]:
            avg_height = sum(stats["heights"]) / len(stats["heights"])
            min_height = min(stats["heights"])
            max_height = max(stats["heights"])
            print(f"\n📏 CALIDAD DE CARACTERES (Altura)")
            print(f"  Promedio:  {avg_height:.2f}")
            print(f"  Mínima:    {min_height}")
            print(f"  Máxima:    {max_height}")

        # Estadísticas de bounding boxes
        print(f"\n🔲 DETECCIÓN DE REGIONES (YOLO)")
        print(
            f"  Con bounding box:    {stats['with_bbox']} ({stats['with_bbox']/max(stats['valid_detections'], 1)*100:.1f}%)"
        )
        print(
            f"  Sin bounding box:    {stats['without_bbox']} ({stats['without_bbox']/max(stats['valid_detections'], 1)*100:.1f}%)"
        )

        # Placas únicas y repetidas
        unique_plates = len(stats["plates_count"])
        repeated_plates = sum(
            1 for count in stats["plates_count"].values() if count > 1)
        total_repetitions = sum(
            count - 1 for count in stats["plates_count"].values() if count > 1
        )

        print(f"\n🔢 ANÁLISIS DE PLACAS")
        print(f"  Placas únicas detectadas:  {unique_plates}")
        print(f"  Placas con repeticiones:   {repeated_plates}")
        print(f"  Total de repeticiones:     {total_repetitions}")

        # Mostrar placas más frecuentes
        if stats["plates_count"]:
            sorted_plates = sorted(
                stats["plates_count"].items(), key=lambda x: x[1], reverse=True
            )
            top_5 = sorted_plates[:5]

            if any(count > 1 for _, count in top_5):
                print(f"\n  🔝 Placas más frecuentes:")
                for plate, count in top_5:
                    if count > 1:
                        print(f"     {plate}: {count} veces")

        # Rutas de salida
        print(f"\n📁 ARCHIVOS GENERADOS")
        print(f"  Imágenes completas:     {self.params.OUTPUT_DIR}")
        print(f"  Placas recortadas (64x256): {self.params.CROPPED_PLATES_DIR}")

        print(f"\n{'='*70}\n")


    def run(self):
        """
        Execute the license plate detection and renaming process.

        Processes all images in the input directory, detects license plates,
        copies images to output directories, and crops plates to standard size.
        Generates detailed statistics at the end.

        Notes
        -----
        Images without valid detections are copied to OUTPUT_DIR_NOREC.
        Plates are cropped and resized to 256x64 pixels.
        """
        stats = {
            "total_images": 0,
            "valid_detections": 0,
            "invalid_detections": 0,
            "errors": 0,
            "plates_count": {},  # Dict para contar repeticiones: {placa: cantidad}
            "confidences": [],
            "heights": [],
            "with_bbox": 0,
            "without_bbox": 0,
        }

        image_paths = self.get_image_paths()
        stats["total_images"] = len(image_paths)

        print(f"Iniciando procesamiento de {stats['total_images']} imágenes...\n")

        for image_path in image_paths:
            try:
                # Llamar directamente al método de reconocimiento
                result = process_image(
                    image_path,
                    self.plate_detector_model,
                    self.reader_alpr_ocr,
                    self.params.OUTPUT_DIR,
                    show_results=False,
                    is_bike=False,
                    return_dict=True,
                )

                if result and result.get("is_valid"):
                    plate = result.get("text", "NO_PLATE")
                    bbox = result.get("bbox")
                    confidence = result.get("confidence", 0)
                    height = result.get("height", 0)

                    # Actualizar estadísticas
                    stats["valid_detections"] += 1
                    stats["confidences"].append(confidence)
                    stats["heights"].append(height)

                    # Contar placas para detectar repeticiones
                    stats["plates_count"][plate] = stats["plates_count"].get(
                        plate, 0) + 1

                    # Copiar imagen completa con nombre de placa
                    #ext = os.path.splitext(image_path)[1]
                    #new_name = f"{plate}{ext}"
                    original_name = os.path.splitext(os.path.basename(image_path))[0]
                    new_name = original_name
                    dest_path = os.path.join(self.params.OUTPUT_DIR, new_name)
                    shutil.copy2(image_path, dest_path)

                    # Recortar y guardar imagen de la placa si se detectó bbox
                    if bbox is not None:
                        stats["with_bbox"] += 1
                        x1, y1, x2, y2 = bbox
                        # Leer imagen original
                        img = cv2.imread(image_path)
                        if img is not None:
                            # Recortar la región de la placa
                            plate_crop = img[y1:y2, x1:x2]
                            # Redimensionar a 64x256
                            plate_resized = cv2.resize(plate_crop, (256, 64))
                            # Guardar imagen recortada
                            #cropped_name = f"{plate}{ext}"
                            cropped_name = original_name
                            cropped_path = os.path.join(
                                self.params.CROPPED_PLATES_DIR, cropped_name)
                            cv2.imwrite(cropped_path, plate_resized)
                            print(f"✓ {image_path}")
                            print(f"  → Placa: {plate} (conf: {confidence}%)")
                            print(f"  → Imagen completa: {dest_path}")
                            print(f"  → Recorte 64x256: {cropped_path}")
                        else:
                            print(
                                f"✓ Procesada (sin recorte): {image_path} -> {dest_path}")
                    else:
                        stats["without_bbox"] += 1
                        print(
                            f"✓ Procesada (sin bbox): {image_path} -> {dest_path}")
                else:
                    stats["invalid_detections"] += 1
                    print(f"✗ No se detectó placa válida en: {image_path}")
                    dest_path = os.path.join(self.params.OUTPUT_DIR_NOREC)
                    shutil.copy2(image_path, dest_path)
            except Exception as e:
                stats["errors"] += 1
                print(f"✗ Error en {image_path}: {e}")

        # Generar reporte de estadísticas
        self.print_statistics(stats)

