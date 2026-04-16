"""Prepara imágenes de placas para entrenamiento OCR a partir de anotaciones LabelMe.

Este módulo recorta placas de imágenes usando JSON de LabelMe y genera un dataset
separado en carpetas train/ y val/.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from app.core.task import Task
from tqdm import tqdm


def get_bbox_from_polygon(points):
    """Extrae el bounding box de un polígono.

    Parameters
    ----------
    points : list
        Lista de puntos [[x1, y1], [x2, y2], ...].

    Returns
    -------
    tuple[int, int, int, int]
        Coordenadas (x_min, y_min, x_max, y_max).
    """
    points = np.array(points)
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    return x_min, y_min, x_max, y_max


def add_padding(x1, y1, x2, y2, img_height, img_width, padding_percent=0.05):
    """Agrega padding al bounding box manteniéndose dentro de la imagen."""
    width = x2 - x1
    height = y2 - y1
    pad_x = int(width * padding_percent)
    pad_y = int(height * padding_percent)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_width, x2 + pad_x)
    y2 = min(img_height, y2 + pad_y)
    return x1, y1, x2, y2


def resize_plate(plate_img, target_height=64, target_width=256):
    """Redimensiona una placa al tamaño objetivo."""
    return cv2.resize(
        plate_img, (target_width, target_height), interpolation=cv2.INTER_AREA
    )


def sanitize_filename(text):
    """Limpia el texto para crear un nombre de archivo válido."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        text = text.replace(char, "")
    text = text.replace(" ", "")
    return text.upper()


def process_image(
    img_path,
    json_path,
    output_dir,
    target_height=64,
    target_width=256,
    padding=0.05,
    mode="auto",
):
    """Procesa una imagen y su JSON para extraer placas recortadas."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error leyendo JSON {json_path}: {e}")
        return False

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"❌ Error cargando imagen {img_path}")
        return False

    img_height, img_width = img.shape[:2]
    shapes = data.get("shapes", [])
    if not shapes:
        print(f"⚠️ No se encontraron anotaciones en {json_path}")
        return False

    plates_processed = 0
    for idx, shape in enumerate(shapes):
        label = shape.get("label", "").lower()
        if "placa" not in label and "plate" not in label and "license" not in label:
            continue

        points = shape.get("points", [])
        if not points:
            continue

        x1, y1, x2, y2 = get_bbox_from_polygon(points)
        x1, y1, x2, y2 = add_padding(
            x1, y1, x2, y2, img_height, img_width, padding
        )
        plate_crop = img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            print(f"⚠️ Recorte vacío en {img_path}")
            continue

        plate_resized = resize_plate(plate_crop, target_height, target_width)
        plate_text = shape.get("description", "").strip()

        if mode == "auto":
            if not plate_text:
                plate_text = f"{Path(img_path).stem}_plate_{idx}"
        elif mode == "manual":
            if not plate_text:
                cv2.imshow("Placa - Ingresa el texto", plate_resized)
                print(f"\n📋 Imagen: {Path(img_path).name}")
                print("   Ingresa el texto de la placa (o presiona Enter para omitir):")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                plate_text = input("   Texto: ").strip().upper()
                if not plate_text:
                    plate_text = f"{Path(img_path).stem}_plate_{idx}"
        elif mode == "sequential":
            plate_text = f"plate_{len(list(Path(output_dir).glob('*.jpg'))):05d}"

        plate_text = sanitize_filename(plate_text)
        output_path = os.path.join(output_dir, f"{plate_text}.jpg")
        if os.path.exists(output_path):
            counter = 1
            while os.path.exists(
                os.path.join(output_dir, f"{plate_text}_{counter}.jpg")
            ):
                counter += 1
            output_path = os.path.join(output_dir, f"{plate_text}_{counter}.jpg")

        cv2.imwrite(output_path, plate_resized)
        plates_processed += 1

    return plates_processed > 0


def process_dataset(
    input_dir,
    output_dir,
    target_height=64,
    target_width=256,
    padding=0.05,
    mode="auto",
    split_ratio=0.8,
):
    """Procesa un directorio completo de imágenes y JSONs para OCR."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(input_path.glob("*.json")))
    if not json_files:
        print(f"❌ No se encontraron archivos JSON en {input_dir}")
        return

    print(f"\n📂 Encontrados {len(json_files)} archivos JSON")
    print(f"📐 Tamaño objetivo: {target_width}x{target_height} pixels")
    print(f"📊 Split: {split_ratio*100:.0f}% train, {(1-split_ratio)*100:.0f}% val")
    print(f"🔤 Modo: {mode}\n")

    split_idx = int(len(json_files) * split_ratio)
    train_count = 0
    val_count = 0
    skipped = 0

    for idx, json_file in enumerate(tqdm(json_files, desc="Procesando")):
        img_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
        img_path = None
        for ext in img_extensions:
            potential_path = json_file.with_suffix(ext)
            if potential_path.exists():
                img_path = potential_path
                break

        if img_path is None:
            print(f"⚠️ No se encontró imagen para {json_file.name}")
            skipped += 1
            continue

        current_output_dir = train_dir if idx < split_idx else val_dir
        success = process_image(
            img_path,
            json_file,
            current_output_dir,
            target_height=target_height,
            target_width=target_width,
            padding=padding,
            mode=mode,
        )

        if success:
            if idx < split_idx:
                train_count += 1
            else:
                val_count += 1
        else:
            skipped += 1

    print("\n" + "=" * 60)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"📁 Train: {train_count} placas guardadas en {train_dir}")
    print(f"📁 Val:   {val_count} placas guardadas en {val_dir}")
    if skipped > 0:
        print(f"⚠️  Omitidas: {skipped} imágenes")
    print(f"\n💾 Total procesado: {train_count + val_count} placas")
    print("=" * 60)


class PrepareImagesToDatasetTask(Task):
    """Tarea para crear un dataset de placas a partir de LabelMe."""

    name = "prepare_images_to_dataset"

    def __init__(self, params):
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        process_dataset(
            input_dir=self.params.get("input_dir"),
            output_dir=self.params.get("output_dir"),
            target_height=int(self.params.get("target_height", 64)),
            target_width=int(self.params.get("target_width", 256)),
            padding=float(self.params.get("padding", 0.05)),
            mode=self.params.get("mode", "auto"),
            split_ratio=float(self.params.get("split_ratio", 0.8)),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepara dataset de placas para entrenamiento de OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Modo automático (usa description del JSON o nombres automáticos)
  python prepare_images_to_dataset.py -i imagenes_para_recortar -o dataset_ocr

  # Modo manual (pide texto para cada placa sin description)
  python prepare_images_to_dataset.py -i imagenes_para_recortar -o dataset_ocr -m manual

  # Modo secuencial (nombres automáticos: plate_00000, plate_00001, ...)
  python prepare_images_to_dataset.py -i imagenes_para_recortar -o dataset_ocr -m sequential

  # Personalizar tamaño y padding
  python prepare_images_to_dataset.py -i imagenes_para_recortar -o dataset_ocr --height 96 --width 384 -p 0.1

  # Cambiar split train/val (70/30 en lugar de 80/20)
  python prepare_images_to_dataset.py -i imagenes_para_recortar -o dataset_ocr -s 0.7
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Directorio con imágenes y archivos JSON de LabelMe",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Directorio de salida (se crearán carpetas train/ y val/)",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Alto objetivo de las placas (default: 64)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Ancho objetivo de las placas (default: 256)",
    )

    parser.add_argument(
        "-p",
        "--padding",
        type=float,
        default=0.05,
        help="Padding a agregar al bbox (0.05 = 5%%, default: 0.05)",
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["auto", "manual", "sequential"],
        default="auto",
        help="Modo de obtención del texto de la placa (default: auto)",
    )

    parser.add_argument(
        "-s",
        "--split",
        type=float,
        default=0.8,
        help="Proporción train/val (default: 0.8 = 80%% train, 20%% val)",
    )

    args = parser.parse_args()

    if not 0.0 < args.split < 1.0:
        print("❌ Error: --split debe estar entre 0 y 1")
        return

    task = PrepareImagesToDatasetTask(
        {
            "input_dir": args.input,
            "output_dir": args.output,
            "target_height": args.height,
            "target_width": args.width,
            "padding": args.padding,
            "mode": args.mode,
            "split_ratio": args.split,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
