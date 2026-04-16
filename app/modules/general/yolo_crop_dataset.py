"""Detecta placas con YOLO, las recorta y prepara un dataset para OCR."""

import argparse
import random
from pathlib import Path

import cv2
from app.core.task import Task
from tqdm import tqdm

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMG_H, IMG_W = 64, 256


def load_model(model_path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Instala ultralytics: pip install ultralytics") from exc
    return YOLO(model_path)


def best_detection(model, img_path, conf_threshold):
    """Retorna el mejor bounding box de la detección YOLO."""
    results = model(str(img_path), verbose=False)[0]
    best = None
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        if best is None or conf > best[4]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            best = (x1, y1, x2, y2, conf)
    return best


def crop_and_resize(img, x1, y1, x2, y2, padding=0.05):
    """Recorta y redimensiona el parche de placa extraído."""
    h, w = img.shape[:2]
    pw = int((x2 - x1) * padding)
    ph = int((y2 - y1) * padding)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def split_and_save(crops, output_dir, split_ratio, seed):
    """Divide los recortes en train y val y guarda los resultados."""
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    shuffled = list(crops)
    random.shuffle(shuffled)

    idx = int(len(shuffled) * split_ratio)
    train, val = shuffled[:idx], shuffled[idx:]

    for name, img in train:
        cv2.imwrite(str(train_dir / name), img)
    for name, img in val:
        cv2.imwrite(str(val_dir / name), img)

    return len(train), len(val)


class YoloCropDatasetTask(Task):
    """Tarea para recortar placas con YOLO y generar dataset OCR."""

    name = "yolo_crop_dataset"

    def __init__(self, params):
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        input_path = Path(self.params.get("input"))
        model_path = self.params.get("model")
        output_path = Path(self.params.get("output"))
        conf = float(self.params.get("conf", 0.5))
        padding = float(self.params.get("padding", 0.05))
        split = float(self.params.get("split", 0.8))
        seed = int(self.params.get("seed", 42))

        if not input_path.exists():
            raise FileNotFoundError(f"No existe: {input_path}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No existe el modelo: {model_path}")

        images = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No se encontraron imágenes en {input_path}")

        model = load_model(model_path)
        crops = []
        no_detection = []

        for img_path in tqdm(images, desc="Detectando placas"):
            img = cv2.imread(str(img_path))
            if img is None:
                no_detection.append(img_path.name)
                continue

            det = best_detection(model, img_path, conf)
            if det is None:
                no_detection.append(img_path.name)
                continue

            x1, y1, x2, y2, _ = det
            cropped = crop_and_resize(img, x1, y1, x2, y2, padding)
            crops.append((img_path.name, cropped))

        if not crops:
            raise RuntimeError("No se detectó ninguna placa. Prueba bajando --conf.")

        train_n, val_n = split_and_save(crops, output_path, split, seed)
        print("\n" + "=" * 50)
        print("✅ DATASET LISTO")
        print("=" * 50)
        print(f"📁 Train : {train_n} imágenes → {output_path}/train/")
        print(f"📁 Val   : {val_n} imágenes → {output_path}/val/")
        print(f"💾 Total : {train_n + val_n} placas recortadas")
        if no_detection:
            print(
                f"⚠️  Sin detección ({len(no_detection)}): {', '.join(no_detection[:10])}"
                + (" ..." if len(no_detection) > 10 else "")
            )
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Recorta placas con YOLO y prepara dataset para OCR"
    )
    parser.add_argument("-i", "--input", required=True, help="Carpeta con imágenes de vehículos")
    parser.add_argument("-m", "--model", required=True, help="Ruta al modelo YOLO (.pt)")
    parser.add_argument("-o", "--output", required=True, help="Carpeta de salida (se crea train/ y val/)")
    parser.add_argument("-c", "--conf", type=float, default=0.5, help="Confianza mínima YOLO (default: 0.5)")
    parser.add_argument("-p", "--padding", type=float, default=0.05, help="Padding al bbox en fracción (default: 0.05)")
    parser.add_argument("-s", "--split", type=float, default=0.8, help="Fracción para train (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducir el split")
    args = parser.parse_args()

    task = YoloCropDatasetTask(
        {
            "input": args.input,
            "model": args.model,
            "output": args.output,
            "conf": args.conf,
            "padding": args.padding,
            "split": args.split,
            "seed": args.seed,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
