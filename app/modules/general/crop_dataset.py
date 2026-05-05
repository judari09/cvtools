"""Recorta regiones de interés (ej: placas) y genera un dataset train/val.

Consolida prepare_images_to_dataset.py (fuente: LabelMe JSON) y
yolo_crop_dataset.py (fuente: modelo YOLO) en un único módulo parametrizable
mediante el campo 'source': 'json' o 'yolo'.
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from app.core.task import Task


# ---------------------------------------------------------------------------
# Helpers compartidos
# ---------------------------------------------------------------------------

def _get_bbox_from_polygon(points):
    pts = np.array(points)
    return (
        int(np.min(pts[:, 0])),
        int(np.min(pts[:, 1])),
        int(np.max(pts[:, 0])),
        int(np.max(pts[:, 1])),
    )


def _add_padding(x1, y1, x2, y2, img_h, img_w, padding=0.05):
    pw = int((x2 - x1) * padding)
    ph = int((y2 - y1) * padding)
    return (
        max(0, x1 - pw),
        max(0, y1 - ph),
        min(img_w, x2 + pw),
        min(img_h, y2 + ph),
    )


def _resize_crop(crop, target_h, target_w):
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _sanitize(text):
    for ch in '<>:"/\\|?* ':
        text = text.replace(ch, "")
    return text.upper()


def _split_and_save(items, output_dir, split_ratio, seed=42):
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    shuffled = list(items)
    random.shuffle(shuffled)
    idx = int(len(shuffled) * split_ratio)

    for name, img in shuffled[:idx]:
        cv2.imwrite(str(train_dir / name), img)
    for name, img in shuffled[idx:]:
        cv2.imwrite(str(val_dir / name), img)

    return len(shuffled[:idx]), len(shuffled[idx:])


# ---------------------------------------------------------------------------
# Fuente JSON
# ---------------------------------------------------------------------------

def _crop_from_json(input_dir, output_dir, target_h, target_w, padding, split_ratio, naming_mode="auto"):
    import json
    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("*.json"))

    if not json_files:
        print(f"No se encontraron archivos JSON en {input_dir}")
        return

    print(f"Encontrados {len(json_files)} archivos JSON | modo: {naming_mode} | split: {split_ratio}")

    crops = []
    skipped = 0

    for idx, json_file in enumerate(tqdm(json_files, desc="Procesando JSON")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error leyendo {json_file.name}: {e}")
            skipped += 1
            continue

        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = json_file.with_suffix(ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]

        for shape_idx, shape in enumerate(data.get("shapes", [])):
            label = shape.get("label", "").lower()
            if not any(kw in label for kw in ("placa", "plate", "license")):
                continue
            points = shape.get("points", [])
            if not points:
                continue

            x1, y1, x2, y2 = _get_bbox_from_polygon(points)
            x1, y1, x2, y2 = _add_padding(x1, y1, x2, y2, img_h, img_w, padding)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized = _resize_crop(crop, target_h, target_w)

            if naming_mode == "sequential":
                name = f"plate_{len(crops):05d}.jpg"
            elif naming_mode == "auto":
                desc = shape.get("description", "").strip()
                text = _sanitize(desc) if desc else f"{json_file.stem}_p{shape_idx}"
                name = f"{text}.jpg"
            else:
                name = f"{json_file.stem}_p{shape_idx}.jpg"

            crops.append((name, resized))

    if not crops:
        print("No se generaron recortes.")
        return

    train_n, val_n = _split_and_save(crops, output_dir, split_ratio)
    print(f"\nTrain: {train_n} | Val: {val_n} | Omitidas: {skipped}")


# ---------------------------------------------------------------------------
# Fuente YOLO
# ---------------------------------------------------------------------------

def _crop_from_yolo(input_dir, output_dir, model_path, conf, target_h, target_w, padding, split_ratio, seed=42):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Instala ultralytics: pip install ultralytics") from exc

    valid_ext = {".jpg", ".jpeg", ".png"}
    input_path = Path(input_dir)
    images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in valid_ext)

    if not images:
        raise FileNotFoundError(f"No se encontraron imágenes en {input_dir}")

    model = YOLO(model_path)
    crops = []
    no_det = []

    for img_path in tqdm(images, desc="Detectando con YOLO"):
        img = cv2.imread(str(img_path))
        if img is None:
            no_det.append(img_path.name)
            continue

        results = model(str(img_path), verbose=False)[0]
        best = None
        for box in results.boxes:
            c = float(box.conf[0])
            if c < conf:
                continue
            if best is None or c > best[4]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best = (x1, y1, x2, y2, c)

        if best is None:
            no_det.append(img_path.name)
            continue

        x1, y1, x2, y2, _ = best
        h, w = img.shape[:2]
        x1, y1, x2, y2 = _add_padding(x1, y1, x2, y2, h, w, padding)
        crop = img[y1:y2, x1:x2]
        resized = _resize_crop(crop, target_h, target_w)
        crops.append((img_path.name, resized))

    if not crops:
        raise RuntimeError("No se detectó ningún objeto. Prueba bajando --conf.")

    train_n, val_n = _split_and_save(crops, output_dir, split_ratio, seed=seed)
    print(f"\nTrain: {train_n} | Val: {val_n}")
    if no_det:
        print(f"Sin detección ({len(no_det)}): {', '.join(no_det[:10])}" + (" ..." if len(no_det) > 10 else ""))


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class CropDatasetTask(Task):
    """Tarea unificada para recortar regiones y generar datasets train/val.

Example YAML:
```yaml
- name: crop_dataset
  params:
    input_dir: /ruta/entrada
    output_dir: /ruta/salida
    source: json        # "json" (LabelMe) o "yolo" (modelo YOLO)
    model_path: null    # requerido si source=yolo
    conf: 0.5           # umbral de confianza (solo source=yolo)
    resize_w: 256
    resize_h: 64
    padding: 0.05
    split_ratio: 0.8
    naming_mode: auto   # auto|sequential|stem (solo source=json)
    seed: 42            # semilla para reproducir split (solo source=yolo)
```"""

    name = "crop_dataset"

    def __init__(self, params):
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        source = self.params.get("source", "json")
        input_dir = self.params.get("input_dir")
        output_dir = self.params.get("output_dir")
        target_h = int(self.params.get("resize_h", 64))
        target_w = int(self.params.get("resize_w", 256))
        padding = float(self.params.get("padding", 0.05))
        split_ratio = float(self.params.get("split_ratio", 0.8))

        if source == "json":
            _crop_from_json(
                input_dir=input_dir,
                output_dir=output_dir,
                target_h=target_h,
                target_w=target_w,
                padding=padding,
                split_ratio=split_ratio,
                naming_mode=self.params.get("naming_mode", "auto"),
            )
        elif source == "yolo":
            _crop_from_yolo(
                input_dir=input_dir,
                output_dir=output_dir,
                model_path=self.params.get("model_path"),
                conf=float(self.params.get("conf", 0.5)),
                target_h=target_h,
                target_w=target_w,
                padding=padding,
                split_ratio=split_ratio,
                seed=int(self.params.get("seed", 42)),
            )
        else:
            raise ValueError(f"source debe ser 'json' o 'yolo', no '{source}'")


def main():
    parser = argparse.ArgumentParser(description="Recorta regiones y genera dataset train/val")
    parser.add_argument("-i", "--input-dir", required=True, help="Carpeta de entrada")
    parser.add_argument("-o", "--output-dir", required=True, help="Carpeta de salida")
    parser.add_argument("--source", choices=["json", "yolo"], default="json",
                        help="Fuente de anotaciones: json (LabelMe) o yolo (modelo YOLO)")
    parser.add_argument("--model-path", default=None, help="Ruta al modelo YOLO (requerido si source=yolo)")
    parser.add_argument("--conf", type=float, default=0.5, help="Umbral de confianza YOLO")
    parser.add_argument("--resize-w", type=int, default=256, help="Ancho del recorte")
    parser.add_argument("--resize-h", type=int, default=64, help="Alto del recorte")
    parser.add_argument("--padding", type=float, default=0.05, help="Padding al bbox")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Fracción train")
    parser.add_argument("--naming-mode", choices=["auto", "sequential", "stem"], default="auto",
                        help="Modo de nomenclatura (solo source=json)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad")
    args = parser.parse_args()

    task = CropDatasetTask({
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "source": args.source,
        "model_path": args.model_path,
        "conf": args.conf,
        "resize_w": args.resize_w,
        "resize_h": args.resize_h,
        "padding": args.padding,
        "split_ratio": args.split_ratio,
        "naming_mode": args.naming_mode,
        "seed": args.seed,
    })
    task.run()


if __name__ == "__main__":
    main()
