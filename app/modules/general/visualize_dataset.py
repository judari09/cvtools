"""Visualiza y analiza un dataset de placas OCR procesado."""

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from app.core.task import Task


def visualize_dataset(dataset_dir, num_samples=16, random_sample=True):
    """Muestra una grilla con imágenes de un dataset."""
    dataset_path = Path(dataset_dir)
    image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))

    if not image_files:
        print(f"❌ No se encontraron imágenes en {dataset_dir}")
        return

    print(f"📂 Encontradas {len(image_files)} placas en {dataset_dir}")
    if random_sample:
        samples = random.sample(image_files, min(num_samples, len(image_files)))
    else:
        samples = image_files[:num_samples]

    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(samples):
        row = idx // cols
        col = idx % cols
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        axes[row, col].set_title(img_path.stem, fontsize=8, pad=2)

    for idx in range(len(samples), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.suptitle(f"Dataset: {dataset_path.name}", fontsize=14, y=1.00)
    plt.show()


def get_dataset_stats(train_dir, val_dir):
    """Muestra estadísticas básicas del dataset train/val."""
    train_path = Path(train_dir)
    val_path = Path(val_dir)
    train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
    val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))

    print("\n" + "=" * 60)
    print("📊 ESTADÍSTICAS DEL DATASET")
    print("=" * 60)
    print(f"📁 Train: {len(train_images)} placas")
    print(f"📁 Val:   {len(val_images)} placas")
    print(f"💾 Total: {len(train_images) + len(val_images)} placas")

    if train_images:
        sample_img = cv2.imread(str(train_images[0]))
        height, width = sample_img.shape[:2]
        print(f"📐 Dimensiones: {width}x{height} pixels")

    print("=" * 60 + "\n")


class VisualizeDatasetTask(Task):
    """Tarea para visualizar y analizar datasets de placas OCR.

Example YAML:
```yaml
- name: visualize_dataset
  params:
    dataset_dir: <value>
    num_samples: <value>
    random_sample: <value>
    show_train: <value>
    show_val: <value>
```

Example YAML:
```yaml
- name: visualize_dataset
  params:
    dataset_dir: <value>
    num_samples: <value>
    random_sample: <value>
    show_train: <value>
    show_val: <value>
```

Example YAML:
```yaml
- name: visualize_dataset
  params:
    dataset_dir: <value>
    num_samples: <value>
    random_sample: <value>
    show_train: <value>
    show_val: <value>
```"""

    name = "visualize_dataset"

    def __init__(self, params):
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        dataset_dir = self.params.get("dataset_dir")
        num_samples = int(self.params.get("num_samples", 16))
        random_sample = bool(self.params.get("random_sample", True))

        get_dataset_stats(
            Path(dataset_dir) / "train",
            Path(dataset_dir) / "val",
        )

        if self.params.get("show_train", True):
            visualize_dataset(dataset_dir + "/train", num_samples, random_sample)
        if self.params.get("show_val", True):
            visualize_dataset(dataset_dir + "/val", num_samples, random_sample)


def main():
    parser = argparse.ArgumentParser(description="Visualiza el dataset de placas procesado")
    parser.add_argument("-d", "--dataset", required=True, help="Directorio del dataset base")
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=16,
        help="Número de muestras a mostrar (default: 16)",
    )
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Mostrar las primeras imágenes en lugar de aleatorias",
    )
    parser.add_argument("--train", action="store_true", help="Mostrar solo train")
    parser.add_argument("--val", action="store_true", help="Mostrar solo val")
    args = parser.parse_args()

    show_train = args.train or (not args.train and not args.val)
    show_val = args.val or (not args.train and not args.val)

    task = VisualizeDatasetTask(
        {
            "dataset_dir": args.dataset,
            "num_samples": args.num_samples,
            "random_sample": not args.no_random,
            "show_train": show_train,
            "show_val": show_val,
        }
    )
    task.run()


if __name__ == "__main__":
    main()