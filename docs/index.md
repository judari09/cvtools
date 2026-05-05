# CVTools

Suite de herramientas de visión por computadora para preparación de datasets, aumento de datos y etiquetado automático. Arquitectura modular basada en tareas que soporta pipelines YAML y ejecución directa por CLI.

## Módulos disponibles

| Módulo | Descripción |
|--------|-------------|
| [Aumentación de datos](modules/albumentations.md) | Augmentación con Albumentations para detección y segmentación YOLO |
| [Auto-etiquetado](modules/autolabeling.md) | Etiquetado automático con YOLO, SAM y OCR |
| [Herramientas generales](modules/general.md) | Utilidades de dataset: split, crop, resize, integridad |
| [Procesamiento JSON](modules/json.md) | Conversión y limpieza de anotaciones LabelMe |

## Instalación

```bash
git clone <repository-url>
cd cvtools
pip install -e .
```

Para módulos OCR (PaddleOCR, Fast-ALPR):

```bash
pip install -e ".[ocr]"
```

## Uso en modo pipeline (YAML)

Define las tareas en un archivo YAML y ejecuta:

```yaml
tasks:
  - name: separar_train_val
    params:
      source_folder: data/dataset
      train_folder: data/train
      val_folder: data/val
      split_ratio: 0.8

  - name: albumentations_for_yolo
    params:
      input_images_dir: data/train/images
      input_labels_dir: data/train/labels
      output_images_dir: data/aug/images
      output_labels_dir: data/aug/labels
```

```bash
python main.py
```

## Uso por CLI

Cada módulo puede ejecutarse directamente:

```bash
# Convertir anotaciones JSON a formato YOLO
python -m app.modules.json.json2txt --input-dir labels/json --output-dir labels/txt

# Separar dataset en train/val
python -m app.modules.general.separar_train_val --source-folder data/combined --train-folder data/train --val-folder data/val

# Auto-etiquetar imágenes con YOLO
python -m app.modules.autolabeling.auto_label_labelme --models yolov8n-seg.pt --input images --output labels/json
```

## Estructura del proyecto

```
cvtools/
├── main.py                 # Punto de entrada para pipelines YAML
├── config_example.yaml     # Ejemplo de configuración
├── app/
│   ├── core/               # Arquitectura base (Task, registry, executor, pipeline)
│   └── modules/
│       ├── albumentations/ # Aumentación de datos
│       ├── autolabeling/   # Etiquetado automático
│       ├── general/        # Utilidades generales
│       └── json/           # Procesamiento de anotaciones JSON
```
