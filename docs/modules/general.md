# Herramientas generales

Utilidades para preparación, validación y organización de datasets de visión por computadora.

---

## check_integrity_dataset

**Clase:** `CheckIntegrityDatasetTask`

Verifica que cada imagen tenga su etiqueta `.txt` correspondiente y reporta archivos faltantes o corruptos.

### Parámetros YAML

```yaml
- name: check_integrity_dataset
  params:
    images_dir: data/images
    labels_dir: data/labels
    output_window: false
```

---

## check_model_classes

**Clase:** `CheckModelClassesTask`

Muestra las clases registradas en uno o varios modelos YOLO.

### Parámetros YAML

```yaml
- name: check_model_classes
  params:
    models:
      - yolov8n.pt
```

---

## check_sizes

**Clase:** `CheckSizesTask`

Analiza y reporta las dimensiones de las imágenes en una carpeta.

### Parámetros YAML

```yaml
- name: check_sizes
  params:
    folder_path: data/images
    extensions:
      - .jpg
      - .png
```

---

## crop_dataset

**Clase:** `CropDatasetTask`

Recorta regiones de interés (p.ej. placas) y genera un dataset train/val. Unifica `prepare_images_to_dataset` y `yolo_crop_dataset` en un único módulo configurable mediante el campo `source`.

### Parámetros YAML

```yaml
- name: crop_dataset
  params:
    input_dir: data/raw
    output_dir: data/cropped
    source: json          # "json" (LabelMe) o "yolo" (modelo YOLO)
    model_path: null      # requerido si source=yolo
    conf: 0.5             # umbral de confianza (solo source=yolo)
    resize_w: 256
    resize_h: 64
    padding: 0.05
    split_ratio: 0.8
    naming_mode: auto     # auto|sequential|stem (solo source=json)
    seed: 42              # semilla para split (solo source=yolo)
```

---

## crop_images

**Clase:** `CropImagesTask`

Recorta un área fija definida por coordenadas en todas las imágenes de una carpeta.

### Parámetros YAML

```yaml
- name: crop_images
  params:
    input_folder: data/images
    output_folder: data/cropped
    x1: 0
    y1: 0
    x2: 640
    y2: 360
```

---

## dataset_builder

**Clase:** `DatasetBuilderTask`

Construye un dataset copiando pares imagen+etiqueta a una carpeta destino. Unifica `imagenes_db` y `move_to_comb`: con `filter_empty: false` copia todas las imágenes con etiqueta; con `filter_empty: true` (por defecto) omite además las etiquetas vacías.

### Parámetros YAML

```yaml
- name: dataset_builder
  params:
    source_images: data/images
    source_labels: data/labels
    destination: data/dataset
    filter_empty: true
```

### CLI

```bash
python -m app.modules.general.dataset_builder \
  --source-images data/images \
  --source-labels data/labels \
  --destination data/dataset
```

---

## extract_frames

**Clase:** `ExtractFramesTask`

Extrae frames de un video a una carpeta de salida con paso configurable.

### Parámetros YAML

```yaml
- name: extract_frames
  params:
    video_path: video.mp4
    output_folder: data/frames
    frame_step: 5
```

---

## imagenes_db

**Clase:** `ImagenesDbTask`

Mueve imágenes que tienen etiqueta TXT asociada a una carpeta destino.

!!! tip
    Para casos más complejos considera usar [`dataset_builder`](#dataset_builder), que consolida este módulo y `move_to_comb`.

### Parámetros YAML

```yaml
- name: imagenes_db
  params:
    source_folder: data/raw
    destination_folder: data/with_labels
```

---

## move_to_comb

**Clase:** `MoveToCombTask`

Copia pares imagen+etiqueta (omitiendo etiquetas vacías) a una carpeta combinada.

!!! tip
    Para casos más complejos considera usar [`dataset_builder`](#dataset_builder).

### Parámetros YAML

```yaml
- name: move_to_comb
  params:
    image_folder: data/images
    label_folder: data/labels
    destination_folder: data/combined
```

---

## prepare_images_to_dataset

**Clase:** `PrepareImagesToDatasetTask`

Crea un dataset de regiones recortadas a partir de anotaciones LabelMe (fuente JSON).

!!! tip
    Considera usar [`crop_dataset`](#crop_dataset) con `source: json`, que unifica este módulo y `yolo_crop_dataset`.

### Parámetros YAML

```yaml
- name: prepare_images_to_dataset
  params:
    input_dir: data/labelme
    output_dir: data/dataset
    target_width: 256
    target_height: 64
    split_ratio: 0.8
    padding: 0.05
    mode: auto
```

---

## resize_folder

**Clase:** `ResizeFolderTask`

Analiza y redimensiona imágenes de una carpeta a las dimensiones especificadas.

### Parámetros YAML

```yaml
- name: resize_folder
  params:
    source_folder: data/images
    destination_folder: data/resized
    width: 640
    height: 640
    do_resize: true
```

---

## separar_train_val

**Clase:** `SepararTrainValTask`

Divide un conjunto de datos en carpetas de entrenamiento y validación según un ratio configurable.

### Parámetros YAML

```yaml
- name: separar_train_val
  params:
    source_folder: data/combined
    train_folder: data/train
    val_folder: data/val
    split_ratio: 0.8
```

---

## unir_videos

**Clase:** `UnirVideosTask`

Une múltiples archivos de video en un único archivo de salida.

### Parámetros YAML

```yaml
- name: unir_videos
  params:
    source_folder: data/videos
    output_path: data/merged.mp4
    target_width: 1920
    target_height: 1080
```

---

## visualize_dataset

**Clase:** `VisualizeDatasetTask`

Visualiza y analiza datasets mostrando muestras de imágenes con sus etiquetas.

### Parámetros YAML

```yaml
- name: visualize_dataset
  params:
    dataset_dir: data/dataset
    num_samples: 10
    random_sample: true
    show_train: true
    show_val: true
```

---

## yolo_crop_dataset

**Clase:** `YoloCropDatasetTask`

Recorta regiones detectadas con un modelo YOLO y genera un dataset OCR con split train/val.

!!! tip
    Considera usar [`crop_dataset`](#crop_dataset) con `source: yolo`, que unifica este módulo y `prepare_images_to_dataset`.

### Parámetros YAML

```yaml
- name: yolo_crop_dataset
  params:
    input: data/images
    model: yolov8n.pt
    output: data/cropped
    conf: 0.5
    padding: 0.05
    split: 0.8
    seed: 42
```
