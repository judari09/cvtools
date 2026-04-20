# General Tools

This page documents the `General Tools` available in CVTools.

## check_integrity_dataset

**Class:** `CheckIntegrityDatasetTask`

Tarea para verificar integridad de imágenes y etiquetas YOLO.

    Esta tarea revisa que cada imagen en un directorio tenga su etiqueta .txt
    correspondiente en el directorio de etiquetas.

Example YAML:
    ```yaml
- name: check_integrity_dataset
  params:
    images_dir: <value>
    labels_dir: <value>
    output_window: <value>
    ```

### YAML example

    ```yaml
- name: check_integrity_dataset
  params:
    images_dir: <value>
    labels_dir: <value>
    output_window: <value>
    ```

## check_model_classes

**Class:** `CheckModelClassesTask`

Tarea para mostrar las clases de modelos YOLO.

Example YAML:
    ```yaml
- name: check_model_classes
  params:
    models: <value>
    ```

### YAML example

    ```yaml
- name: check_model_classes
  params:
    models: <value>
    ```

## check_sizes

**Class:** `CheckSizesTask`

Tarea para comprobar el tamaño de imágenes en un directorio.

Example YAML:
    ```yaml
- name: check_sizes
  params:
    folder_path: <value>
    extensions: <value>
    ```

### YAML example

    ```yaml
- name: check_sizes
  params:
    folder_path: <value>
    extensions: <value>
    ```

## crop_images

**Class:** `CropImagesTask`

Tarea para recortar imágenes en un área fija.

Example YAML:
    ```yaml
- name: crop_images
  params:
    input_folder: <value>
    output_folder: <value>
    x1: <value>
    y1: <value>
    x2: <value>
    y2: <value>
    ```

### YAML example

    ```yaml
- name: crop_images
  params:
    input_folder: <value>
    output_folder: <value>
    x1: <value>
    y1: <value>
    x2: <value>
    y2: <value>
    ```

## extract_frames

**Class:** `ExtractFramesTask`

Tarea para extraer frames de un video.

Example YAML:
    ```yaml
- name: extract_frames
  params:
    video_path: <value>
    output_folder: <value>
    frame_step: <value>
    ```

### YAML example

    ```yaml
- name: extract_frames
  params:
    video_path: <value>
    output_folder: <value>
    frame_step: <value>
    ```

## imagenes_db

**Class:** `ImagenesDbTask`

Tarea para mover imágenes que tienen etiquetas TXT asociadas.

Example YAML:
    ```yaml
- name: imagenes_db
  params:
    source_folder: <value>
    destination_folder: <value>
    ```

### YAML example

    ```yaml
- name: imagenes_db
  params:
    source_folder: <value>
    destination_folder: <value>
    ```

## move_to_comb

**Class:** `MoveToCombTask`

Tarea para copiar imágenes y etiquetas emparejadas a una carpeta combinada.

Example YAML:
    ```yaml
- name: move_to_comb
  params:
    image_folder: <value>
    label_folder: <value>
    destination_folder: <value>
    ```

### YAML example

    ```yaml
- name: move_to_comb
  params:
    image_folder: <value>
    label_folder: <value>
    destination_folder: <value>
    ```

## prepare_images_to_dataset

**Class:** `PrepareImagesToDatasetTask`

Tarea para crear un dataset de placas a partir de LabelMe.

Example YAML:
    ```yaml
- name: prepare_images_to_dataset
  params:
    input_dir: <value>
    output_dir: <value>
    target_width: <value>
    target_height: <value>
    split_ratio: <value>
    padding: <value>
    mode: <value>
    ```

### YAML example

    ```yaml
- name: prepare_images_to_dataset
  params:
    input_dir: <value>
    output_dir: <value>
    target_width: <value>
    target_height: <value>
    split_ratio: <value>
    padding: <value>
    mode: <value>
    ```

## resize_folder

**Class:** `ResizeFolderTask`

Tarea para analizar y redimensionar imágenes en una carpeta.

Example YAML:
    ```yaml
- name: resize_folder
  params:
    source_folder: <value>
    destination_folder: <value>
    width: <value>
    height: <value>
    do_resize: <value>
    ```

### YAML example

    ```yaml
- name: resize_folder
  params:
    source_folder: <value>
    destination_folder: <value>
    width: <value>
    height: <value>
    do_resize: <value>
    ```

## separar_train_val

**Class:** `SepararTrainValTask`

Tarea para separar un conjunto en train y validation.

Example YAML:
    ```yaml
- name: separar_train_val
  params:
    source_folder: <value>
    train_folder: <value>
    val_folder: <value>
    split_ratio: <value>
    ```

### YAML example

    ```yaml
- name: separar_train_val
  params:
    source_folder: <value>
    train_folder: <value>
    val_folder: <value>
    split_ratio: <value>
    ```

## unir_videos

**Class:** `UnirVideosTask`

Tarea para unir múltiples videos en un solo archivo.

Example YAML:
    ```yaml
- name: unir_videos
  params:
    source_folder: <value>
    output_path: <value>
    target_width: <value>
    target_height: <value>
    ```

### YAML example

    ```yaml
- name: unir_videos
  params:
    source_folder: <value>
    output_path: <value>
    target_width: <value>
    target_height: <value>
    ```

## visualize_dataset

**Class:** `VisualizeDatasetTask`

Tarea para visualizar y analizar datasets de placas OCR.

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

### YAML example

    ```yaml
- name: visualize_dataset
  params:
    dataset_dir: <value>
    num_samples: <value>
    random_sample: <value>
    show_train: <value>
    show_val: <value>
    ```

## yolo_crop_dataset

**Class:** `YoloCropDatasetTask`

Tarea para recortar placas con YOLO y generar dataset OCR.

Example YAML:
    ```yaml
- name: yolo_crop_dataset
  params:
    input: <value>
    model: <value>
    output: <value>
    conf: <value>
    padding: <value>
    split: <value>
    seed: <value>
    ```

### YAML example

    ```yaml
- name: yolo_crop_dataset
  params:
    input: <value>
    model: <value>
    output: <value>
    conf: <value>
    padding: <value>
    split: <value>
    seed: <value>
    ```

