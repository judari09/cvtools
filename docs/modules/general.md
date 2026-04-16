# General Tools

This page documents the `General Tools` available in CVTools.

## check_integrity_dataset

**Class:** `CheckIntegrityDatasetTask`

Tarea para verificar integridad de imágenes y etiquetas YOLO.
Example YAML:
    ```yaml
    - name: check_integrity_dataset
      params:
        # TODO: replace with task-specific parameters
        example_param: value
    ```

### YAML example

```yaml
- name: check_integrity_dataset
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## check_model_classes

**Class:** `CheckModelClassesTask`

Tarea para mostrar las clases de modelos YOLO.

### YAML example

```yaml
- name: check_model_classes
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## check_sizes

**Class:** `CheckSizesTask`

Tarea para comprobar el tamaño de imágenes en un directorio.

### YAML example

```yaml
- name: check_sizes
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## crop_images

**Class:** `CropImagesTask`

Tarea para recortar imágenes en un área fija.

### YAML example

```yaml
- name: crop_images
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## extract_frames

**Class:** `ExtractFramesTask`

Tarea para extraer frames de un video.

### YAML example

```yaml
- name: extract_frames
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## imagenes_db

**Class:** `ImagenesDbTask`

Tarea para mover imágenes que tienen etiquetas TXT asociadas.

### YAML example

```yaml
- name: imagenes_db
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## move_to_comb

**Class:** `MoveToCombTask`

Tarea para copiar imágenes y etiquetas emparejadas a una carpeta combinada.

### YAML example

```yaml
- name: move_to_comb
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## prepare_images_to_dataset

**Class:** `PrepareImagesToDatasetTask`

Tarea para crear un dataset de placas a partir de LabelMe.

### YAML example

```yaml
- name: prepare_images_to_dataset
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## resize_folder

**Class:** `ResizeFolderTask`

Tarea para analizar y redimensionar imágenes en una carpeta.

### YAML example

```yaml
- name: resize_folder
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## separar_train_val

**Class:** `SepararTrainValTask`

Tarea para separar un conjunto en train y validation.

### YAML example

```yaml
- name: separar_train_val
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## unir_videos

**Class:** `UnirVideosTask`

Tarea para unir múltiples videos en un solo archivo.

### YAML example

```yaml
- name: unir_videos
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## visualize_dataset

**Class:** `VisualizeDatasetTask`

Tarea para visualizar y analizar datasets de placas OCR.

### YAML example

```yaml
- name: visualize_dataset
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## yolo_crop_dataset

**Class:** `YoloCropDatasetTask`

Tarea para recortar placas con YOLO y generar dataset OCR.

### YAML example

```yaml
- name: yolo_crop_dataset
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```
