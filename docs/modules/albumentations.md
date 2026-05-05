# Aumentación de datos

Módulos para aplicar transformaciones de datos con [Albumentations](https://albumentations.ai/) sobre datasets YOLO.

---

## albumentations_for_yolo

**Clase:** `AlbumentationsForYoloTask`

Aplica aumentaciones de imagen y bounding box sobre datasets en formato YOLO detección. Genera múltiples versiones aumentadas de cada imagen con sus coordenadas actualizadas.

### Parámetros YAML

```yaml
- name: albumentations_for_yolo
  params:
    input_images_dir: data/images
    input_labels_dir: data/labels
    output_images_dir: data/aug/images
    output_labels_dir: data/aug/labels
```

### CLI

```bash
python -m app.modules.albumentations.albumentations_for_yolo \
  --input-images-dir data/images \
  --input-labels-dir data/labels \
  --output-images-dir data/aug/images \
  --output-labels-dir data/aug/labels
```

---

## albumentations_for_yoloseg

**Clase:** `AlbumentationsForYolosegTask`

Aplica aumentaciones sobre datasets YOLO segmentación con polígonos. Usa estrategias de aumentación por niveles según la frecuencia de clase para balancear el dataset.

### Parámetros YAML

```yaml
- name: albumentations_for_yoloseg
  params:
    input_images_dir: data/images
    input_labels_dir: data/labels
    output_images_dir: data/aug/images
    output_labels_dir: data/aug/labels
```

### CLI

```bash
python -m app.modules.albumentations.albumentations_for_yoloseg \
  --input-images-dir data/images \
  --input-labels-dir data/labels \
  --output-images-dir data/aug/images \
  --output-labels-dir data/aug/labels
```
