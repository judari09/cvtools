# Auto-etiquetado

Módulos para generar anotaciones automáticamente usando modelos YOLO, SAM y OCR.

---

## auto_label_labelme

**Clase:** `AutoLabelLabelmeTask`

Etiqueta imágenes automáticamente usando modelos YOLO de segmentación y detección, con refinamiento opcional mediante SAM. Genera archivos JSON compatibles con LabelMe.

### Parámetros YAML

```yaml
- name: auto_label_labelme
  params:
    input: data/images
    output: data/labels/json
    models:
      - yolov8n-seg.pt
    det_models:
      - yolov8n.pt
    sam_model: sam_b.pt
    conf: 0.5
    class_map:
      car: 0
      plate: 1
    epsilon: 2.0
    use_sam: false
```

### CLI

```bash
python -m app.modules.autolabeling.auto_label_labelme \
  --models yolov8n-seg.pt \
  --input data/images \
  --output data/labels/json \
  --conf 0.5
```

---

## auto_label_ocr

**Clase:** `AutoLabelOcrTask`

Realiza OCR sobre imágenes usando PaddleOCR y genera un archivo de texto con el resultado por imagen.

!!! note "Dependencia opcional"
    Requiere `pip install -e ".[ocr]"` para instalar PaddleOCR.

### Parámetros YAML

```yaml
- name: auto_label_ocr
  params:
    input_folder: data/images
    output_file: data/ocr_results.txt
```

### CLI

```bash
python -m app.modules.autolabeling.auto_label_ocr \
  --input-folder data/images \
  --output-file data/ocr_results.txt
```

---

## copy_and_rename_by_plate

**Clase:** `CopyAndRenameByPlateTask`

Detecta placas vehiculares con YOLO y Fast-ALPR, luego copia y renombra las imágenes con el texto de la placa detectada.

!!! note "Dependencia opcional"
    Requiere `pip install -e ".[ocr]"` para instalar Fast-ALPR.

### Parámetros YAML

```yaml
- name: copy_and_rename_by_plate
  params:
    CROPPED_PLATES_DIR: data/plates_crop
    OUTPUT_DIR: data/renamed
    OUTPUT_DIR_NOREC: data/no_recognized
```
