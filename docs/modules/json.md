# Procesamiento JSON

Módulos para convertir, limpiar y optimizar anotaciones LabelMe en formato JSON.

---

## clean_labels_json

**Clase:** `CleanLabelsJsonTask`

Modifica campos específicos en archivos JSON de etiquetas, útil para eliminar datos innecesarios como `imageData`.

### Parámetros YAML

```yaml
- name: clean_labels_json
  params:
    folder_path: data/labels/json
    field: imageData
    value: null
```

---

## fix_labelme_json

**Clase:** `FixLabelmeJsonTask`

Corrige archivos JSON generados por `auto_label_labelme` para que sean totalmente válidos y abribles en LabelMe.

### Parámetros YAML

```yaml
- name: fix_labelme_json
  params:
    labels_dir: data/labels/json
    images_dir: data/images
    version: "5.3.1"
    dry_run: false
```

---

## json2txt

**Clase:** `Json2TxtTask`

Convierte anotaciones LabelMe JSON a formato YOLO `.txt`. Soporta `rectangle`, `polygon` y polígonos de 4 puntos como bounding box.

Las dimensiones de imagen se obtienen por prioridad: imagen real → metadatos JSON → fallback 640×360.

### Parámetros YAML

```yaml
- name: json2txt
  params:
    input_dir: data/labels/json
    output_dir: data/labels/txt
    carpeta_imagenes: data/images   # opcional, por defecto igual a input_dir
    class_map:
      car: 0
      plate: 1
    default_class_id: 0
    polygon_4pt_as_bbox: false      # true: polígonos de 4 pts → bbox detección
```

### CLI

```bash
python -m app.modules.json.json2txt \
  --input-dir data/labels/json \
  --output-dir data/labels/txt \
  --class-map car:0 plate:1 \
  --polygon-4pt-as-bbox
```

---

## listar_clases_json

**Clase:** `ListarClasesJsonTask`

Lista y cuenta las clases únicas presentes en un directorio de archivos JSON LabelMe.

### Parámetros YAML

```yaml
- name: listar_clases_json
  params:
    folder_path: data/labels/json
```

---

## optimize_polygons_json

**Clase:** `OptimizePolygonsJsonTask`

Simplifica y optimiza polígonos en anotaciones LabelMe usando Douglas-Peucker y suavizado opcional.

### Parámetros YAML

```yaml
- name: optimize_polygons_json
  params:
    folder_path: data/labels/json
    epsilon: 2.0
    min_points: 4
    min_dist: 1.0
    smooth: false
    smooth_window: 3
    target_label: null        # null para procesar todas las clases
```

---

## remove_label_json

**Clase:** `RemoveLabelJsonTask`

Elimina todas las anotaciones de una clase específica en un directorio de archivos JSON.

### Parámetros YAML

```yaml
- name: remove_label_json
  params:
    folder_path: data/labels/json
    label_to_remove: background
```
