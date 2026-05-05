# CVTools

Suite completa de herramientas de visión por computadora diseñada para la preparación de datasets, aumento de datos, etiquetado automático y procesamiento general de imágenes. Construida con una arquitectura modular basada en tareas que soporta pipelines YAML y ejecución directa por CLI.

## Características

### Arquitectura principal

- **Diseño basado en tareas**: Módulos independientes que se pueden combinar en pipelines complejos
- **Configuración YAML**: Define y ejecuta flujos de trabajo de múltiples pasos mediante archivos de configuración
- **Soporte CLI**: Ejecución directa desde la línea de comandos para cada herramienta
- **Extensible**: Fácil de agregar nuevas tareas e integrar con flujos de trabajo de visión existentes

### Módulos disponibles

#### Herramientas generales (`app/modules/general/`)

- **Verificación de integridad de datasets**: Verifica la consistencia del dataset y detecta archivos corruptos
- **Validación de clases del modelo**: Revisa y valida configuraciones de clases de modelos YOLO
- **Análisis de tamaño de imágenes**: Analiza y reporta dimensiones de imágenes en datasets
- **Recorte de imágenes**: Recorta imágenes según diversos criterios
- **Extracción de frames**: Extrae frames de archivos de video
- **Base de datos de imágenes**: Gestiona y organiza colecciones de imágenes
- **División de datasets**: Divide datasets en conjuntos de entrenamiento, validación y prueba
- **Redimensionado de imágenes**: Redimensiona imágenes por lotes a dimensiones especificadas
- **Visualización de datasets**: Genera reportes visuales y estadísticas de datasets
- **Recorte de datasets YOLO**: Recorte especializado para datasets en formato YOLO

#### Procesamiento JSON (`app/modules/json/`)

- **Limpieza de etiquetas**: Elimina o modifica campos específicos en archivos JSON de etiquetas
- **Corrección de JSON LabelMe**: Corrige archivos JSON de LabelMe para compatibilidad
- **JSON a TXT**: Convierte anotaciones JSON de LabelMe a formato YOLO `.txt`
- **Listar clases**: Extrae y cuenta clases únicas desde archivos JSON
- **Optimización de polígonos**: Simplifica y optimiza anotaciones de polígonos
- **Eliminar etiquetas**: Elimina tipos de etiquetas específicos de archivos JSON

#### Aumento de datos (`app/modules/albumentations/`)

- **Aumento para YOLO**: Aplica aumentaciones a datasets YOLO de detección de objetos
- **Aumento para YOLO Segmentación**: Aumenta datasets de segmentación YOLO con preservación de polígonos

#### Auto-etiquetado (`app/modules/autolabeling/`)

- **Auto-etiquetado con LabelMe**: Etiquetado automático con modelos YOLO y refinamiento SAM
- **Auto-etiquetado OCR**: Extrae texto de imágenes usando PaddleOCR
- **Procesamiento de placas**: Detecta y procesa placas vehiculares con ALPR

## Instalación

### Requisitos previos

- Python 3.12 o superior
- pip

### Instalación básica

```bash
git clone <repository-url>
cd cvtools
pip install -e .
```

### Dependencias adicionales

Según los módulos que vayas a usar, instala los paquetes adicionales:

```bash
# Para operaciones generales de visión por computadora
pip install opencv-python numpy tqdm

# Para tareas basadas en YOLO
pip install ultralytics

# Para aumento de datos
pip install albumentations

# Para tareas OCR
pip install paddlepaddle paddleocr

# Para reconocimiento de placas
pip install fast-alpr
```

## Uso

### Modo pipeline YAML (recomendado)

CVTools soporta flujos de trabajo complejos mediante archivos de configuración YAML, permitiendo encadenar múltiples tareas y ejecutarlas en secuencia.

#### Ejemplo de configuración

Consulta `config_example.yaml` para un ejemplo completo. Aquí un pipeline básico:

```yaml
tasks:
  - name: check_sizes
    params:
      folder_path: "data/images"
      output_file: "size_report.txt"

  - name: prepare_images_to_dataset
    params:
      input_folder: "data/raw_images"
      output_folder: "data/dataset"
      resize_width: 640
      resize_height: 480

  - name: separar_train_val
    params:
      dataset_folder: "data/dataset"
      train_ratio: 0.7
      val_ratio: 0.2
      test_ratio: 0.1
```

#### Ejecutar un pipeline

```bash
python main.py
```

El pipeline ejecutará todas las tareas definidas en `config_example.yaml` en orden.

### Modo CLI

Cada módulo puede ejecutarse directamente desde la línea de comandos para tareas rápidas o pruebas.

#### Ejemplos — Herramientas generales

```bash
# Verificar integridad del dataset
python -m app.modules.general.check_integrity_dataset --folder-path data/images

# Redimensionar imágenes
python -m app.modules.general.resize_folder --input-folder data/images --output-folder data/resized --width 640 --height 480

# Dividir dataset
python -m app.modules.general.separar_train_val --dataset-folder data/dataset --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

#### Ejemplos — Procesamiento JSON

```bash
# Convertir JSON a formato YOLO
python -m app.modules.json.json2txt --input-dir labels/json --output-dir labels/txt

# Corregir archivos JSON de LabelMe
python -m app.modules.json.fix_labelme_json --labels-dir labels/json --images-dir images

# Listar clases únicas
python -m app.modules.json.listar_clases_json --folder-path labels/json
```

#### Ejemplos — Aumento de datos

```bash
# Aumentar dataset YOLO
python -m app.modules.albumentations.albumentations_for_yolo --input-images-dir data/images --input-labels-dir data/labels --output-images-dir data/aug_images --output-labels-dir data/aug_labels
```

#### Ejemplos — Auto-etiquetado

```bash
# Auto-etiquetar con YOLO
python -m app.modules.autolabeling.auto_label_labelme --models yolov8n-seg.pt --input images --output labels/json --conf 0.5

# Etiquetado OCR
python -m app.modules.autolabeling.auto_label_ocr --input-folder images --output-file labels.txt
```

## Estructura del proyecto

```
cvtools/
├── main.py                    # Punto de entrada para pipelines YAML
├── config_example.yaml        # Ejemplo de configuración de pipeline
├── pyproject.toml             # Configuración del proyecto y dependencias
├── app/
│   ├── core/                  # Arquitectura principal
│   │   ├── task.py            # Clase base Task
│   │   ├── registry.py        # Sistema de registro de tareas
│   │   ├── executor.py        # Motor de ejecución de pipelines
│   │   └── pipeline.py        # Cargador de pipelines YAML
│   └── modules/               # Módulos de tareas
│       ├── general/           # Utilidades generales de visión
│       ├── json/              # Herramientas de procesamiento JSON
│       ├── albumentations/    # Aumento de datos
│       └── autolabeling/      # Herramientas de auto-etiquetado
├── test/                      # Archivos de prueba
├── pipelines/                 # Ejemplos adicionales de pipelines
└── utils/                     # Scripts de utilidades
```

## Configuración

### Parámetros de tareas

Cada tarea acepta parámetros específicos. Consulta el docstring de la tarea o usa `--help` en modo CLI para información detallada de los parámetros.

### Pipelines personalizados

Crea tu propio `config.yaml` y modifica `main.py` para cargarlo:

```python
pipeline = Pipeline("your_config.yaml")
```

## Documentación

La documentación del proyecto está generada en la carpeta `docs/` y configurada con `mkdocs.yml`.

Para previsualizar la documentación localmente:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Para construir el sitio estático:

```bash
mkdocs build
```

## Contribuir

1. Haz un fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -m 'Agrega nueva funcionalidad'`)
4. Sube la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

### Agregar nuevas tareas

1. Crea una clase de tarea que herede de `Task`
2. Implementa el método `run()`
3. Agrega el wrapper CLI con `if __name__ == "__main__"`
4. Registra la tarea en `main.py` si es necesario para pipelines
5. Actualiza la documentación

## Licencia

Este proyecto está bajo la Licencia MIT — consulta el archivo LICENSE para más detalles.

## Soporte

Para reportar problemas, hacer preguntas o contribuir, abre un issue en el repositorio de GitHub.

## Reconocimientos

- Construido con Ultralytics YOLO para tareas de visión por computadora
- Usa Albumentations para aumento de datos
- Aprovecha PaddleOCR para reconocimiento de texto
- Fast-ALPR para reconocimiento de placas vehiculares
