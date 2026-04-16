# CVTools

A comprehensive suite of computer vision tools designed for dataset preparation, augmentation, auto-labeling, and general image processing tasks. Built with a modular task-based architecture that supports both YAML-driven pipelines and direct CLI execution.

## Features

### Core Architecture

- **Task-Based Design**: Modular tasks that can be composed into complex pipelines
- **YAML Configuration**: Define and execute multi-step workflows via configuration files
- **CLI Support**: Direct command-line execution for individual tools
- **Extensible**: Easy to add new tasks and integrate with existing CV workflows

### Available Modules

#### General Tools (`app/modules/general/`)

- **Dataset Integrity Check**: Verify dataset consistency and detect corrupted files
- **Model Classes Validation**: Check and validate YOLO model class configurations
- **Image Size Analysis**: Analyze and report image dimensions across datasets
- **Image Cropping**: Crop images based on various criteria
- **Frame Extraction**: Extract frames from video files
- **Image Database**: Manage and organize image collections
- **Dataset Splitting**: Split datasets into train/validation/test sets
- **Image Resizing**: Batch resize images to specified dimensions
- **Dataset Visualization**: Generate visual reports and statistics for datasets
- **YOLO Dataset Cropping**: Specialized cropping for YOLO-formatted datasets

#### JSON Processing (`app/modules/json/`)

- **Clean Labels**: Remove or modify specific fields in JSON label files
- **Fix LabelMe JSON**: Correct LabelMe JSON files for compatibility
- **JSON to TXT**: Convert LabelMe JSON annotations to YOLO .txt format
- **List Classes**: Extract and count unique classes from JSON files
- **Optimize Polygons**: Simplify and optimize polygon annotations
- **Remove Labels**: Remove specific label types from JSON files

#### Data Augmentation (`app/modules/albumentations/`)

- **YOLO Augmentation**: Apply augmentations to YOLO object detection datasets
- **YOLO Segmentation Augmentation**: Augment YOLO segmentation datasets with polygon preservation

#### Auto-Labeling (`app/modules/autolabeling/`)

- **LabelMe Auto-Labeling**: Automatic labeling using YOLO models with SAM refinement
- **OCR Auto-Labeling**: Extract text from images using PaddleOCR
- **License Plate Processing**: Detect and process license plates with ALPR

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Basic Installation

```bash
git clone <repository-url>
cd cvtools
pip install -e .
```

### Additional Dependencies

Depending on the modules you plan to use, install additional packages:

```bash
# For general CV operations
pip install opencv-python numpy tqdm

# For YOLO-based tasks
pip install ultralytics

# For data augmentation
pip install albumentations

# For OCR tasks
pip install paddlepaddle paddleocr

# For license plate recognition
pip install fast-alpr
```

## Usage

### YAML Pipeline Mode (Recommended)

CVTools supports complex workflows through YAML configuration files. This allows you to chain multiple tasks together and execute them in sequence.

#### Example Configuration

See `config_example.yaml` for a complete example. Here's a basic pipeline:

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

#### Running a Pipeline

```bash
python main.py
```

The pipeline will execute all tasks defined in `config_example.yaml` in order.

### CLI Mode

Each module can be executed directly from the command line for quick tasks or testing.

#### General Tools Examples

```bash
# Check dataset integrity
python -m app.modules.general.check_integrity_dataset --folder-path data/images

# Resize images
python -m app.modules.general.resize_folder --input-folder data/images --output-folder data/resized --width 640 --height 480

# Split dataset
python -m app.modules.general.separar_train_val --dataset-folder data/dataset --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

#### JSON Processing Examples

```bash
# Convert JSON to YOLO format
python -m app.modules.json.json2txt --input-dir labels/json --output-dir labels/txt

# Fix LabelMe JSON files
python -m app.modules.json.fix_labelme_json --labels-dir labels/json --images-dir images

# List unique classes
python -m app.modules.json.listar_clases_json --folder-path labels/json
```

#### Augmentation Examples

```bash
# Augment YOLO dataset
python -m app.modules.albumentations.albumentations_for_yolo --input-images-dir data/images --input-labels-dir data/labels --output-images-dir data/aug_images --output-labels-dir data/aug_labels
```

#### Auto-Labeling Examples

```bash
# Auto-label with YOLO
python -m app.modules.autolabeling.auto_label_labelme --models yolov8n-seg.pt --input images --output labels/json --conf 0.5

# OCR labeling
python -m app.modules.autolabeling.auto_label_ocr --input-folder images --output-file labels.txt
```

## Project Structure

```
cvtools/
├── main.py                    # Main entry point for YAML pipelines
├── config_example.yaml        # Example pipeline configuration
├── pyproject.toml            # Project configuration and dependencies
├── app/
│   ├── core/                 # Core architecture
│   │   ├── task.py          # Base Task class
│   │   ├── registry.py      # Task registration system
│   │   ├── executor.py      # Pipeline execution engine
│   │   └── pipeline.py      # YAML pipeline loader
│   └── modules/             # Task modules
│       ├── general/         # General CV utilities
│       ├── json/            # JSON processing tools
│       ├── albumentations/  # Data augmentation
│       └── autolabeling/    # Auto-labeling tools
├── test/                    # Test files
├── pipelines/               # Additional pipeline examples
└── utils/                   # Utility scripts
```

## Configuration

### Task Parameters

Each task accepts specific parameters. Refer to the task's docstring or use `--help` with CLI mode for detailed parameter information.

### Custom Pipelines

Create your own `config.yaml` and modify `main.py` to load it:

```python
pipeline = Pipeline("your_config.yaml")
```

## Documentation

Project documentation is generated in the `docs/` folder and configured with `mkdocs.yml`.

To preview the documentation locally:

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

To build the static site:

```bash
mkdocs build
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Adding New Tasks

1. Create a new task class inheriting from `Task`
2. Implement the `run()` method
3. Add CLI wrapper with `if __name__ == "__main__"`
4. Register the task in `main.py` if needed for pipelines
5. Update documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

## Acknowledgments

- Built with Ultralytics YOLO for computer vision tasks
- Uses Albumentations for data augmentation
- Leverages PaddleOCR for text recognition
- Fast-ALPR for license plate recognition
