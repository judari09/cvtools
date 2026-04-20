# Autolabeling Tools

This page documents the `Autolabeling Tools` available in CVTools.

## auto_label_labelme

**Class:** `AutoLabelLabelmeTask`

Task for auto-labeling images using YOLO models and generating LabelMe JSON annotations.

    This class extends the base Task class to perform automatic labeling of images
    using segmentation and detection YOLO models, optionally refining with SAM.
    Generates LabelMe-compatible JSON files with polygon annotations.

    Attributes
    ----------
    params : object
        Configuration parameters for the task.

Example YAML:
    ```yaml
- name: auto_label_labelme
  params:
    input: <value>
    output: <value>
    models: <value>
    det_models: <value>
    sam_model: <value>
    conf: <value>
    class_map: <value>
    epsilon: <value>
    use_sam: <value>
    ```

### YAML example

    ```yaml
- name: auto_label_labelme
  params:
    input: <value>
    output: <value>
    models: <value>
    det_models: <value>
    sam_model: <value>
    conf: <value>
    class_map: <value>
    epsilon: <value>
    use_sam: <value>
    ```

## auto_label_ocr

**Class:** `AutoLabelOcrTask`

Task for auto-labeling images using OCR (Optical Character Recognition).

    This class extends the base Task class to perform OCR on images in a specified input directory,
    generating a text file with the detected text for each image. It uses the PaddleOCR library
    to perform text detection and recognition.

    Attributes
    ----------
    ocr : PaddleOCR
        An instance of the PaddleOCR class for performing OCR on images.

Example YAML:
    ```yaml
- name: auto_label_ocr
  params:
    input_folder: <value>
    output_file: <value>
    ```

### YAML example

    ```yaml
- name: auto_label_ocr
  params:
    input_folder: <value>
    output_file: <value>
    ```

## copy_and_rename_by_plate

**Class:** `CopyAndRenameByPlateTask`

Task for copying and renaming images based on detected license plates.

    This class extends the base Task class to process images, detect license plates
    using YOLO and ALPR OCR, and copy/rename images based on the detected plate text.
    It also crops and resizes detected plates to a standard size.

    Attributes
    ----------
    plate_detector_model : YOLO
        YOLO model for license plate detection.
    reader_alpr_ocr : ALPR
        ALPR instance for OCR on detected plates.
    valid_ext : tuple
        Valid image file extensions.

Example YAML:
    ```yaml
- name: copy_and_rename_by_plate
  params:
    CROPPED_PLATES_DIR: <value>
    OUTPUT_DIR: <value>
    OUTPUT_DIR_NOREC: <value>
    ```

### YAML example

    ```yaml
- name: copy_and_rename_by_plate
  params:
    CROPPED_PLATES_DIR: <value>
    OUTPUT_DIR: <value>
    OUTPUT_DIR_NOREC: <value>
    ```

