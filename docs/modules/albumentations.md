# Albumentations Tools

This page documents the `Albumentations Tools` available in CVTools.

## albumentations_for_yolo

**Class:** `AlbumentationsForYoloTask`

Task for applying data augmentation to YOLO-formatted datasets using Albumentations.

    This class extends the base Task class to perform image and bounding box augmentations
    specifically tailored for YOLO object detection models. It applies a series of transformations
    to input images and their corresponding label files, generating multiple augmented versions
    of each image with updated bounding box coordinates.

    Attributes
    ----------
    transform : albumentations.Compose
        The composition of augmentation transformations to apply to images and bounding boxes.

Example YAML:
    ```yaml
- name: albumentations_for_yolo
  params:
    input_images_dir: <value>
    input_labels_dir: <value>
    output_images_dir: <value>
    output_labels_dir: <value>
    ```

### YAML example

    ```yaml
- name: albumentations_for_yolo
  params:
    input_images_dir: <value>
    input_labels_dir: <value>
    output_images_dir: <value>
    output_labels_dir: <value>
    ```

## albumentations_for_yoloseg

**Class:** `AlbumentationsForYolosegTask`

Task for applying data augmentation to YOLO segmentation datasets using Albumentations.

    This class extends the base Task class to perform image and polygon augmentation
    specifically tailored for YOLO segmentation models. It applies tiered augmentation
    strategies based on class frequency to balance the dataset, generating multiple
    augmented versions of each image with updated polygon coordinates.

    Attributes
    ----------
    KP_PARAMS : albumentations.KeypointParams
        Parameters for keypoint handling in augmentations.

Example YAML:
    ```yaml
- name: albumentations_for_yoloseg
  params:
    input_images_dir: <value>
    input_labels_dir: <value>
    output_images_dir: <value>
    output_labels_dir: <value>
    ```

### YAML example

    ```yaml
- name: albumentations_for_yoloseg
  params:
    input_images_dir: <value>
    input_labels_dir: <value>
    output_images_dir: <value>
    output_labels_dir: <value>
    ```

