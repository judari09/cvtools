# Auto-Labeling

This page documents the `Auto-Labeling` available in CVTools.

## auto_label_labelme

**Class:** `AutoLabelLabelMeTask`

Task for auto-labeling images using YOLO models and generating LabelMe JSON annotations.
Example YAML:
    ```yaml
    - name: auto_label_labelme
      params:
        # TODO: replace with task-specific parameters
        example_param: value
    ```

### YAML example

```yaml
- name: auto_label_labelme
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## auto_label_ocr

**Class:** `AutoLabelOcrTask`

Task for auto-labeling images using OCR (Optical Character Recognition).
Example YAML:
    ```yaml
    - name: auto_label_ocr
      params:
        # TODO: replace with task-specific parameters
        example_param: value
    ```

### YAML example

```yaml
- name: auto_label_ocr
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```

## copy_and_rename_by_plate

**Class:** `CopyAndRenameByPlateTask`

Task for copying and renaming images based on detected license plates.
Example YAML:
    ```yaml
    - name: copy_and_rename_by_plate
      params:
        # TODO: replace with task-specific parameters
        example_param: value
    ```

### YAML example

```yaml
- name: copy_and_rename_by_plate
  params:
    # TODO: replace with task-specific parameters
    example_param: value
```
