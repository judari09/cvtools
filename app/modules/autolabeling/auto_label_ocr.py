import os
from paddleocr import PaddleOCR
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task


class AutoLabelOcrTask(Task):
    """Task for auto-labeling images using OCR (Optical Character Recognition).

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

Example YAML:
```yaml
- name: auto_label_ocr
  params:
    input_folder: <value>
    output_file: <value>
```

Example YAML:
```yaml
- name: auto_label_ocr
  params:
    input_folder: <value>
    output_file: <value>
```"""

    name = "auto_label_ocr"
    
    def __init__(self, params):
        """
        Initialize the AutoLabelOcrTask with given parameters.

        Parameters
        ----------
        params : object
            Configuration parameters for the task, including input/output directories
            and OCR settings.
        """
        super().__init__(name="auto_label_ocr", params=params)
        self.params = params
        # Inicializar PaddleOCR
        # Puedes ajustar 'lang' según el idioma de tus displays (ej. 'en' para inglés, 'ch' para chino, etc.)
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en",drop_confidence = 0.5)  

    def run(self):
        """
        Execute the OCR auto-labeling process.

        Processes all images in the input folder, performs OCR on each image,
        and writes the detected text to the output file. If no text is detected,
        writes "0" as the label.

        Notes
        -----
        Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff.
        The output file is overwritten if it exists.
        """
        with open(self.params.output_file, 'w') as f:
            for filename in os.listdir(self.params.input_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_path = os.path.join(self.params.input_folder, filename)
                    result = self.ocr.ocr(image_path,det=True, rec=True, cls=True, bin=True)
                    
                    # Inicializamos la etiqueta como "0" en caso de que no se detecte nada
                    label = "    0"
                    if result and result[0]:
                        detected_texts = []
                        for result_ocr in result[0]:
                            bbox, (texto, confidence) = result_ocr
                        # Si se detectó texto, unirlo; de lo contrario, se mantiene "0"
                        if texto:
                            label = f"    {texto}"
                    
                    f.write(f"{filename}{label}\n")
                    print(f"Procesada: {filename} -> {label}")

        print(f"\nArchivo de etiquetas generado: {self.params.output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OCR-based auto-labeling on a folder of images."
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output text file for OCR results.",
    )
    args = parser.parse_args()
    params = argparse.Namespace(
        input_folder=args.input_folder,
        output_file=args.output_file,
    )
    AutoLabelOcrTask(params).run()
