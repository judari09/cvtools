import os
from paddleocr import PaddleOCR
from core.task import Task

class AutoLabelOcrTask(Task):
    """
    Task for auto-labeling images using OCR (Optical Character Recognition).

    This class extends the base Task class to perform OCR on images in a specified input directory,
    generating a text file with the detected text for each image. It uses the PaddleOCR library
    to perform text detection and recognition.

    Attributes
    ----------
    ocr : PaddleOCR
        An instance of the PaddleOCR class for performing OCR on images.
    """

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