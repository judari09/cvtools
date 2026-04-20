import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task




class AlbumentationsForYoloTask(Task):
    """Task for applying data augmentation to YOLO-formatted datasets using Albumentations.

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

Example YAML:
```yaml
- name: albumentations_for_yolo
  params:
    input_images_dir: <value>
    input_labels_dir: <value>
    output_images_dir: <value>
    output_labels_dir: <value>
```

Example YAML:
```yaml
- name: albumentations_for_yolo
  params:
    input_images_dir: <value>
    input_labels_dir: <value>
    output_images_dir: <value>
    output_labels_dir: <value>
```"""

    name = "albumentations_for_yolo"
    def __init__(self, params):
        """
        Initialize the AlbumentationsForYoloTask with given parameters.

        Parameters
        ----------
        params : object
            Configuration parameters for the task, including input/output directories
            and augmentation settings.
        """
        super().__init__(name="albumentations_for_yolo", params=params)

        # Definir transformaciones
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            A.Rotate(limit=20, p=0.5),
            A.Blur(blur_limit=3, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.7),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    def read_yolo_labels(self,label_path):
        """
        Read YOLO-formatted labels from a file.

        Parameters
        ----------
        label_path : str
            Path to the YOLO label file (.txt).

        Returns
        -------
        bboxes : list of list of float
            List of bounding boxes, each as [x_center, y_center, width, height].
        labels : list of int
            List of class labels corresponding to each bounding box.
        """
        bboxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split() #aplica el separador de " "
                labels.append(int(parts[0])) # Clase de la caja normalmente es entero y es el primer elemento
                bboxes.append([float(x) for x in parts[1:5]]) # Coordenadas de la caja en formato [x_center, y_center, width, height]
        return bboxes, labels

    def save_yolo_labels(self, label_path, bboxes, labels):
        """
        Save YOLO-formatted labels to a file.

        Parameters
        ----------
        label_path : str
            Path where the YOLO label file (.txt) will be saved.
        bboxes : list of list of float
            List of bounding boxes, each as [x_center, y_center, width, height].
        labels : list of int
            List of class labels corresponding to each bounding box.
        """
        with open(label_path, 'w') as f:
            for label, bbox in zip(labels, bboxes):
                # Formatear las coordenadas de la caja en formato YOLO: [x_center, y_center, width, height]
                # Las coordenadas deben estar normalizadas entre 0 y 1 toma hasta 6 decimales
                bbox_str = ' '.join([f'{x:.6f}' for x in bbox])
                # Escribir la clase y las coordenadas en el archivo
                # label es un entero que representa la clase de la caja
                f.write(f"{int(label)} {bbox_str}\n")

    def run(self):
        """
        Execute the data augmentation process for YOLO datasets.

        This method processes all images in the input directory, applies augmentations,
        and saves the augmented images and updated labels to the output directories.
        For each input image, 10 augmented versions are generated.

        Notes
        -----
        - Input images must be in .jpg, .png, or .jpeg format.
        - Corresponding label files must exist in YOLO .txt format.
        - Output images are saved as .jpg files.
        """
        os.makedirs(self.params.output_images_dir, exist_ok=True)
        os.makedirs(self.params.output_labels_dir, exist_ok=True)

        for img_name in os.listdir(self.params.input_images_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Si no es una imagen, saltar al siguiente
                print(f"Advertencia: {img_name} no es una imagen válida. Se omite.")
                continue
            img_path = os.path.join(self.params.input_images_dir, img_name) # Ruta completa de la imagen
            label_path = os.path.join(self.params.input_labels_dir, os.path.splitext(img_name)[0] + '.txt') # Ruta completa del archivo de etiquetas

            if not os.path.exists(label_path):
                # Si no existe el archivo de etiquetas, saltar al siguiente
                print(f"Advertencia: No se encontró el archivo de etiquetas para {img_name}.")
                continue
            
            # Cargar imagen, convertir a RGB y obtener dimensiones
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_h, img_w = image.shape[:2]

            # Leer las etiquetas en formato YOLO
            bboxes, labels = self.read_yolo_labels(label_path)

            for i in range(10):  # Número de aumentos por imagen
                # aplicar transformaciones
                augmented = self.transform(image=image, bboxes=bboxes, category_ids=labels)
                # Convertir la imagen aumentada de RGB a BGR para OpenCV
                # y obtener las bounding boxes y etiquetas aumentadas
                image_aug = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                bboxes_aug = augmented['bboxes']
                labels_aug = augmented['category_ids']

                # Guardar imagen aumentada
                out_img_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg" # Nombre de la imagen aumentada
                out_img_path = os.path.join(self.params.output_images_dir, out_img_name) # Ruta completa de la imagen aumentada
                cv2.imwrite(out_img_path, image_aug) # Guardar imagen aumentada en formato BGR

                # Guardar labels aumentados
                out_label_path = os.path.join(self.params.output_labels_dir, f"{os.path.splitext(img_name)[0]}_aug{i}.txt") # Ruta completa del archivo de etiquetas aumentadas
                self.save_yolo_labels(out_label_path, bboxes_aug, labels_aug) # Guardar etiquetas aumentadas en formato YOLO

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply Albumentations augmentations to a YOLO dataset."
    )
    parser.add_argument(
        "--input-images-dir",
        required=True,
        help="Input images folder.",
    )
    parser.add_argument(
        "--input-labels-dir",
        required=True,
        help="Input YOLO label files folder.",
    )
    parser.add_argument(
        "--output-images-dir",
        required=True,
        help="Output folder for augmented images.",
    )
    parser.add_argument(
        "--output-labels-dir",
        required=True,
        help="Output folder for augmented label files.",
    )
    args = parser.parse_args()
    params = argparse.Namespace(
        input_images_dir=args.input_images_dir,
        input_labels_dir=args.input_labels_dir,
        output_images_dir=args.output_images_dir,
        output_labels_dir=args.output_labels_dir,
    )
    AlbumentationsForYoloTask(params).run()
