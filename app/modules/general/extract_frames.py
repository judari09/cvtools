"""Extrae frames de un video en intervalos regulares.

Este módulo permite generar imágenes a partir de un archivo de video
cada N fotogramas.
"""

import argparse
import os

import cv2

from app.core.task import Task


class ExtractFramesTask(Task):
    """Tarea para extraer frames de un video.

Example YAML:
```yaml
- name: extract_frames
  params:
    video_path: <value>
    output_folder: <value>
    frame_step: <value>
```

Example YAML:
```yaml
- name: extract_frames
  params:
    video_path: <value>
    output_folder: <value>
    frame_step: <value>
```

Example YAML:
```yaml
- name: extract_frames
  params:
    video_path: <value>
    output_folder: <value>
    frame_step: <value>
```"""

    name = "extract_frames"

    def __init__(self, params):
        """Inicializa la tarea.

        Parameters
        ----------
        params : dict
            Debe contener 'video_path' y 'output_folder'.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        """Ejecuta la extracción de frames."""
        video_path = self.params.get("video_path")
        output_folder = self.params.get("output_folder")
        frame_step = int(self.params.get("frame_step", 100))

        if not video_path or not output_folder:
            raise ValueError("Se requieren 'video_path' y 'output_folder'")

        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total de frames: {total_frames}")

        contador = 0
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if contador % frame_step == 0:
                nombre_frame = os.path.join(output_folder, f"frame_{frame_id:05d}.png")
                cv2.imwrite(nombre_frame, frame)
                frame_id += 1

            contador += 1

        cap.release()
        print(f"Frames guardados en la carpeta: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Extrae frames de un video")
    parser.add_argument("--video", required=True, help="Ruta al video de entrada")
    parser.add_argument("--output", required=True, help="Carpeta de salida para los frames")
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="Extrae un frame cada N fotogramas (default: 100)",
    )
    args = parser.parse_args()

    task = ExtractFramesTask(
        {
            "video_path": args.video,
            "output_folder": args.output,
            "frame_step": args.step,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
