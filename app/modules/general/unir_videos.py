"""Une múltiples archivos de video en un solo archivo de salida."""

import argparse
import os
from pathlib import Path

import cv2
from app.core.task import Task


def collect_videos(source_folder, extensions=None):
    """Recopila los videos válidos de la carpeta de origen."""
    if extensions is None:
        extensions = {'.mp4'}
    return sorted(
        Path(source_folder) / f
        for f in os.listdir(source_folder)
        if Path(f).suffix.lower() in extensions
    )


def merge_videos(video_paths, output_path, target_width=None, target_height=None):
    """Une los videos en un único archivo de salida."""
    if not video_paths:
        raise ValueError("No hay videos para unir")

    first_video = cv2.VideoCapture(str(video_paths[0]))
    fps = first_video.get(cv2.CAP_PROP_FPS)
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()

    if target_width is None:
        target_width = width
    if target_height is None:
        target_height = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        print(f"Procesando: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (frame.shape[1], frame.shape[0]) != (target_width, target_height):
                frame = cv2.resize(frame, (target_width, target_height))
            out.write(frame)
        cap.release()

    out.release()
    print(f"Video combinado guardado como: {output_path}")


class UnirVideosTask(Task):
    """Tarea para unir múltiples videos en un solo archivo.

Example YAML:
```yaml
- name: unir_videos
  params:
    source_folder: <value>
    output_path: <value>
    target_width: <value>
    target_height: <value>
```

Example YAML:
```yaml
- name: unir_videos
  params:
    source_folder: <value>
    output_path: <value>
    target_width: <value>
    target_height: <value>
```

Example YAML:
```yaml
- name: unir_videos
  params:
    source_folder: <value>
    output_path: <value>
    target_width: <value>
    target_height: <value>
```"""

    name = "unir_videos"

    def __init__(self, params):
        super().__init__(name=self.name, params=params)
        self.params = params

    def run(self):
        source_folder = self.params.get("source_folder")
        output_path = self.params.get("output_path")
        target_width = self.params.get("target_width")
        target_height = self.params.get("target_height")

        if not source_folder or not output_path:
            raise ValueError("Se requieren 'source_folder' y 'output_path'")

        video_paths = collect_videos(source_folder)
        merge_videos(video_paths, output_path, target_width, target_height)


def main():
    parser = argparse.ArgumentParser(description="Une múltiples videos en uno solo")
    parser.add_argument("--source", required=True, help="Carpeta que contiene los videos")
    parser.add_argument("--output", required=True, help="Ruta del video de salida")
    parser.add_argument("--width", type=int, help="Ancho de salida opcional")
    parser.add_argument("--height", type=int, help="Alto de salida opcional")
    args = parser.parse_args()

    task = UnirVideosTask(
        {
            "source_folder": args.source,
            "output_path": args.output,
            "target_width": args.width,
            "target_height": args.height,
        }
    )
    task.run()


if __name__ == "__main__":
    main()
