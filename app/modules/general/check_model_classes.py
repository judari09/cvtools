"""Muestra las clases disponibles de uno o más modelos YOLO.

Este módulo permite inspeccionar la etiqueta de clases interna de modelos
Ultralytics/YOLO cargados desde disco.
"""

import argparse

from app.core.task import Task


class CheckModelClassesTask(Task):
    """Tarea para mostrar las clases de modelos YOLO.

Example YAML:
```yaml
- name: check_model_classes
  params:
    models: <value>
```

Example YAML:
```yaml
- name: check_model_classes
  params:
    models: <value>
```

Example YAML:
```yaml
- name: check_model_classes
  params:
    models: <value>
```"""

    name = "check_model_classes"

    def __init__(self, params):
        """Inicializa la tarea con los parámetros dados.

        Parameters
        ----------
        params : dict
            Debe contener la clave 'models' con lista de rutas a modelos.
        """
        super().__init__(name=self.name, params=params)
        self.params = params

    @staticmethod
    def _load_model(path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Instala ultralytics: pip install ultralytics") from exc
        return YOLO(path)

    def _print_model_classes(self, model_path):
        model = self._load_model(model_path)
        names = model.names
        print(f"\n{'=' * 50}")
        print(f"Modelo: {model_path}")
        print(f"Total clases: {len(names)}")
        print(f"{'=' * 50}")
        for cls_id, name in names.items():
            print(f"  {cls_id}: {name}")

    def run(self):
        """Ejecuta la tarea y muestra las clases de los modelos."""
        models = self.params.get("models") or []
        if isinstance(models, str):
            models = [models]

        if not models:
            raise ValueError("Debe proporcionar al menos un modelo en 'models'")

        for path in models:
            self._print_model_classes(path)


def main():
    parser = argparse.ArgumentParser(description="Ver clases de modelos YOLO")
    parser.add_argument("models", nargs="+", help="Rutas a modelos YOLO (.pt)")
    args = parser.parse_args()

    task = CheckModelClassesTask({"models": args.models})
    task.run()


if __name__ == "__main__":
    main()
