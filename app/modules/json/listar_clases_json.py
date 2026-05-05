"""
Revisa todos los archivos de etiquetas en formato JSON (LabelMe)
de una carpeta y lista todas las clases únicas encontradas,
junto con la cantidad de apariciones de cada una.
"""

import os
try:
    from app.core.task import Task
    from app.utils.class_utils import count_classes_json
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task
    from app.utils.class_utils import count_classes_json


class ListarClasesJsonTask(Task):
    """Tarea para listar_clases_json.

Example YAML:
```yaml
- name: listar_clases_json
  params:
    folder_path: <value>
```"""

    
    name = "listar_clases_json"
    
    def __init__(self, params):
        """Initialize the ListarClasesJsonTask.

        Parameters
        ----------
        params : object
            Parameters object containing configuration.
        """
        super().__init__(name="listar_clases_json", params=params)
        self.params = params


    def listar_clases(self, folder_path):
        """List all unique classes from JSON files in a folder."""
        contador_clases = count_classes_json(folder_path)
        json_count = sum(1 for f in os.listdir(folder_path) if f.endswith('.json'))

        print(f"\n{'='*50}")
        print(f"Carpeta: {folder_path}")
        print(f"Archivos JSON procesados: {json_count}")
        print(f"Total de anotaciones: {sum(contador_clases.values())}")
        print(f"Clases únicas encontradas: {len(contador_clases)}")
        print(f"{'='*50}")

        print(f"\n{'Clase':<30} {'Cantidad':>10}")
        print(f"{'-'*30} {'-'*10}")
        for clase, cantidad in contador_clases.most_common():
            print(f"{clase:<30} {cantidad:>10}")

        return contador_clases

    def run(self):
        """Run the class listing task."""
        self.listar_clases(self.params.get("folder_path"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="List unique classes from LabelMe JSON files."
    )
    parser.add_argument(
        "--folder-path",
        required=True,
        help="Folder containing JSON files.",
    )
    args = parser.parse_args()
    params = {"folder_path": args.folder_path}
    ListarClasesJsonTask(params).run()
