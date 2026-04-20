"""
Revisa todos los archivos de etiquetas en formato JSON (LabelMe)
de una carpeta y lista todas las clases únicas encontradas,
junto con la cantidad de apariciones de cada una.
"""

import os
import json
from collections import Counter
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task


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


    def listar_clases(self,folder_path):
        """List all unique classes from JSON files in a folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing JSON files.

        Returns
        -------
        Counter
            Counter object with class names and their counts.
        """
        contador_clases = Counter()
        archivos_procesados = 0
        archivos_con_error = 0

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.endswith('.json'):
                continue

            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Error al leer {file_name}: {e}")
                archivos_con_error += 1
                continue

            archivos_procesados += 1
            for shape in data.get("shapes", []):
                label = shape.get("label", "<sin_etiqueta>")
                contador_clases[label] += 1

        # Resumen
        print(f"\n{'='*50}")
        print(f"Carpeta: {folder_path}")
        print(f"Archivos JSON procesados: {archivos_procesados}")
        if archivos_con_error:
            print(f"Archivos con error: {archivos_con_error}")
        print(f"Total de anotaciones: {sum(contador_clases.values())}")
        print(f"Clases únicas encontradas: {len(contador_clases)}")
        print(f"{'='*50}")

        # Tabla de clases ordenada por cantidad (descendente)
        print(f"\n{'Clase':<30} {'Cantidad':>10}")
        print(f"{'-'*30} {'-'*10}")
        for clase, cantidad in contador_clases.most_common():
            print(f"{clase:<30} {cantidad:>10}")

        return contador_clases

    def run(self):
        """Run the class listing task.
        """
        self.listar_clases(self.params.folder_path)


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
    params = argparse.Namespace(folder_path=args.folder_path)
    ListarClasesJsonTask(params).run()
