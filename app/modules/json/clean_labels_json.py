import os
import json
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task


class CleanLabelsJsonTask(Task):
    """
    Task for cleaning specific fields in JSON label files.

    This class extends the base Task class to modify JSON files by setting
    specified fields to given values, typically used for cleaning up
    unnecessary data like imageData fields.

    Attributes
    ----------
    params : object
        Configuration parameters including folder_path, field, and value.
    """

    name = "clean_labels_json"

    def __init__(self, params):
        """
        Initialize the CleanLabelsJsonTask with given parameters.

        Parameters
        ----------
        params : object
            Configuration parameters for the task.
        """
        super().__init__(name="clean_labels_json", params=params)
        self.params = params

    def clean_image_data_in_json(self, folder_path, field, value=None):
        """
        Clean a specific field in all JSON files within a folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing JSON files.
        field : str
            Name of the field to modify in each JSON file.
        value : any, optional
            Value to set for the field. Defaults to None.
        """
        # Buscar todos los archivos JSON en la carpeta
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)

                # Cargar el contenido del archivo JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Establecer el parámetro "imageData" en null
                data[field] = value

                # Guardar los cambios en el archivo JSON
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"Procesado: {file_path}")

    def run(self):
        """
        Execute the JSON cleaning process.

        Calls clean_image_data_in_json with the configured parameters.
        """
        self.clean_image_data_in_json(self.params.folder_path, self.params.field, self.params.value)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean JSON label files by setting a field value."
    )
    parser.add_argument(
        "--folder-path",
        required=True,
        help="Path to the folder containing JSON files.",
    )
    parser.add_argument(
        "--field",
        default="imageData",
        help="JSON field to update for every file.",
    )
    parser.add_argument(
        "--value",
        default="null",
        help="Value to write for the field (use 'null' to set None).",
    )
    args = parser.parse_args()
    value = None if args.value.lower() == "null" else args.value
    params = argparse.Namespace(
        folder_path=args.folder_path,
        field=args.field,
        value=value,
    )
    CleanLabelsJsonTask(params).run()
