import os
import json
try:
    from app.core.task import Task
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from app.core.task import Task


class RemoveLabelJsonTask(Task):
    """Tarea para remove_label_json.

Example YAML:
```yaml
- name: remove_label_json
  params:
    folder_path: <value>
    label_to_remove: <value>
```"""

    
    name = "remove_label_json"
    
    def __init__(self, params):
        """Initialize the RemoveLabelJsonTask.

        Parameters
        ----------
        params : object
            Parameters object containing configuration.
        """
        super().__init__(name="remove_label_json", params=params)
        self.params = params

    def remove_label_from_jsons(self,folder_path, label_to_remove):
        """Remove shapes with a specific label from all JSON files in a folder.

        Parameters
        ----------
        folder_path : str
            Path to folder containing JSON files.
        label_to_remove : str
            Label to remove from shapes.
        """
        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

        if not json_files:
            print(f"No se encontraron archivos JSON en {folder_path}")
            return

        total_removed = 0
        files_modified = 0

        for file_name in json_files:
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            shapes_original = data.get("shapes", [])
            shapes_filtered = [s for s in shapes_original if s["label"] != label_to_remove]

            removed_count = len(shapes_original) - len(shapes_filtered)

            if removed_count > 0:
                data["shapes"] = shapes_filtered
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                total_removed += removed_count
                files_modified += 1
                print(
                    f"  {file_name}: {removed_count} etiqueta(s) '{label_to_remove}' eliminada(s)"
                )

        print(
            f"\nResumen: {total_removed} etiqueta(s) eliminada(s) en {files_modified} archivo(s)"
        )

    def run(self):
        """Run the label removal task.
        """
        self.remove_label_from_jsons(self.params.folder_path, self.params.label_to_remove)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove shapes with a specific label from LabelMe JSON files."
    )
    parser.add_argument(
        "--folder-path",
        required=True,
        help="Folder containing JSON files.",
    )
    parser.add_argument(
        "--label-to-remove",
        required=True,
        help="Label value to remove from each JSON file.",
    )
    args = parser.parse_args()
    params = argparse.Namespace(
        folder_path=args.folder_path,
        label_to_remove=args.label_to_remove,
    )
    RemoveLabelJsonTask(params).run()
