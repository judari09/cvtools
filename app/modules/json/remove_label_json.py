import os
import json
from core.task import Task

class RemoveLabelJsonTask(Task):
    
    name = "remove_label_json"
    
    def __init__(self, params):
        super().__init__(name="remove_label_json", params=params)
        self.params = params

    def remove_label_from_jsons(self,folder_path, label_to_remove):
        """
        Recorre todos los archivos JSON (formato LabelMe) en la carpeta
        y elimina las shapes cuyo 'label' coincida con label_to_remove.
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
        self.remove_label_from_jsons(self.params.folder_path, self.params.label_to_remove)