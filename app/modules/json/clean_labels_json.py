import os
import json
from core.task import Task

class CleanLabelsJsonTask(Task):
    
    name = "clean_labels_json"
    
    def __init__(self, params):
        super().__init__(name="clean_labels_json", params=params)
        self.params = params

    def clean_image_data_in_json(self, folder_path, field, value=None):
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
        self.clean_image_data_in_json(self.params.folder_path, self.params.field, self.params.value)
