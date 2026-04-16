import yaml
from app.core.pipeline import Pipeline
from app.core.registry import registry
from app.core.task import Task
from app.core.executor import executor

from app.modules.general.check_integrity_dataset import CheckIntegrityDatasetTask
from app.modules.general.check_model_classes import CheckModelClassesTask
from app.modules.general.check_sizes import CheckSizesTask
from app.modules.general.crop_images import CropImagesTask
from app.modules.general.extract_frames import ExtractFramesTask
from app.modules.general.imagenes_db import ImagenesDbTask
from app.modules.general.move_to_comb import MoveToCombTask
from app.modules.general.prepare_images_to_dataset import PrepareImagesToDatasetTask
from app.modules.general.resize_folder import ResizeFolderTask
from app.modules.general.separar_train_val import SepararTrainValTask
from app.modules.general.unir_videos import UnirVideosTask
from app.modules.general.visualize_dataset import VisualizeDatasetTask
from app.modules.general.yolo_crop_dataset import YoloCropDatasetTask


# Crear una tarea de ejemplo
class ResizeImagesTask(Task):
    name = "resize_images"
    
    def __init__(self, params):
        super().__init__(name="resize_images", params=params)
    
    def run(self):
        print(f"Ejecutando tarea: {self.name}")
        print(f"Parámetros: {self.params}")
        print("✓ Imágenes redimensionadas correctamente")


class ConvertFormatTask(Task):
    name = "convert_format"
    
    def __init__(self, params):
        super().__init__(name="convert_format", params=params)
    
    def run(self):
        print(f"Ejecutando tarea: {self.name}")
        print(f"Parámetros: {self.params}")
        print("✓ Formato convertido correctamente")


def main():
    print("=== CVTools - Sistema de Tareas ===\n")
    
    # Inicializar el registro de tareas
    task_reg = registry()
    pipeline_executor = executor()

    # Mapa local de todas las clases disponibles (no cargadas en registro aun)
    available_tasks = {
        ResizeImagesTask.name: ResizeImagesTask,
        ConvertFormatTask.name: ConvertFormatTask,
        CheckIntegrityDatasetTask.name: CheckIntegrityDatasetTask,
        CheckModelClassesTask.name: CheckModelClassesTask,
        CheckSizesTask.name: CheckSizesTask,
        CropImagesTask.name: CropImagesTask,
        ExtractFramesTask.name: ExtractFramesTask,
        ImagenesDbTask.name: ImagenesDbTask,
        MoveToCombTask.name: MoveToCombTask,
        PrepareImagesToDatasetTask.name: PrepareImagesToDatasetTask,
        ResizeFolderTask.name: ResizeFolderTask,
        SepararTrainValTask.name: SepararTrainValTask,
        UnirVideosTask.name: UnirVideosTask,
        VisualizeDatasetTask.name: VisualizeDatasetTask,
        YoloCropDatasetTask.name: YoloCropDatasetTask,
    }

    # Cargar configuración desde YAML
    print("Cargando configuración desde YAML...")
    try:
        pipeline = Pipeline("config_example.yaml")
        print("✓ Configuración cargada exitosamente")
        print(f"Tareas en la configuración: {[task['name'] for task in pipeline.config['tasks']]}\n")

        # Registrar solo las tareas que existen en el YAML using registry method
        task_names = [task_config['name'] for task_config in pipeline.config['tasks']]
        result = task_reg.register_tasks_from_list(task_names, available_tasks)

        if result['missing']:
            for missing_task in result['missing']:
                print(f"⚠ Tarea '{missing_task}' no está disponible en el catálogo de tareas")

        print("Registrando solo las tareas necesarias...")
        print(f"Tareas registradas: {list(task_reg.task_registry.keys())}\n")

        # Instanciar y ejecutar solo las tareas cargadas desde YAML
        print("Ejecutando tareas...\n")
        pipeline_executor.execute(pipeline.config['tasks'], task_reg)
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
