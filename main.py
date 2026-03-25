import yaml
from app.core.pipeline import Pipeline
from app.core.registry import registry
from app.core.task import Task
from app.core.executor import executor


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
