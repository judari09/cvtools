import yaml

class Pipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if self.config.get('tasks') is None:
            raise ValueError("El archivo de configuración debe contener una sección 'tasks'")
        
        if not isinstance(self.config['tasks'], list):
            raise ValueError("La sección 'tasks' debe ser una lista de tareas")
        
        if len(self.config['tasks']) == 0:
            raise ValueError("La sección 'tasks' no puede estar vacía")
        
        if self.config['tasks']['name'] is None:
            raise ValueError("Cada tarea debe tener un nombre")
        
        if self.config['tasks']['params'] is None:
            raise ValueError("Cada tarea debe tener parámetros")