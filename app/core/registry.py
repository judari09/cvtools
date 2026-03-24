"""Logica de registro de tareas"""
from task import Task


class registry:
    """Clase para manejar el registro de tareas"""

def __init__(self):
    self.task_registry = {}


    def register_task(self, task_class):
        """Registra una tarea en el registro"""
        if not issubclass(task_class, Task):
            raise ValueError("La clase debe ser una subclase de Task")
        self.task_registry[task_class.name] = task_class