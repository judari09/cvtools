"""Módulo para definir la clase base abstracta de tareas.

Este módulo proporciona la interfaz base que deben implementar todas las tareas
en el sistema de pipelines de CVTools. Define el contrato que cualquier tarea
concreta debe cumplir para ser ejecutada dentro del framework.
"""
from abc import ABC


class Task(ABC):
    """Clase base abstracta para todas las tareas en el sistema.
    
    Define la interfaz estándar que deben implementar todas las tareas concretas.
    Esta clase no puede ser instanciada directamente; debe heredarse para crear
    tareas específicas.
    
    La arquitectura de tareas permite:
    - Encapsular lógica de procesamiento de forma modular
    - Reutilizar tareas en diferentes pipelines
    - Registrar dinámicamente nuevas tareas
    - Ejecutar tareas en secuencia mediante configuración YAML
    
    Attributes
    ----------
    name : str
        Identificador único de la tarea. Se utiliza para registrar y recuperar
        la tarea del registro global.
    params : dict
        Diccionario con los parámetros de configuración específicos de la tarea.
        La estructura y contenido depende de cada implementación concreta.
    
    Examples
    --------
    Para crear una tarea concreta, hereda de Task:
    
    >>> class MiTarea(Task):
    ...     name = "mi_tarea"
    ...     def __init__(self, params):
    ...         super().__init__(name="mi_tarea", params=params)
    ...     def run(self):
    ...         print(f"Ejecutando: {self.name}")
    ...         print(f"Parámetros: {self.params}")
    
    Notes
    -----
    - Siempre debe definir un atributo de clase `name` para registrar la tarea
    - El método `run()` es el punto de entrada para la ejecución
    - Los parámetros deben validarse en el `__init__` de la implementación concreta
    """
    
    def __init__(self, name, params):
        """Inicializa una tarea con nombre y parámetros.
        
        Parameters
        ----------
        name : str
            Identificador único de la tarea.
        params : dict
            Diccionario con los parámetros de configuración.
        """
        self.name = name
        self.params = params
        
    def run(self):
        """Ejecuta la lógica de la tarea.
        
        Este es un método abstracto que debe ser implementado por todas las
        subclases concretas.
        
        Raises
        ------
        NotImplementedError
            Siempre se lanza si se intenta llamar en la clase base.
            Indica que la subclase debe proporcionar una implementación.
        """
        raise NotImplementedError("Subclasses must implement this method")
    