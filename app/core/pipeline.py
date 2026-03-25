"""Módulo para cargar y validar configuraciones de pipelines desde YAML.

Este módulo proporciona la funcionalidad de leer archivos YAML que definen
qué tareas ejecutar y en qué orden, permitiendo configurabilidad sin necesidad
de cambiar código.

La configuración YAML define el flujo completo de ejecución de un pipeline,
incluido los parámetros específicos para cada tarea.
"""
import yaml


class Pipeline:
    """Cargador y validador de configuraciones de pipelines YAML.
    
    Lee un archivo de configuración YAML que especifica una secuencia de tareas
    con sus respectivos parámetros, y valida que la estructura sea correcta.
    
    La estructura esperada del YAML es:
    
        tasks:
          - name: nombre_tarea_1
            params:
              parametro1: valor1
              parametro2: valor2
          - name: nombre_tarea_2
            params:
              parametro1: valor1
    
    Attributes
    ----------
    config : dict
        Configuración cargada del archivo YAML con la sección 'tasks' validada.
    
    Examples
    --------
    Cargar una configuración desde un archivo YAML:
    
    >>> pipeline = Pipeline("config.yaml")
    >>> print(pipeline.config['tasks'][0]['name'])
    'resize_images'
    >>> print(pipeline.config['tasks'][0]['params'])
    {'width': 640, 'height': 480}
    
    Notes
    -----
    - La sección 'tasks' es obligatoria
    - Cada tarea debe tener 'name' y 'params' (al menos un diccionario vacío)
    - El validador es estricto para evitar configuraciones inválidas
    
    Raises
    ------
    FileNotFoundError
        Si el archivo YAML no existe.
    yaml.YAMLError
        Si el YAML tiene formato inválido.
    ValueError
        Si la estructura del YAML no cumple con los requisitos.
    """
    
    def __init__(self, config_path):
        """Carga y valida un archivo de configuración YAML.
        
        Abre el archivo YAML, lo analiza y valida que contenga una sección
        'tasks' válida con al menos una tarea correctamente configurada.
        
        Parameters
        ----------
        config_path : str
            Ruta del archivo YAML a cargar (relativa o absoluta).
        
        Raises
        ------
        FileNotFoundError
            Si el archivo en config_path no existe.
        yaml.YAMLError
            Si el YAML tiene sintaxis inválida.
        ValueError
            Si:
            - No existe sección 'tasks'
            - La sección 'tasks' no es una lista
            - La sección 'tasks' está vacía
            - Alguna tarea no tiene 'name'
            - Alguna tarea no tiene 'params'
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if self.config.get('tasks') is None:
            raise ValueError("El archivo de configuración debe contener una sección 'tasks'")
        
        if not isinstance(self.config['tasks'], list):
            raise ValueError("La sección 'tasks' debe ser una lista de tareas")
        
        if len(self.config['tasks']) == 0:
            raise ValueError("La sección 'tasks' no puede estar vacía")
        
        # Validar cada tarea
        for task in self.config['tasks']:
            if task.get('name') is None:
                raise ValueError("Cada tarea debe tener un nombre")
            
            if task.get('params') is None:
                raise ValueError("Cada tarea debe tener parámetros")