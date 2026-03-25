"""Módulo para gestionar el registro dinámico de tareas.

Este módulo proporciona el sistema de registro que permite:
- Registrar nuevas tareas de forma dinámica
- Recuperar tareas registradas por nombre
- Validar que las tareas cumplen con la interfaz Task

El patrón Registry es fundamental para desacoplar la definición de tareas
de su instanciación y ejecución.
"""
from app.core.task import Task


class registry:
    """Gestor centralizado del registro de tareas.
    
    Implementa el patrón Registry para almacenar y gestionar referencias
    a todas las clases de tareas disponibles en el sistema.
    
    Esta clase actúa como un diccionario tipado que garantiza que solo
    se registren clases que heredan de Task.
    
    Attributes
    ----------
    task_registry : dict
        Diccionario donde la clave es el nombre de la tarea (str) y el valor
        es la clase de la tarea (subclase de Task).
    
    Examples
    --------
    Registrar y recuperar tareas:
    
    >>> reg = registry()
    >>> reg.register_task(MiTarea)
    >>> task_class = reg.task_registry["mi_tarea"]
    >>> tarea = task_class({"param1": "valor"})
    
    Notes
    -----
    - El registro es local a la instancia de registry
    - Para usar un registro global, crear una instancia única y compartirla
    - Las tareas se identifican por su atributo de clase `name`
    
    See Also
    --------
    Task : Clase base que deben heredar todas las tareas
    Pipeline : Carga configuraciones YAML que especifican qué tareas ejecutar
    """

    def __init__(self):
        """Inicializa un registro vacío de tareas.
        
        Crea un diccionario vacío para almacenar las tareas registradas.
        """
        self.task_registry = {}

    def register_task(self, task_class):
        """Registra una clase de tarea en el registry.
        
        Valida que la clase sea una subclase de Task y la añade al registro
        usando su atributo `name` como clave.
        
        Parameters
        ----------
        task_class : type
            Clase de la tarea a registrar. Debe ser una subclase de Task
            y tener un atributo de clase `name`.
        
        Raises
        ------
        ValueError
            Si la clase no es una subclase de Task.
        AttributeError
            Si la clase no tiene el atributo `name`.
        
        Examples
        --------
        >>> class MiTarea(Task):
        ...     name = "mi_tarea"
        ...     def run(self):
        ...         pass
        >>> reg = registry()
        >>> reg.register_task(MiTarea)  # Registración exitosa
        
        Notes
        -----
        - Si una tarea con el mismo nombre ya existe, será sobrescrita
        - La clase se almacena por referencia, no por instancia
        """
        if not issubclass(task_class, Task):
            raise ValueError("La clase debe ser una subclase de Task")
        self.task_registry[task_class.name] = task_class

    def register_tasks_from_list(self, task_names, available_tasks):
        """Registro selectivo de tareas desde nombre según lista de pipeline.

        Parameters
        ----------
        task_names : list[str]
            Lista de nombres de tareas declaradas en la configuración (YAML).
        available_tasks : dict
            Diccionario de todas las clases de tareas disponibles {name: clase}.

        Returns
        -------
        dict
            Un diccionario con llaves 'registered' y 'missing' conteniendo
            listas de nombres.

        Notes
        -----
        - Solo registra tareas presentes en available_tasks.
        - Omite tareas no reconocidas, sin lanzar excepción.
        """
        registered = []
        missing = []
        for task_name in task_names:
            if task_name in available_tasks:
                if task_name not in self.task_registry:
                    self.register_task(available_tasks[task_name])
                registered.append(task_name)
            else:
                missing.append(task_name)
        return {
            "registered": registered,
            "missing": missing,
        }