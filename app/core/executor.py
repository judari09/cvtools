"""Módulo para ejecutar tareas en base a la configuración de pipeline."""


class executor():
    """Clase encargada de ejecutar el flujo de tareas del pipeline.

    La clase `executor` recorre la lista de definiciones de tareas (como
    la cargada por `Pipeline`) y ejecuta todas las tareas que están registradas
    en el `registry`.

    Attributes
    ----------
    (No tiene atributos persistentes en esta versión).  

    Notes
    -----
    - No se encarga de registrar tareas; la responsabilidad de registrar
      recae en la clase `registry`.
    - Si alguna tarea listada en `tasks_configs` no está en el registro,
      es ignorada con aviso de consola.
    """

    def __init__(self):
        """Inicializa el ejecutor.

        Mantiene la misma interfaz simple que permite reutilizarlo de forma
        stateless en múltiples ejecuciones.
        """
        pass

    def execute(self, tasks_configs, task_reg):
        """Ejecuta la lista de tareas definidas en el pipeline.

        Parameters
        ----------
        tasks_configs : list[dict]
            Lista de configuraciones de tareas con las claves `name` y `params`.
        task_reg : registry
            Instancia de `registry` que contiene las tareas registradas.

        Returns
        -------
        None

        Notes
        -----
        - Cada tarea se instancia con los parámetros definidos y se ejecuta
          mediante su método `run()`.
        - Las tareas no registradas se omiten con un mensaje de advertencia.
        """
        for task_config in tasks_configs:
            task_name = task_config['name']
            task_params = task_config['params']

            if task_name in task_reg.task_registry:
                task_class = task_reg.task_registry[task_name]
                task = task_class(task_params)
                task.run()
            else:
                print(f"⚠ Tarea '{task_name}' no está registrada; se omite.")