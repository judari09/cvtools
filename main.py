from app.core.pipeline import Pipeline
from app.core.registry import registry
from app.core.executor import executor


def main():
    print("=== CVTools - Sistema de Tareas ===\n")

    task_reg = registry()

    print("Descubriendo módulos disponibles...")
    result = task_reg.autodiscover('app.modules')
    print(f"✓ {len(result['registered'])} tareas registradas: {result['registered']}")
    if result['skipped']:
        print(f"⚠ {len(result['skipped'])} módulos omitidos (dependencias no instaladas)")
    print()

    print("Cargando configuración desde YAML...")
    try:
        pipeline = Pipeline("config_example.yaml")
        print("✓ Configuración cargada exitosamente")
        print(f"Tareas en la configuración: {[task['name'] for task in pipeline.config['tasks']]}\n")

        print("Ejecutando tareas...\n")
        executor().execute(pipeline.config['tasks'], task_reg)
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
