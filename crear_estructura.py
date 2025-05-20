import os

estructura = {
    "README.md": "",
    "requirements.txt": "",
    "data/": {
        "dataset_raw/": {},
        "dataset_processed/": {}
    },
    "src/": {
        "__init__.py": "",
        "data_collection.py": "",
        "video_processor.py": "",
        "feature_extractor.py": "",
        "model_training.py": "",
        "visualization.py": "",
        "utils.py": ""
    },
    "notebooks/": {
        "data_exploration.ipynb": "",
        "model_evaluation.ipynb": ""
    },
    "app/": {
        "__init__.py": "",
        "main.py": "",
        "gui.py": ""
    },
    "docs/": {
        "plan_despliegue.md": "",
        "impacto_solucion.md": "",
        "reporte_segunda_entrega.md": ""
    }
}

def crear_estructura(base_path, estructura):
    for nombre, contenido in estructura.items():
        ruta = os.path.join(base_path, nombre)
        if isinstance(contenido, dict):
            os.makedirs(ruta, exist_ok=True)
            crear_estructura(ruta, contenido)
        else:
            with open(ruta, 'w', encoding='utf-8') as f:
                f.write(contenido)

if __name__ == "__main__":
    crear_estructura(".", estructura)
    print("Estructura de proyecto creada correctamente.")
