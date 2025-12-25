import os
from pathlib import Path

current_file = Path(__file__).resolve()
src_dir = current_file.parent
project_root = src_dir.parent  

print(f"Config: Project root detected at {project_root}")

MLFLOW_CONFIG = {
    'tracking_uri': None,
    'experiment_name': 'Wine_Quality_Experiment',
    'run_tags': {
        'project': 'red-wine-quality-prediction',
        'dataset': 'winequality-red',
        'version': '1.0'
    }
}

MLFLOW_CONFIG_EXPLICIT = {
    'tracking_uri': f"file:///{project_root}/mlflow/mlruns".replace('\\', '/'),
    'experiment_name': 'Wine_Quality_Explicit',
}

MLFLOW_CONFIG_SQLITE = {
    'tracking_uri': f"sqlite:///{project_root}/mlflow.db",
    'experiment_name': 'Wine_Quality_SQLite',
}

ACTIVE_CONFIG = MLFLOW_CONFIG  

print(f"Config: Using MLflow directory at {project_root}/mlflow/mlruns")