import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from datetime import datetime
import pandas as pd
from pathlib import Path

class MLflowManager:
    def __init__(self, experiment_name=None, tracking_uri=None):

        current_file = Path(__file__).resolve()
        src_dir = current_file.parent
        project_root = src_dir.parent 
        mlflow_dir = project_root / "mlflow"
        
        print(f"Project root: {project_root}")
        print(f"MLflow directory: {mlflow_dir}")
        
        if tracking_uri is None:
            mlruns_dir = mlflow_dir / "mlruns"
            mlruns_dir.mkdir(parents=True, exist_ok=True)
            
            if os.name == 'nt': 
                tracking_uri = f"file:///{mlruns_dir.resolve()}".replace('\\', '/')
            else: 
                tracking_uri = f"file://{mlruns_dir.resolve()}"
            
            print(f"Auto-configured tracking URI: {tracking_uri}")
        
        elif isinstance(tracking_uri, str) and tracking_uri.startswith('file://'):
            path_part = tracking_uri[7:]
            if not os.path.isabs(path_part):
                absolute_path = (project_root / path_part).resolve()
                
                if os.name == 'nt':  # Windows
                    tracking_uri = f"file:///{absolute_path}".replace('\\', '/')
                else:
                    tracking_uri = f"file://{absolute_path}"
                
                print(f"Converted to absolute path: {tracking_uri}")
        
        print(f"Final tracking URI: {tracking_uri}")
        
        try:
            mlflow.set_tracking_uri(tracking_uri)
            print(f"Successfully set tracking URI")
            
            current_tracking_uri = mlflow.get_tracking_uri()
            print(f"Current MLflow tracking URI: {current_tracking_uri}")
            
        except Exception as e:
            print(f"Warning: Could not set tracking URI {tracking_uri}: {e}")
            print("Falling back to local ./mlruns")
            
            fallback_dir = project_root / "mlruns_fallback"
            fallback_dir.mkdir(exist_ok=True)
            
            if os.name == 'nt':
                fallback_uri = f"file:///{fallback_dir.resolve()}".replace('\\', '/')
            else:
                fallback_uri = f"file://{fallback_dir.resolve()}"
            
            mlflow.set_tracking_uri(fallback_uri)
            tracking_uri = fallback_uri
        
        if experiment_name:
            self.experiment_name = experiment_name
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    print(f"Experiment '{experiment_name}' does not exist. Creating it.")
                    experiment_id = mlflow.create_experiment(experiment_name)
                    experiment = mlflow.get_experiment(experiment_id)
                else:
                    print(f"Experiment '{experiment_name}' found (ID: {experiment.experiment_id})")
                mlflow.set_experiment(experiment_name=experiment_name)
                print(f"Active experiment set to: {experiment_name}")
            except Exception as e:
                print(f"Error during experiment setup: {e}")
                raise
        else:
            self.experiment_name = "Default"
            mlflow.set_experiment(experiment_name="Default")
                
        self.client = MlflowClient()
        self.tracking_uri = tracking_uri
        
        self._check_connection()
        
    def _check_connection(self):
        try:
            experiments = mlflow.search_experiments()
            print(f"MLflow connection successful. Found {len(experiments)} experiments.")
            return True
        except Exception as e:
            print(f"MLflow connection warning: {e}")
            print("MLflow will work in local mode.")
            return False

    def start_run(self, run_name=None, tags=None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return mlflow.start_run(run_name=run_name, tags=tags)

    def end_run(self):
        mlflow.end_run()

    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=step)

    def log_model(self, model, model_name, signature=None, input_example=None):
        mlflow.sklearn.log_model(
            sk_model=model,
            name=model_name,
            signature=signature,
            input_example=input_example
        )
    
    def log_params(self, params):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def get_best_run(self, experiment_name=None, metric="accuracy", mode="max"):
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"]
        )
        
        if runs.empty:
            return None
        
        best_run = runs.iloc[0]
        return {
            'run_id': best_run.run_id,
            'experiment_id': best_run.experiment_id,
            'metrics': {k.replace('metrics.', ''): v 
                       for k, v in best_run.items() 
                       if k.startswith('metrics.') and pd.notna(v)},
            'params': {k.replace('params.', ''): v 
                      for k, v in best_run.items() 
                      if k.startswith('params.') and pd.notna(v)},
            'artifact_uri': best_run.artifact_uri
        }