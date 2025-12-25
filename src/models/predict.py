import joblib
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.data.preprocessing import DataPreprocessor
    from src.monitoring.mlflow_manager import MLflowManager
    from src.models.evaluate import ModelEvaluator
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")


class ModelPredictor:
    def __init__(self, models_dir="../saved-models", use_mlflow=False):
        self.models_dir = Path(models_dir)
        self.use_mlflow = use_mlflow
        self.loaded_models = {}
        self.preprocessor = None
        self.evaluator = ModelEvaluator(use_mlflow=False)
        
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        if use_mlflow:
            try:
                self.mlflow_manager = MLflowManager(
                    experiment_name="Production_Predictions"
                )
                print("MLflow initialized for predictions tracking")
            except Exception as e:
                print(f"Could not initialize MLflow: {e}")
                self.use_mlflow = False
                self.mlflow_manager = None
    
    def load_model(self, model_name, model_path=None):
        if model_path is None:
            possible_paths = [
                self.models_dir / f"{model_name}.pkl",
                self.models_dir / f"{model_name}.joblib",
                self.models_dir / f"{model_name}_model.pkl",
            ]
            
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"Model file for '{model_name}' not found in {self.models_dir}"
                )
        else:
            model_path = Path(model_path)
        
        print(f"Loading model from: {model_path}")
        
        try:
            model = joblib.load(model_path)
            self.loaded_models[model_name] = {
                'model': model,
                'path': str(model_path),
                'type': type(model).__name__
            }
            
            print(f"Model '{model_name}' loaded successfully")
            print(f"   Model type: {type(model).__name__}")
            
            if hasattr(model, 'n_features_in_'):
                print(f"   Expected features: {model.n_features_in_}")
            if hasattr(model, 'classes_'):
                print(f"   Classes: {len(model.classes_)}")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_latest_mlflow_model(self, model_name, stage="Production"):
        if not self.use_mlflow:
            raise ValueError("MLflow is not enabled. Enable with use_mlflow=True")
        
        try:
            import mlflow
            
            model_uri = f"models:/{model_name}/{stage}"
            print(f"Loading model from MLflow: {model_uri}")
            
            model = mlflow.sklearn.load_model(model_uri)
            
            self.loaded_models[f"{model_name}_{stage}"] = {
                'model': model,
                'path': model_uri,
                'type': type(model).__name__,
                'source': 'mlflow'
            }
            
            print(f"Model '{model_name}' (stage: {stage}) loaded from MLflow")
            return model
            
        except Exception as e:
            print(f"Error loading model from MLflow: {e}")
            raise
    
    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor
        print("Preprocessor set")
    
    def predict(self, model_name, X, return_proba=False, feature_names=None):
        if model_name not in self.loaded_models:
            raise KeyError(f"Model '{model_name}' not loaded. Load it first with load_model()")
        
        model_info = self.loaded_models[model_name]
        model = model_info['model']
        
        if not isinstance(X, pd.DataFrame):
            if feature_names is not None:
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()
        
        print(f"Making predictions with model: {model_name}")
        print(f"Input shape: {X_df.shape}")
        
        if self.preprocessor is not None:
            print("Applying preprocessing...")
            try:
                X_processed = self.preprocessor.transform(X_df)
                if isinstance(X_processed, np.ndarray) and feature_names is not None:
                    X_processed = pd.DataFrame(X_processed, columns=feature_names)
            except Exception as e:
                print(f"Preprocessing failed: {e}. Using raw data.")
                X_processed = X_df
        else:
            X_processed = X_df
        
        try:
            predictions = model.predict(X_processed)
            
            if self.use_mlflow:
                self._log_prediction(
                    model_name=model_name,
                    input_data=X_df,
                    predictions=predictions,
                    model_type=model_info['type']
                )
            
            if return_proba and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_processed)
                print(f"Predictions complete. Returning {len(predictions)} predictions with probabilities")
                return predictions, probabilities
            else:
                if return_proba and not hasattr(model, 'predict_proba'):
                    print("Model doesn't support predict_proba. Returning only predictions.")
                print(f"Predictions complete. Returning {len(predictions)} predictions")
                return predictions
                
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, model_name, batch_generator, batch_size=100):
        all_predictions = []
        
        for i, batch in enumerate(batch_generator):
            print(f"Processing batch {i+1}...")
            batch_preds = self.predict(model_name, batch, return_proba=False)
            all_predictions.extend(batch_preds)
        
        print(f"Batch prediction complete. Total: {len(all_predictions)} predictions")
        return np.array(all_predictions)
    
    def _log_prediction(self, model_name, input_data, predictions, model_type):

        if not self.use_mlflow or self.mlflow_manager is None:
            return
        
        try:
            with self.mlflow_manager.start_run(run_name=f"prediction_{model_name}"):
                self.mlflow_manager.log_params({
                    "prediction_model": model_name,
                    "model_type": model_type,
                    "n_predictions": len(predictions),
                    "input_shape": input_data.shape
                })
                
                if predictions.dtype.kind in 'iuf':
                    pred_stats = {
                        "predictions_mean": float(predictions.mean()),
                        "predictions_std": float(predictions.std()),
                        "predictions_min": float(predictions.min()),
                        "predictions_max": float(predictions.max())
                    }
                    self.mlflow_manager.log_metrics(pred_stats)
                
                if predictions.dtype.kind in 'i':
                    unique, counts = np.unique(predictions, return_counts=True)
                    for cls, count in zip(unique, counts):
                        self.mlflow_manager.log_metrics({
                            f"class_{cls}_count": int(count)
                        })
                
                print("Prediction logged to MLflow")
                
        except Exception as e:
            print(f"⚠️  Failed to log prediction to MLflow: {e}")
    
    def evaluate_on_data(self, model_name, X_test, y_test, metrics=None):
        if model_name not in self.loaded_models:
            raise KeyError(f"Model '{model_name}' not loaded")
        
        model = self.loaded_models[model_name]['model']
        
        metrics_result = self.evaluator.evaluate_classification(
            model=model,
            X_test=X_test,
            y_test=y_test,
            run_name=f"eval_{model_name}",
            save_results=True
        )
        
        print(f"Evaluation complete for model: {model_name}")
        print(f"   Accuracy: {metrics_result.get('accuracy', 'N/A'):.4f}")
        if 'f1' in metrics_result:
            print(f"   F1 Score: {metrics_result['f1']:.4f}")
        
        return metrics_result
    
    def get_model_info(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        else:
            raise KeyError(f"Model '{model_name}' not found")
    
    def list_available_models(self):
        model_files = []
        for ext in ['.pkl', '.joblib']:
            model_files.extend(list(self.models_dir.glob(f"*{ext}")))
        
        return [f.stem for f in model_files]
