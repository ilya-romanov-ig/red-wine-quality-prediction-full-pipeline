import joblib
import numpy as np
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
import sys
import os

sys.path.append(os.path.abspath('../'))

from src.monitoring.mlflow_manager import MLflowManager


class ModelTrainer:
    def __init__(self, cv_folds=5, use_mlflow=False, mlflow_config=None):
       
        self.model = None
        self.cv_folds = cv_folds
        self.cv_results_ = None
        self.is_trained = False
        self.use_mlflow = use_mlflow
        
        if use_mlflow:
            self.mlflow_manager = MLflowManager(
                tracking_uri=mlflow_config.get('tracking_uri') if mlflow_config else None,
                experiment_name=mlflow_config.get('experiment_name') if mlflow_config else None
            )
        else:
            self.mlflow_manager = None
    
    def train(self, model, X_train, y_train, X_test=None, y_test=None,
              model_name="model", params=None, use_cv=False,
              grid_search=None):
        
        if self.use_mlflow:
            run_name = f"{model_name}_{np.random.randint(1000)}"
            run_tags = {"model_type": type(model).__name__, "task": "classification"}
            
            with self.mlflow_manager.start_run(run_name=run_name, tags=run_tags):
                return self._train_logic(model, X_train, y_train, X_test, y_test,
                                        model_name, params, use_cv, grid_search)
        else:
            # Без MLflow
            return self._train_logic(model, X_train, y_train, X_test, y_test,
                                    model_name, params, use_cv, grid_search)
    
    def _train_logic(self, model, X_train, y_train, X_test=None, y_test=None,
                    model_name="model", params=None, use_cv=False,
                    grid_search=None):
        
        try:
            if self.use_mlflow:
                base_params = {
                    "model": type(model).__name__,
                    "cv_folds": self.cv_folds,
                    "train_samples": len(X_train),
                    "n_features": X_train.shape[1]
                }
                
                if params:
                    base_params.update(params)
                
                self.mlflow_manager.log_params(base_params)
            
            if grid_search:
                gs = GridSearchCV(
                    model, 
                    grid_search['param_grid'],
                    cv=grid_search.get('cv', self.cv_folds),
                    scoring=grid_search.get('scoring', 'accuracy'),
                    n_jobs=grid_search.get('n_jobs', -1)
                )
                gs.fit(X_train, y_train)
                self.model = gs.best_estimator_
                
                if self.use_mlflow:
                    self.mlflow_manager.log_params({"best_params": str(gs.best_params_)})
                    self.mlflow_manager.log_metrics(
                        {"best_cv_score": gs.best_score_}
                    )
            else:
                self.model = model
            
            if use_cv:
                cv_results = cross_validate(
                    self.model, X_train, y_train,
                    cv=self.cv_folds,
                    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                    return_train_score=True,
                    return_estimator=False
                )
                self.cv_results_ = cv_results
                
                if self.use_mlflow:
                    cv_metrics = {}
                    for key in cv_results:
                        if 'test' in key and 'time' not in key:
                            metric_name = key.replace('test_', 'cv_')
                            cv_metrics[f"{metric_name}_mean"] = np.mean(cv_results[key])
                            cv_metrics[f"{metric_name}_std"] = np.std(cv_results[key])
                    
                    self.mlflow_manager.log_metrics(cv_metrics)
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            if self.use_mlflow:
                signature = infer_signature(X_train[:10], self.model.predict(X_train[:10]))
                
                self.mlflow_manager.log_model(
                    model=self.model,
                    model_name=model_name,
                    signature=signature,
                    input_example=X_train[:5] if len(X_train) > 5 else X_train
                )
                
                if X_test is not None and y_test is not None:
                    y_pred = self.model.predict(X_test)
                    test_acc = accuracy_score(y_test, y_pred)
                    self.mlflow_manager.log_metrics({"test_accuracy": test_acc})
            
            return self
            
        except Exception as e:
            if self.use_mlflow:
                try:
                    self.mlflow_manager.log_params({"error": str(e)})
                except:
                    print(f"Не удалось залогировать ошибку в MLflow: {e}")
            raise e
    
    def save(self, model_path):
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        
        return self