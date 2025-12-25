import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, mean_squared_error, r2_score,
    mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import mlflow
import os


class ModelEvaluator:
    def __init__(self, use_mlflow=False):
        self.metrics_history = []
        self.feature_importance = None
        self.use_mlflow = use_mlflow
    
    def _extract_feature_importance(self, model):
        try:
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = model.feature_importances_
                print(f"Feature importance extracted from model (shape: {self.feature_importance.shape})")
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if len(coef.shape) > 1:
                    self.feature_importance = np.abs(coef).mean(axis=0)
                else:
                    self.feature_importance = np.abs(coef)
                print(f"Feature importance extracted from coefficients (shape: {self.feature_importance.shape})")
            else:
                self.feature_importance = None
                print("Model doesn't support feature importance extraction")
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            self.feature_importance = None

    def evaluate_classification(self, model, X_test, y_test, 
                                run_name=None, tags=None,
                                average='weighted', save_results=False):
        if self.use_mlflow:
            import mlflow
            if run_name is None:
                run_name = f"eval_{type(model).__name__}"
            
            mlflow.start_run(run_name=run_name, tags=tags)
            
            mlflow.log_param("evaluation_task", "classification")
            mlflow.log_param("average_method", average)
        
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average=average),
                'recall': recall_score(y_test, y_pred, average=average),
                'f1': f1_score(y_test, y_pred, average=average),
            }
            
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

                except:
                    pass
            
            report = classification_report(y_test, y_pred, output_dict=True)
            metrics['classification_report'] = report
            
            if self.use_mlflow:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
            self._extract_feature_importance(model)
            
            if save_results:
                self.metrics_history.append(metrics)
            
            return metrics
            
        finally:
            if self.use_mlflow:
                mlflow.end_run()
    
    def evaluate_regression(self, model, X_test, y_test, 
                           run_name=None, tags=None,
                           save_results=False):
        if self.use_mlflow:
            import mlflow
            mlflow.start_run(run_name=run_name or f"eval_reg_{type(model).__name__}", tags=tags)
            mlflow.log_param("evaluation_task", "regression")
        
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'predictions': y_pred.tolist()
            }
            
            if self.use_mlflow:
                for key, value in metrics.items():
                    if key != 'predictions' and isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
            if save_results:
                self.metrics_history.append(metrics)
            
            return metrics
            
        finally:
            if self.use_mlflow:
                mlflow.end_run()