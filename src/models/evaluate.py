import os
import json
from sklearn.metrics import accuracy_score, roc_auc_score
from models.inference import predict

def eval(model, X_test, y_test, metrics_path='reports/metrics.json'):
    """
    Evaluation method
    
    :param model: Trained model
    :param X_test: Test data
    :param y_test: Test target
    :param metrics_path: Path to reports dir
    """
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    preds, probs = predict(model, X_test)

    metrics = {
        'accuracy_score': accuracy_score(y_test, preds),
    }

    if probs is not None:
        metrics['roc_auc_score'] = roc_auc_score(y_test, probs)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics
    

