import os
from sklearn.metrics import accuracy_score, roc_auc_score
import json

def eval(model, X_test, y_test, metrics_path='reports/metrics.json'):
    """
    Evaluation method
    
    :param model: Trained model
    :param X_test: Test data
    :param y_test: Test target
    :param metrics_path: Path to reports dir
    """
    os.makedirs('reports', exist_ok=True)

    try:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy_score': accuracy_score(y_test, preds),
            'roc_auc_score': roc_auc_score(y_test, probs)
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        return metrics
    except Exception as e:
        return f'Error! {e}'
    

