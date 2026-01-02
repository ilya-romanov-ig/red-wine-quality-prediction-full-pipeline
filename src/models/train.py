import joblib

def train_model(estimator, X, y, params=None, random_state=42, fit=True):
    """
    Model train method
    
    :param estimator: Baseline estimator
    :param X: Train data
    :param y: Train target
    :param params: Model params
    :param random_state: random_state
    :param fit: Use fit method for model
    """
    params = params or {}

    model = estimator(**params, random_state=random_state)

    if fit:
        model.fit(X, y)
    return model

def save_model(model, path):
    joblib.dump(model, path)