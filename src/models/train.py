import joblib

def train(estimator, X, y, params=None, random_state=42):
    """
    Final model training
    
    :param estimator: Baseline estimator
    :param X: Train data
    :param y: Train target
    :param params: Model params
    :param random_state: random_state
    """
    if params is None:
        params = {}

    try:
        model = estimator(**params, random_state=random_state)
        model.fit(X, y)
        return model
    except Exception as e:
        return f'Error! {e}'

def save_model(model, path):
    joblib.dump(model, path)