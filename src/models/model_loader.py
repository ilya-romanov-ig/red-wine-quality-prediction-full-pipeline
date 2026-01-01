import joblib

def load_model(model_path):
    """
    Trainder model loader
    
    :param model_path: Path to model
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        return f'Error {e}'