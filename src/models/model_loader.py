import joblib

def load_model(model_path):
    """
    Trainder model loader
    
    :param model_path: Path to model.pkl
    """
    model = joblib.load(model_path)
    return model