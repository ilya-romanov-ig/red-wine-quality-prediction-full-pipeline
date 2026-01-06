from pathlib import Path
from src.models.inference import predict
from src.models.model_loader import load_model

def run_pipeline(data):
    root_path = Path(__file__).resolve().parent.parent

    model = load_model(root_path / 'models' / 'trained_XGB_model.pkl')

    preds, probs = predict(model, data)

    return preds, probs