import pandas as pd
import json
from pathlib import Path
from xgboost.sklearn import XGBClassifier
from src.models.train import train_model, save_model
from src.models.optimize import optimize
from src.data.preprocessing import DataPreprocessor
from src.data.data_loader import load_data
from src.models.evaluate import eval

def run_pipeline(toRetrain=False):
    root_path = Path(__file__).resolve().parent.parent

    baseline_model = XGBClassifier()
    dp = DataPreprocessor()

    if toRetrain:
        data = load_data(root_path / 'data' / 'raw' / 'data.csv')
        X_train, X_test, y_train, y_test = dp.fit_transform(data, save_path=root_path / 'data' / 'processed')
        params = optimize(X_train, y_train, save_path=root_path / 'models' / 'best_params.json')
    else:
        train_data = load_data(root_path / 'data' / 'processed' / 'train.csv')
        test_data = load_data(root_path / 'data' / 'processed' / 'test.csv')
        X_train = train_data.drop('quality')
        y_train = train_data['quality']
        X_test = test_data.drop('quality')
        y_test = test_data['quality']
        params = json.load(root_path / 'models' / 'best_params.json')
    
    model = train_model(baseline_model, X_train, y_train, params)

    save_model(model, root_path / 'models' / 'trained_XGB_model.pkl')

    metrics = eval(model, X_test, y_test, metrics_path=root_path / 'reports' / 'metrics.json')

    return