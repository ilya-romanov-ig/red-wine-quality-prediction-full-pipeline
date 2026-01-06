import optuna
import json
import pandas as pd
from sklearn.model_selection import cross_val_score
from models.train import train_model
from xgboost.sklearn import XGBClassifier

def objective(trial, X, y):
    """
    Docstring для objective
    
    :param trial: Описание
    :param X: Training data
    :param y: Training target
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 32),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }

    clf = train_model(XGBClassifier(), X, y, params=params, fit=False)
    score = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()

    return score

def optimize(X, y, save_path='models/best_params.json'):
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    
    study.optimize(lambda t: objective(t, X, y), n_trials=100)

    with open(save_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    return study.best_params