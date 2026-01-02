def predict(model, data):
    """
    Prediction method for data
    
    :param model: Model
    :param data: Data: 
    """
    preds = model.predict(data)
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(data)[:, 1]
    return preds, probs