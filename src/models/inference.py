def predict(model, data):
    """
    Prediction method for data
    
    :param model: Model
    :param data: Data: 
    """
    try:
        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1]
        return preds, probs
    except Exception as e:
        return f'Error! {e}'