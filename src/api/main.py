from fastapi import FastAPI
from src.api.models.data_model import Data
from src.api.models.prediction_model import Prediction
from src.pipeline.inference_pipeline import run_pipeline as run_inference_pipeline
from src.pipeline.trainnig_pipeline import run_pipeline as run_training_pipline

app = FastAPI()

@app.get("/predict")
async def predict(data: Data):
    response = Prediction()
    response.value, response.prob = run_inference_pipeline(data)
    return {'prediction': response.value, 'probability': response.prob}

@app.get("/retrain")
async def retrain():
    run_training_pipline(toRetrain=True)
    return {'status': 'retrained'}

@app.get('/health')
async def check_health():
    return {'status' : 'healthy'}