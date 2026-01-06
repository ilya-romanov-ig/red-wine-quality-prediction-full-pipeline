import pandas as pd
from pydantic import BaseModel

class Prediction(BaseModel):
    value: float
    prob: float