import pandas as pd
from pydantic import BaseModel
from typing import Optional

class Data(BaseModel):
    X: Optional[pd.DataFrame]