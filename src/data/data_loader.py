import pandas as pd

def load_data(data_path='data/raw/data.csv'):
    data = pd.read_csv(data_path)

    return data