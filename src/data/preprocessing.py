import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import sys

sys.path.append('..')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _load_data(self, path='../../data/raw/winequality-red.csv'):
        self.data = pd.read_csv(path, sep=';')
        return self
    
    def _clean_data(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == 'quality':
                continue
                
            q1 = self.data[col].quantile(0.25)
            q3 = self.data[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            mean_val = self.data[col].mean()
            self.data.loc[self.data[col] < lower_bound, col] = mean_val
            self.data.loc[self.data[col] > upper_bound, col] = mean_val
        
        return self
    
    def _scale_data(self):
        X = self.data.drop(columns=['quality'])
        y = self.data['quality'].apply(lambda r: 1 if r >= 6.5 else 0)
        
        feature_names = X.columns.tolist()
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        self.data = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
        
        return self
    
    def _test_train_split(self):
        X = self.data.drop(columns=['quality'])
        y = self.data['quality']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        os.makedirs('../../data/processed', exist_ok=True)
        
        train_df = pd.concat([self.X_train, self.y_train], axis=1)
        test_df = pd.concat([self.X_test, self.y_test], axis=1)
        
        train_df.to_csv('../../data/processed/train.csv', index=False)
        test_df.to_csv('../../data/processed/test.csv', index=False)
        
        return self
    
    def fit_transform(self, X=None, y=None):
        if X is not None and y is not None:
            if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
                self.data = pd.concat([X, y], axis=1)
                if y.name:
                    self.data.rename(columns={y.name: 'quality'}, inplace=True)
            else:
                print("X должен быть DataFrame, y должен быть Series или DataFrame")
                return None
        else:
            self._load_data()
        
        self._clean_data()
        self._scale_data()
        self._test_train_split()
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_scaler(self, path='../../models/scaler.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self.scaler.transform(X)