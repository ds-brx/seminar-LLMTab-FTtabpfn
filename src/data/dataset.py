import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.y_encoder = LabelEncoder()
        self.n_features = 0  # Number of numerical features
        self.num_cls = 0  # Number of classes in target variable y
        self.cardinalities = []  # List to store cardinality for each categorical feature
        self.nan_numerical = 0  # Number of NaNs in numerical features
        self.nan_categorical = 0  # Number of NaNs in categorical features

    def fit(self, X, y):
        # Identify numerical and categorical features
        self.num_features = X.select_dtypes(include=['number']).columns
        self.cat_features = X.select_dtypes(include=['category', 'object']).columns
        
        # Store the number of numerical features
        self.n_features = len(self.num_features)
        
        # Count NaNs in numerical and categorical features
        self.nan_numerical = X[self.num_features].isna().sum().sum()
        self.nan_categorical = X[self.cat_features].isna().sum().sum()
        
        # Drop rows with missing numerical values
        X_clean = X.dropna(subset=self.num_features)
        y = y.loc[X_clean.index]
        
        # Fit scaler
        self.scaler.fit(X_clean[self.num_features])
        
        # Fit label encoders for categorical features
        for col in self.cat_features:
            X_clean[col] = X_clean[col].astype('category').cat.add_categories(['Unknown']).fillna('Unknown')
            le = LabelEncoder()
            le.fit(X_clean[col])
            self.label_encoders[col] = le
        
        # Record cardinality for each categorical feature
        self.cardinalities = [len(le.classes_) for le in self.label_encoders.values()]
        
        # Fit label encoder for target variable
        self.y_encoder.fit(y)
        self.num_cls = len(self.y_encoder.classes_)
        return X_clean, y

    def transform(self, X, y=None):
        X_num = self.scaler.transform(X[self.num_features])
        X_cat = X[self.cat_features].copy()
        
        for col, le in self.label_encoders.items():
            X_cat[col] = le.transform(X_cat[col])
        
        y_transformed = self.y_encoder.transform(y) if y is not None else None
        
        return X_num, X_cat.values, y_transformed

    def inverse_transform_y(self, y_encoded):
        return self.y_encoder.inverse_transform(y_encoded)


class FT_Dataset(Dataset):
    def __init__(self, X_num, X_cat, y):

        self.X_num = torch.tensor(X_num, dtype=torch.float32)  
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)     
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]