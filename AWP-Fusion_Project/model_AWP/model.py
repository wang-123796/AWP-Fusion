import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class BalancedWeightedMeanPooling(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=32, dropout=0.6, temperature=3.0):
        super().__init__()
        self.temperature = temperature
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.dropout(h)
        scores = self.fc2(h).squeeze(-1)
        weights = torch.softmax(scores / self.temperature, dim=0)
        weighted_feat = torch.sum(weights.unsqueeze(-1) * x, dim=0)
        return weighted_feat, weights

class BalancedE2ERegressionNet(nn.Module):
    def __init__(self, input_dim=1280, pool_hidden=32, head_hidden=128, dropout=0.6, temperature=3.0):
        super().__init__()
        self.pooling = BalancedWeightedMeanPooling(input_dim, pool_hidden, dropout, temperature)
        self.head = nn.Sequential(
            nn.Linear(input_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1)
        )
    
    def forward(self, x):
        feat, weights = self.pooling(x)
        pred = self.head(feat).squeeze(-1)
        return pred, feat, weights

class AWP_KNN_Regressor:
    def __init__(self, k_values=[1, 3, 5, 7, 10, 15, 20, 25, 30]):
        self.k_values = k_values
        self.best_k = None
        self.scaler = StandardScaler()
        self.model = None

    def tune_and_train(self, X_train, y_train, X_val, y_val):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        best_r2 = -float('inf')
        best_k = self.k_values[0]
        for k in self.k_values:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
            knn.fit(X_train_scaled, y_train)
            pred_val = knn.predict(X_val_scaled)
            r2 = r2_score(y_val, pred_val)
            if r2 > best_r2:
                best_r2 = r2
                best_k = k
        
        self.best_k = best_k
        print(f"  -> Best K found: {self.best_k} (Val R2 = {best_r2:.4f})")

        self.model = KNeighborsRegressor(n_neighbors=self.best_k, weights='distance', n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call tune_and_train first.")
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)