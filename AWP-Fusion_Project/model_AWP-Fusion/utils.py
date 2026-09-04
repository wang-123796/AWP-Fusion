import numpy as np
import pandas as pd
import torch
import os
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model_AWP.model import BalancedE2ERegressionNet
from model_AWP.utils import load_sequence_features, extract_weighted_features

def load_balanced_model(weight_path, temp, device):
    model = BalancedE2ERegressionNet(temperature=temp).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def optimize_fusion_weight(pred1, pred2, y_true):
    def loss(w):
        pred = w * pred1 + (1 - w) * pred2
        return mean_squared_error(y_true, pred)
    result = minimize(loss, x0=0.5, bounds=[(0, 1)], method='L-BFGS-B')
    return result.x[0]


def weighted_average(pred1, pred2, w):
    return w * pred1 + (1 - w) * pred2

class KNNExperiment:
    def __init__(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed

    def train_knn_with_tuning(self, X_train, y_train, X_val, y_val):
        best_k = 5
        best_r2 = -np.inf
        k_values = [1, 3, 5, 7, 10, 15, 20]
        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
            knn.fit(X_train, y_train)
            r2 = r2_score(y_val, knn.predict(X_val))
            if r2 > best_r2:
                best_r2 = r2
                best_k = k
        final_knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance', n_jobs=-1)
        final_knn.fit(X_train, y_train)
        return final_knn

    def run(self, train_df_full, test_df, X_train_seq, X_test_seq, 
            X_train_mean, X_test_mean, X_train_weighted, X_test_weighted, 
            target_col, val_size=0.15):
        
        train_df, val_df = train_test_split(train_df_full, test_size=val_size, random_state=self.seed)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        train_idx, val_idx = train_df.index, val_df.index
        
        X_train_weighted_split = X_train_weighted[train_idx]
        X_val_weighted_split = X_train_weighted[val_idx]
        
        scaler_weighted = StandardScaler()
        X_train_weighted_scaled = scaler_weighted.fit_transform(X_train_weighted_split)
        X_val_weighted_scaled = scaler_weighted.transform(X_val_weighted_split)
        X_test_weighted_scaled = scaler_weighted.transform(X_test_weighted)
        
        knn_weighted = self.train_knn_with_tuning(X_train_weighted_scaled, y_train, X_val_weighted_scaled, y_val)
        pred_weighted_val = knn_weighted.predict(X_val_weighted_scaled)
        pred_weighted_test = knn_weighted.predict(X_test_weighted_scaled)
        
        X_train_mean_split = X_train_mean[train_idx]
        X_val_mean_split = X_train_mean[val_idx]
        
        scaler_mean = StandardScaler()
        X_train_mean_scaled = scaler_mean.fit_transform(X_train_mean_split)
        X_val_mean_scaled = scaler_mean.transform(X_val_mean_split)
        X_test_mean_scaled = scaler_mean.transform(X_test_mean)
        
        knn_mean = self.train_knn_with_tuning(X_train_mean_scaled, y_train, X_val_mean_scaled, y_val)
        pred_mean_val = knn_mean.predict(X_val_mean_scaled)
        pred_mean_test = knn_mean.predict(X_test_mean_scaled)
        
        w_opt = optimize_fusion_weight(pred_mean_val, pred_weighted_val, y_val)
        pred_fusion_test = weighted_average(pred_mean_test, pred_weighted_test, w_opt)
        
        return {
            'weighted_pred': pred_weighted_test,  # AWP
            'fusion_pred': pred_fusion_test,      # AWP-Fusion
            'w': w_opt
        }