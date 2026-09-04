import sys, os
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model_AWP.utils import load_sequence_features, set_seed

def load_mean_features(df, feat_dir):
    """Load mean pooled features (1280-dim)"""
    feats = []
    for pid in df["uniprot_id"]:
        x = torch.load(f"{feat_dir}/{pid}.pt", map_location="cpu")
        feats.append(x.mean(dim=0).numpy())
    return np.array(feats)

class KNNBaselineExperiment:
    def __init__(self, seed=42):
        set_seed(seed)
        self.seed = seed
        self.best_k = None

    def run(self, train_df_full, test_df, train_feat_dir, test_feat_dir, target_col):
        train_df, val_df = train_test_split(train_df_full, test_size=0.15, random_state=self.seed)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        X_train_mean = load_mean_features(train_df, train_feat_dir)
        X_val_mean = load_mean_features(val_df, train_feat_dir)
        X_test_mean = load_mean_features(test_df, test_feat_dir)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_mean)
        X_val_scaled = scaler.transform(X_val_mean)
        X_test_scaled = scaler.transform(X_test_mean)
        
        k_values = [1, 3, 5, 7, 10, 15, 20]
        best_r2 = -float('inf')
        best_k = 5
        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
            knn.fit(X_train_scaled, y_train)
            val_r2 = r2_score(y_val, knn.predict(X_val_scaled))
            if val_r2 > best_r2:
                best_r2 = val_r2
                best_k = k
        
        self.best_k = best_k
        
        final_knn = KNeighborsRegressor(n_neighbors=self.best_k, weights='distance', n_jobs=-1)
        final_knn.fit(X_train_scaled, y_train)  
        
        pred_test = final_knn.predict(X_test_scaled)
        
        return {
            'best_k': self.best_k,
            'pred_test': pred_test,
            'y_test': y_test
        }