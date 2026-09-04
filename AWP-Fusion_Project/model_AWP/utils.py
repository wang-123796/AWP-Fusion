import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_sequence_features(df, feat_dir):
    seq_feats = []
    print(f"  Loading {len(df)} sequences...")
    with torch.no_grad():
        for i, pid in enumerate(df["uniprot_id"]):
            x = torch.load(f"{feat_dir}/{pid}.pt", map_location="cpu")
            seq_feats.append(x)
            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{len(df)}...")
    return seq_feats

def load_pure_mean_features(df, feat_dir):
    feats = []
    with torch.no_grad():
        for pid in df["uniprot_id"]:
            x = torch.load(f"{feat_dir}/{pid}.pt", map_location="cpu")
            mean_feat = x.mean(dim=0).numpy()
            feats.append(mean_feat)
    return np.array(feats)

def extract_weighted_features(model, seq_data, device):
    features = []
    all_weights = []
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(seq_data):
            x = x.to(device)
            feat, weights = model.pooling(x)
            features.append(feat.cpu().numpy())
            all_weights.append(weights.cpu().numpy())
            if (i + 1) % 500 == 0:
                print(f"    Extracted {i+1}/{len(seq_data)}...")
    return np.array(features), all_weights

def evaluate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

