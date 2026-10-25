import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

torch.set_num_threads(1)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import optimize_fusion_weight, weighted_average
from model_AWP.utils import load_sequence_features, extract_weighted_features, load_pure_mean_features
from model_AWP.model import BalancedE2ERegressionNet


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_T(X_train_seq, y_train, X_val_seq, y_val, device, T, batch_size=32):
    """Train AWP module for a SINGLE temperature with gradient accumulation."""
    set_seed(42)
    model = BalancedE2ERegressionNet(
        input_dim=1280, pool_hidden=32, head_hidden=128, dropout=0.6,
        temperature=T, num_layers=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    best_val_r2 = -float('inf')
    best_state = None
    patience = 0

    n_train = len(X_train_seq)
    indices = np.arange(n_train)

    for epoch in range(100):
        model.train()
        np.random.shuffle(indices)
        
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start:start + batch_size]
            optimizer.zero_grad()
            for i in batch_idx:
                x = X_train_seq[i].to(device)
                y = y_train[i]
                pred, _, _ = model(x)
                loss = nn.MSELoss()(pred, torch.tensor(y, dtype=torch.float32).to(device))
                loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for x in X_val_seq:
                pred, _, _ = model(x.to(device))
                val_preds.append(pred.cpu().item())
        val_r2 = r2_score(y_val, np.array(val_preds))

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 25:
                break

    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_state, best_val_r2


def run_cross_validation(task, T, n_splits=5, val_size=0.15):
    print(f"\n{'='*70}")
    print(f"5-Fold CV | task={task} | Testing T={T}")
    print(f"{'='*70}")

    target_col_map = {'pH': 'pHopt', 'Tm': 'tm', 'Topt': 'topt'}
    target_col = target_col_map[task]

    data_dir = "../data/raw"
    feat_dir = "../data/processed"
    weight_dir = "../data/weights"
    result_dir = "../results"

    os.makedirs(f"{result_dir}/{task}/cv_search", exist_ok=True)

    train_path = f"{data_dir}/train_{task}.csv"
    test_path = f"{data_dir}/test_{task}.csv"
    train_feat = f"{feat_dir}/{task}_sequence_train_feat"
    test_feat = f"{feat_dir}/{task}_sequence_test_feat"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col].values

    
    all_folds_exist = True
    for fold in range(1, n_splits + 1):
        weight_path = f"{weight_dir}/{task}_cv_fold{fold}_T{T}.pt"
        if not os.path.exists(weight_path):
            all_folds_exist = False
            break
    
    if all_folds_exist:
        print(f"All folds for T={T} already exist. Skipping.")
        return

    print("Loading test features...")
    X_test_seq = load_sequence_features(test_df, test_feat)
    X_test_mean = load_pure_mean_features(test_df, test_feat)

    
    y_binned = pd.qcut(train_df_full[target_col], q=n_splits, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df_full, y_binned), 1):
        weight_path = f"{weight_dir}/{task}_cv_fold{fold}_T{T}.pt"
        
        fold_train_df = train_df_full.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = train_df_full.iloc[val_idx].reset_index(drop=True)
        y_train = fold_train_df[target_col].values
        y_val = fold_val_df[target_col].values

        
        awp_train_df, awp_val_df = tts(fold_train_df, test_size=val_size, random_state=42)
        y_awp_train = awp_train_df[target_col].values
        y_awp_val = awp_val_df[target_col].values

        if not os.path.exists(weight_path):
            print(f"  [Fold {fold}] Training AWP with T={T}...")
            X_awp_train_seq = load_sequence_features(awp_train_df, train_feat)
            X_awp_val_seq = load_sequence_features(awp_val_df, train_feat)
            
            best_state, best_val_r2 = train_single_T(
                X_awp_train_seq, y_awp_train, X_awp_val_seq, y_awp_val, device, T
            )
            
            torch.save({'model_state_dict': best_state, 'T': T, 'best_val_r2': best_val_r2}, weight_path)
            del X_awp_train_seq, X_awp_val_seq
        else:
            print(f"  [Fold {fold}] Weights for T={T} exist, loading...")
            checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
            best_state = checkpoint['model_state_dict']

        
        print(f"  [Fold {fold}] Extracting features...")
        model = BalancedE2ERegressionNet(
            input_dim=1280, pool_hidden=32, head_hidden=128, dropout=0.6,
            temperature=T, num_layers=2
        ).to(device)
        model.load_state_dict(best_state)
        model.eval()

        
        X_fold_train_seq = load_sequence_features(fold_train_df, train_feat)
        X_fold_val_seq = load_sequence_features(fold_val_df, train_feat)

        X_fold_train_w, _ = extract_weighted_features(model, X_fold_train_seq, device)
        X_fold_val_w, _ = extract_weighted_features(model, X_fold_val_seq, device)
        X_test_w, _ = extract_weighted_features(model, X_test_seq, device)

        X_fold_train_mean = load_pure_mean_features(fold_train_df, train_feat)
        X_fold_val_mean = load_pure_mean_features(fold_val_df, train_feat)

        
        scaler_w = StandardScaler()
        X_tr_w_scaled = scaler_w.fit_transform(X_fold_train_w)
        knn_w = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=1).fit(X_tr_w_scaled, y_train)
        pred_w_test = knn_w.predict(scaler_w.transform(X_test_w))
        pred_w_val = knn_w.predict(scaler_w.transform(X_fold_val_w))

        scaler_m = StandardScaler()
        X_tr_m_scaled = scaler_m.fit_transform(X_fold_train_mean)
        knn_m = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=1).fit(X_tr_m_scaled, y_train)
        pred_m_test = knn_m.predict(scaler_m.transform(X_test_mean))
        pred_m_val = knn_m.predict(scaler_m.transform(X_fold_val_mean))

        
        w_opt = optimize_fusion_weight(pred_m_val, pred_w_val, y_val)
        pred_fusion = weighted_average(pred_m_test, pred_w_test, w_opt)

        
        r2 = r2_score(y_test, pred_fusion)
        rmse = np.sqrt(mean_squared_error(y_test, pred_fusion))
        mae = mean_absolute_error(y_test, pred_fusion)

        print(f"  [Fold {fold}] T={T} | R2={r2:.4f}, RMSE={rmse:.4f}")
        
        fold_results.append({'fold': fold, 'T': T, 'r2': r2, 'rmse': rmse, 'mae': mae})

        
        del model, X_fold_train_w, X_fold_val_w, X_test_w, X_fold_train_seq, X_fold_val_seq
        del X_fold_train_mean, X_fold_val_mean
        del knn_w, knn_m, scaler_w, scaler_m
        torch.cuda.empty_cache()
        gc.collect()

    if fold_results:
        r2_mean = np.mean([r['r2'] for r in fold_results])
        print(f"\n>>> Finished T={T}. Mean R2 across folds: {r2_mean:.4f}")
        
        
        pd.DataFrame(fold_results).to_csv(f"{result_dir}/{task}/cv_search/summary_T{T}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['pH', 'Tm', 'Topt'], default='pH')
    parser.add_argument('--T', type=float, required=True, help="Specify ONE temperature to test")
    args = parser.parse_args()
    run_cross_validation(args.task, args.T)