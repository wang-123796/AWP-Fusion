import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import optimize_fusion_weight, weighted_average
from model_AWP.utils import load_sequence_features, extract_weighted_features, load_pure_mean_features
from model_AWP.model import BalancedE2ERegressionNet


def train_knn(X_train, y_train, X_val, y_val, k_values=[1, 3, 5, 7, 10, 15, 20, 25, 30]):
    """Train KNN with k tuned on validation set."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    best_k = 5
    best_r2 = -np.inf
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
        knn.fit(X_train_scaled, y_train)
        r2 = r2_score(y_val, knn.predict(X_val_scaled))
        if r2 > best_r2:
            best_r2 = r2
            best_k = k

    knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance', n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    return knn, best_k, scaler


def run_intermediate_fusion(task, best_T, val_size=0.15):
    print(f"\n{'='*70}")
    print(f"Intermediate vs Late Fusion | task={task}, T={best_T}")
    print(f"{'='*70}")

    target_col_map = {'pH': 'pHopt', 'Tm': 'tm', 'Topt': 'topt'}
    target_col = target_col_map[task]

    data_dir = "../data/raw"
    feat_dir = "../data/processed"
    weight_dir = "../data/weights"
    result_dir = "../results"

    os.makedirs(f"{result_dir}/{task}/ablation", exist_ok=True)

    train_path = f"{data_dir}/train_{task}.csv"
    test_path = f"{data_dir}/test_{task}.csv"
    train_feat = f"{feat_dir}/{task}_sequence_train_feat"
    test_feat = f"{feat_dir}/{task}_sequence_test_feat"
    weight_path = f"{weight_dir}/{task}_T{best_T}.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_df_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col].values

    print("Loading sequence features...")
    X_train_seq = load_sequence_features(train_df_full, train_feat)
    X_test_seq = load_sequence_features(test_df, test_feat)

    print("Loading mean pooled features...")
    X_train_mean = load_pure_mean_features(train_df_full, train_feat)
    X_test_mean = load_pure_mean_features(test_df, test_feat)

    print(f"Loading model from: {weight_path}...")
    if not os.path.exists(weight_path):
        print(f"Error: weight file not found: {weight_path}")
        return

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    best_model_state = checkpoint['model_state_dict']
    best_T = checkpoint.get('best_T', best_T)

    model = BalancedE2ERegressionNet(
        input_dim=1280, pool_hidden=32, head_hidden=128, dropout=0.6, temperature=best_T
    ).to(device)
    model.load_state_dict(best_model_state)
    model.eval()

    print("Extracting weighted features...")
    X_train_weighted, _ = extract_weighted_features(model, X_train_seq, device)
    X_test_weighted, _ = extract_weighted_features(model, X_test_seq, device)

    
    X_train_concat = np.concatenate([X_train_mean, X_train_weighted], axis=1)
    X_test_concat = np.concatenate([X_test_mean, X_test_weighted], axis=1)
    print(f"Concatenated feature dim: {X_train_concat.shape[1]} (mean: {X_train_mean.shape[1]} + weighted: {X_train_weighted.shape[1]})")

    seeds = [42, 123, 456, 789, 1011]

    results = {
        'late': {'r2': [], 'rmse': [], 'mae': [], 'w': []},
        'intermediate': {'r2': [], 'rmse': [], 'mae': []},
    }

    print(f"\nRunning {len(seeds)} seeds...")
    for i, seed in enumerate(seeds, 1):
        print(f"\n--- Seed {seed} ({i}/{len(seeds)}) ---")
        np.random.seed(seed)

        train_df, val_df = train_test_split(train_df_full, test_size=val_size, random_state=seed)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values

        train_idx, val_idx = train_df.index, val_df.index

        
        X_tr_w = X_train_weighted[train_idx]
        X_val_w = X_train_weighted[val_idx]
        knn_w, k_w, scaler_w = train_knn(X_tr_w, y_train, X_val_w, y_val)
        pred_w_val = knn_w.predict(scaler_w.transform(X_val_w))
        pred_w_test = knn_w.predict(scaler_w.transform(X_test_weighted))

        
        X_tr_m = X_train_mean[train_idx]
        X_val_m = X_train_mean[val_idx]
        knn_m, k_m, scaler_m = train_knn(X_tr_m, y_train, X_val_m, y_val)
        pred_m_val = knn_m.predict(scaler_m.transform(X_val_m))
        pred_m_test = knn_m.predict(scaler_m.transform(X_test_mean))

        
        w_opt = optimize_fusion_weight(pred_m_val, pred_w_val, y_val)
        pred_late = weighted_average(pred_m_test, pred_w_test, w_opt)

        # ==========  (Intermediate Fusion) ==========
        
        X_tr_concat = X_train_concat[train_idx]
        X_val_concat = X_train_concat[val_idx]
        knn_c, k_c, scaler_c = train_knn(X_tr_concat, y_train, X_val_concat, y_val)
        pred_intermediate = knn_c.predict(scaler_c.transform(X_test_concat))

        
        for name, pred in [('late', pred_late), ('intermediate', pred_intermediate)]:
            r2 = r2_score(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            results[name]['r2'].append(r2)
            results[name]['rmse'].append(rmse)
            results[name]['mae'].append(mae)

        results['late']['w'].append(w_opt)

        print(f"  Late fusion:         R2={results['late']['r2'][-1]:.4f} (w={w_opt:.4f})")
        print(f"  Intermediate fusion: R2={results['intermediate']['r2'][-1]:.4f}")

    
    print(f"\n{'='*70}")
    print(f"Summary | task={task}")
    print(f"{'='*70}")
    print(f"\n{'Method':<25} {'R²':<20} {'RMSE':<20} {'MAE':<20}")
    print("-" * 85)
    for name in ['late', 'intermediate']:
        r2_mean = np.mean(results[name]['r2'])
        r2_std = np.std(results[name]['r2'])
        rmse_mean = np.mean(results[name]['rmse'])
        mae_mean = np.mean(results[name]['mae'])
        label = 'Late Fusion' if name == 'late' else 'Intermediate Fusion'
        print(f"{label:<25} {r2_mean:.4f}±{r2_std:.4f}     {rmse_mean:.4f}            {mae_mean:.4f}")

    
    result_path = f"{result_dir}/{task}/ablation/intermediate_vs_late_fusion.csv"
    pd.DataFrame({
        'seed': seeds,
        'late_w': results['late']['w'],
        'late_r2': results['late']['r2'],
        'late_rmse': results['late']['rmse'],
        'late_mae': results['late']['mae'],
        'intermediate_r2': results['intermediate']['r2'],
        'intermediate_rmse': results['intermediate']['rmse'],
        'intermediate_mae': results['intermediate']['mae'],
    }).to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intermediate vs Late Fusion comparison")
    parser.add_argument('--task', type=str, choices=['pH', 'Tm', 'Topt'], default='pH')
    parser.add_argument('--best_T', type=float, default=1.0, help="Optimal temperature T")
    args = parser.parse_args()
    run_intermediate_fusion(args.task, args.best_T)