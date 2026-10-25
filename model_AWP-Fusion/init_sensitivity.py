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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


torch.set_num_threads(4)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model_AWP.utils import load_sequence_features, extract_weighted_features
from model_AWP.model import BalancedE2ERegressionNet


def set_init_seed(seed):
    """Set all random seeds for MLP initialization."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_mlp(X_train_seq, y_train, X_val_seq, y_val, device, T, init_seed, weight_path):
    """Train MLP with a fixed data split and a specific init seed."""
    print(f"\nTraining MLP with init_seed={init_seed}, T={T}")
    set_init_seed(init_seed)

    model = BalancedE2ERegressionNet(
        input_dim=1280, pool_hidden=32, head_hidden=128, dropout=0.6,
        temperature=T, num_layers=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    best_val_r2 = -float('inf')
    best_state = None
    patience = 0

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for x, y in zip(X_train_seq, y_train):
            pred, _, _ = model(x.to(device))
            loss = nn.MSELoss()(pred, torch.tensor(y, dtype=torch.float32).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for x in X_val_seq:
                pred, _, _ = model(x.to(device))
                val_preds.append(pred.cpu().item())
        val_r2 = r2_score(y_val, np.array(val_preds))

        
        print(f"  Epoch {epoch:3d} | train_loss={train_loss/len(X_train_seq):.4f} | val_r2={val_r2:.4f} | best={best_val_r2:.4f} | patience={patience}")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 25:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Best Val R2: {best_val_r2:.4f}")
    torch.save({'model_state_dict': best_state, 'T': T, 'init_seed': init_seed,
                'best_val_r2': best_val_r2}, weight_path)

    del model
    del optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return best_state, best_val_r2


def run_single_init(task, T, init_seed, val_size=0.15):
    print(f"\n{'='*70}")
    print(f"Init Sensitivity | task={task}, T={T}, init_seed={init_seed}")
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_df_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col].values

    train_df, val_df = train_test_split(train_df_full, test_size=val_size, random_state=42)
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values

    print(f"Fixed data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    print("Loading sequence features...")
    X_train_seq = load_sequence_features(train_df, train_feat)
    X_val_seq = load_sequence_features(val_df, train_feat)
    X_test_seq = load_sequence_features(test_df, test_feat)

    weight_path = f"{weight_dir}/{task}_T{T}_init{init_seed}.pt"
    if os.path.exists(weight_path):
        print(f"Loading existing weights: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        best_state = checkpoint['model_state_dict']
    else:
        best_state, _ = train_mlp(
            X_train_seq, y_train, X_val_seq, y_val, device, T, init_seed, weight_path
        )

    
    model = BalancedE2ERegressionNet(
        input_dim=1280, pool_hidden=32, head_hidden=128, dropout=0.6,
        temperature=T, num_layers=2
    ).to(device)
    model.load_state_dict(best_state)
    model.eval()

    X_train_w, _ = extract_weighted_features(model, X_train_seq, device)
    X_test_w, _ = extract_weighted_features(model, X_test_seq, device)

    
    np.random.seed(42)
    X_tr, X_val_knn, y_tr, y_val_knn = train_test_split(
        X_train_w, y_train, test_size=val_size, random_state=42
    )

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val_knn)

    best_k = 5
    best_r2 = -np.inf
    for k in [1, 3, 5, 7, 10, 15, 20, 25, 30]:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=4)
        knn.fit(X_tr_scaled, y_tr)
        r2 = r2_score(y_val_knn, knn.predict(X_val_scaled))
        if r2 > best_r2:
            best_r2 = r2
            best_k = k

    knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance', n_jobs=4)
    knn.fit(X_tr_scaled, y_tr)
    pred_test = knn.predict(scaler.transform(X_test_w))

    r2 = r2_score(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    mae = mean_absolute_error(y_test, pred_test)

    print(f"\nResult for init_seed={init_seed}:")
    print(f"  k={best_k}, R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    result_path = f"{result_dir}/{task}/ablation/init_sensitivity_seed{init_seed}.csv"
    pd.DataFrame({
        'init_seed': [init_seed],
        'best_k': [best_k],
        'r2': [r2],
        'rmse': [rmse],
        'mae': [mae],
    }).to_csv(result_path, index=False)
    print(f"Result saved to: {result_path}")

    del model, X_train_w, X_test_w, knn, scaler
    del X_train_seq, X_val_seq, X_test_seq
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP initialization sensitivity (single seed)")
    parser.add_argument('--task', type=str, choices=['pH', 'Tm', 'Topt'], default='pH')
    parser.add_argument('--T', type=float, default=1.0, help="Fixed temperature T")
    parser.add_argument('--init_seed', type=int, required=True, help="MLP initialization seed")
    args = parser.parse_args()
    run_single_init(args.task, args.T, args.init_seed)