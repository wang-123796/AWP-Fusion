import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from model import BalancedE2ERegressionNet
from utils import set_seed, load_sequence_features, extract_weighted_features, evaluate_metrics


def train_knn_with_validation(X_train, y_train, seed, val_size=0.15):
    """Split data by seed, tune K on validation set, and train final KNN."""
    np.random.seed(seed)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=seed)

    best_k = 5
    best_r2 = -np.inf
    k_values = [1, 3, 5, 7, 10, 15, 20, 25, 30]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    for k in k_values:
        try:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
            knn.fit(X_tr_scaled, y_tr)
            r2 = r2_score(y_val, knn.predict(X_val_scaled))
            if r2 > best_r2:
                best_r2 = r2
                best_k = k
        except Exception as e:
            continue

    knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance', n_jobs=-1)
    knn.fit(X_tr_scaled, y_tr)
    return knn, best_k, scaler


def train_weighted_model(X_train_seq, y_train, X_val_seq, y_val, device, temperature_list, weight_path):
    """Train weighted pooling model with multiple temperatures."""
    print("\nTraining weighted pooling model...")
    best_global_r2 = -float('inf')
    best_model_state = None
    best_T = None

    for T in temperature_list:
        print(f"\nTraining Temperature T = {T}...")
        model = BalancedE2ERegressionNet(
            input_dim=1280, pool_hidden=32, head_hidden=128, dropout=0.6, temperature=T
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
        best_val_r2 = -float('inf')
        patience = 0
        for epoch in range(500):
            model.train()
            train_loss = 0.0
            for x, y in zip(X_train_seq, y_train):
                pred, _, _ = model(x.to(device))
                loss = nn.MSELoss()(pred, torch.tensor([y], dtype=torch.float32).to(device))
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
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model_state = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= 25:
                    break
        print(f"  Best Val R2 for T={T}: {best_val_r2:.4f}")
        if best_val_r2 > best_global_r2:
            best_global_r2 = best_val_r2
            best_T = T
    print(f"\n>>> Best temperature: T = {best_T} (Val R2 = {best_global_r2:.4f})")
    torch.save({'model_state_dict': best_model_state, 'best_T': best_T, 'best_val_r2': best_global_r2}, weight_path)
    print(f"Best model saved to: {weight_path}")
    return best_model_state, best_T


def run_awp_pipeline(task, best_T):
    print(f"\n{'='*60}")
    print(f"Starting AWP + KNN pipeline for Task: {task}")
    print(f"{'='*60}")

    target_col_map = {'pH': 'pHopt', 'Tm': 'tm', 'Topt': 'topt'}
    target_col = target_col_map[task]

    data_dir = "../data/raw"
    feat_dir = "../data/processed"
    weight_dir = "../data/weights"
    result_dir = "../results"

    os.makedirs(f"{result_dir}/{task}/predictions", exist_ok=True)

    train_path = f"{data_dir}/train_{task}.csv"
    test_path = f"{data_dir}/test_{task}.csv"
    train_feat = f"{feat_dir}/{task}_sequence_train_feat"
    test_feat = f"{feat_dir}/{task}_sequence_test_feat"

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_df_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col].values
    test_ids = test_df["uniprot_id"].values

    X_train_all = load_sequence_features(train_df_full, train_feat)
    X_test_all = load_sequence_features(test_df, test_feat)

    weight_path = f"{weight_dir}/{task}_T{best_T}.pt"
    if os.path.exists(weight_path):
        print(f"\nFound pre-trained weights: {weight_path}")
        print("Loading model...")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        best_model_state = checkpoint['model_state_dict']
    else:
        print(f"\nNo pre-trained weights found: {weight_path}")
        print("Starting training...")
        train_df, val_df = train_test_split(train_df_full, test_size=0.15, random_state=42)
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        X_train_seq = load_sequence_features(train_df, train_feat)
        X_val_seq = load_sequence_features(val_df, train_feat)
        
        temperature_list = [0.1, 0.2, 0.3, 0.4]
        best_model_state, best_T = train_weighted_model(
            X_train_seq, y_train, X_val_seq, y_val, device, temperature_list, weight_path
        )

    print("\nExtracting weighted features...")
    final_model = BalancedE2ERegressionNet(
        input_dim=1280, pool_hidden=32, head_hidden=128, dropout=0.6, temperature=best_T
    ).to(device)
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    X_train_w, _ = extract_weighted_features(final_model, X_train_all, device)
    X_test_w, _ = extract_weighted_features(final_model, X_test_all, device)

    y_train_all = train_df_full[target_col].values
    X_train_w_np = np.array(X_train_w)  
    y_train_all_np = np.array(y_train_all)

    seeds = [42, 123, 456, 789, 1011]
    all_preds = []
    all_k_values = []
    
    print(f"\nRunning {len(seeds)} random seed experiments...")
    for i, seed in enumerate(seeds, 1):
        print(f"--- KNN Experiment {i}/{len(seeds)}: Seed {seed} ---")
        try:
            knn, best_k, scaler = train_knn_with_validation(
                X_train_w_np, y_train_all_np, seed
            )
            
            X_test_scaled = scaler.transform(X_test_w)
            pred_test = knn.predict(X_test_scaled)
            all_preds.append(pred_test)
            all_k_values.append(best_k)
            
            r2 = r2_score(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))
            mae = mean_absolute_error(y_test, pred_test)
            print(f"  Best k: {best_k}, R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        except Exception as e:
            print(f"  Experiment {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    pred_test_avg = np.mean(all_preds, axis=0)
    r2_avg, rmse_avg, mae_avg = evaluate_metrics(y_test, pred_test_avg)
    
    print(f"\nFinal Test Metrics (AWP + KNN - 5-seed average):")
    print(f"  R2:   {r2_avg:.4f}")
    print(f"  RMSE: {rmse_avg:.4f}")
    print(f"  MAE:  {mae_avg:.4f}")

    result_path = f"{result_dir}/{task}/predictions/AWP_{task}_all_seeds.csv"
    result_dict = {"uniprot_id": test_ids, "true_value": np.round(y_test, 2)}
    
    for i, seed in enumerate(seeds[:len(all_preds)]):
        result_dict[f"pred_seed_{seed}"] = np.round(all_preds[i], 2)
    
    result_dict["pred_mean"] = np.round(pred_test_avg, 2)
    result_dict["pred_std"] = np.round(np.std(all_preds, axis=0), 2)
    
    pd.DataFrame(result_dict).to_csv(result_path, index=False)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AWP weighted pooling and KNN pipeline.")
    parser.add_argument('--task', type=str, choices=['pH', 'Tm', 'Topt'], required=True)
    parser.add_argument('--best_T', type=float, required=True)
    args = parser.parse_args()
    run_awp_pipeline(args.task, args.best_T)