import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 

_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_dir)
from utils import load_balanced_model, KNNExperiment

_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from model_AWP.utils import load_sequence_features, extract_weighted_features, load_pure_mean_features
from model_AWP.model import BalancedE2ERegressionNet


def run_fusion_pipeline(task, best_T):
    print(f"\n{'='*70}")
    print(f"Running AWP-Fusion 5-seed experiment for task: {task}")
    print(f"{'='*70}")

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
    
    weight_path = f"{weight_dir}/{task}_T{best_T}.pt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_df_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col].values
    test_ids = test_df["uniprot_id"].values
    print(f"Train size: {len(train_df_full)}, Test size: {len(test_df)}")

    print("Loading sequence features...")
    X_train_seq = load_sequence_features(train_df_full, train_feat)
    X_test_seq = load_sequence_features(test_df, test_feat)
    
    print("Loading mean pooled features...")
    X_train_mean = load_pure_mean_features(train_df_full, train_feat)
    X_test_mean = load_pure_mean_features(test_df, test_feat)

    print(f"Loading model from: {weight_path}...")
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
    print(f"Weighted feature dim: {X_train_weighted.shape[1]}")

    seeds = [42, 123, 456, 789, 1011]
    all_fusion_preds = []
    all_weights = []
    
    print(f"\nRunning {len(seeds)} random seed experiments...")
    for i, seed in enumerate(seeds, 1):
        print(f"--- Experiment {i}/{len(seeds)}: Seed {seed} ---")
        try:
            exp = KNNExperiment(seed)
            res = exp.run(
                train_df_full, test_df, 
                X_train_seq, X_test_seq, 
                X_train_mean, X_test_mean, 
                X_train_weighted, X_test_weighted, 
                target_col
            )
            all_fusion_preds.append(res['fusion_pred'])
            all_weights.append(res['w'])
            
            print(f"  Experiment {i} completed, Fusion weight w = {res['w']:.4f}")
            print(f"  R2: {r2_score(y_test, res['fusion_pred']):.4f}")
            
        except Exception as e:
            print(f"  Experiment {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_fusion_preds) == 0:
        print("\nError: All experiments failed!")
        return

    result_path = f"{result_dir}/{task}/predictions/AWP-Fusion_{task}_all_seeds.csv"
    seed_cols = [f"seed_{s}" for s in seeds[:len(all_fusion_preds)]]
    
    result_dict = {
        "uniprot_id": test_ids,
        "true_value": np.round(y_test, 2)
    }
    
    for i, seed in enumerate(seeds[:len(all_fusion_preds)]):
        result_dict[f"pred_seed_{seed}"] = np.round(all_fusion_preds[i], 2)
    
    result_dict["pred_mean"] = np.round(np.mean(all_fusion_preds, axis=0), 2)
    result_dict["pred_std"] = np.round(np.std(all_fusion_preds, axis=0), 2)
    
    pd.DataFrame(result_dict).to_csv(result_path, index=False)
    print(f"\n✅ AWP-Fusion results saved to: {result_path}")

    print("\n" + "="*70)
    print("5-Seed Statistics (AWP-Fusion):")
    print("="*70)
    for i in range(len(all_fusion_preds)):
        y_pred = all_fusion_preds[i]
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Seed {seeds[i]}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    print(f"\nMean R2: {np.mean([r2_score(y_test, p) for p in all_fusion_preds]):.4f}")
    print(f"Fusion weight w: {np.mean(all_weights):.4f} ± {np.std(all_weights):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AWP-Fusion 5-seed experiment.")
    parser.add_argument('--task', type=str, choices=['pH', 'Tm', 'Topt'], required=True, help="Task name.")
    parser.add_argument('--best_T', type=float, required=True, help="Optimal temperature T")
    args = parser.parse_args()
    run_fusion_pipeline(args.task, args.best_T)