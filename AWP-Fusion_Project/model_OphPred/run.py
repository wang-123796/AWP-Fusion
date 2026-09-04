import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils import KNNBaselineExperiment

def run_oph_pred_pipeline(task):
    print(f"\n{'='*70}")
    print(f"Running OphPred Baseline (Mean Pooling + KNN) for task: {task}")
    print(f"{'='*70}")

    target_col_map = {'pH': 'pHopt', 'Tm': 'tm', 'Topt': 'topt'}
    target_col = target_col_map[task]

    data_dir = "../data/raw"
    feat_dir = "../data/processed"
    result_dir = "../results"

    os.makedirs(f"{result_dir}/{task}/predictions", exist_ok=True)

    train_path = f"{data_dir}/train_{task}.csv"
    test_path = f"{data_dir}/test_{task}.csv"
    train_feat = f"{feat_dir}/{task}_sequence_train_feat"
    test_feat = f"{feat_dir}/{task}_sequence_test_feat"

    train_df_full = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_test = test_df[target_col].values
    test_ids = test_df["uniprot_id"].values
    
    print(f"Train size: {len(train_df_full)}, Test size: {len(test_df)}")
    print(f"Target column: '{target_col}'")

    seeds = [42, 123, 456, 789, 1011]
    all_preds = []
    all_k_values = []
    all_r2_values = []
    all_rmse_values = []
    all_mae_values = []
    
    print(f"\nRunning {len(seeds)} random seed experiments...")
    print("="*70)
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n--- Experiment {i}/{len(seeds)}: Seed {seed} ---")
        try:
            exp = KNNBaselineExperiment(seed=seed)
            result = exp.run(train_df_full, test_df, train_feat, test_feat, target_col)
            
            pred_test = result['pred_test']
            best_k = result['best_k']
            
            r2 = r2_score(y_test, pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred_test))
            mae = mean_absolute_error(y_test, pred_test)
            
            all_preds.append(pred_test)
            all_k_values.append(best_k)
            all_r2_values.append(r2)
            all_rmse_values.append(rmse)
            all_mae_values.append(mae)
            
            print(f"  Best K: {best_k}")
            print(f"  R2:   {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            
        except Exception as e:
            print(f"  Experiment {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_preds) == 0:
        print("\nError: All experiments failed!")
        return
    
    print(f"\nSuccessfully completed {len(all_preds)}/{len(seeds)} experiments")

    pred_test_avg = np.mean(all_preds, axis=0)
    pred_test_std = np.std(all_preds, axis=0)
    
    r2_mean = np.mean(all_r2_values)
    r2_std = np.std(all_r2_values)
    rmse_mean = np.mean(all_rmse_values)
    rmse_std = np.std(all_rmse_values)
    mae_mean = np.mean(all_mae_values)
    mae_std = np.std(all_mae_values)
    
    print("\n" + "="*70)
    print(f"5-Seed Statistics (Mean Pooling + KNN) - Task: {task}")
    print("="*70)
    
    print(f"\n{'Metric':<15} {'Mean ± Std':<30}")
    print("-"*45)
    print(f"{'R²':<15} {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"{'RMSE':<15} {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"{'MAE':<15} {mae_mean:.4f} ± {mae_std:.4f}")
    
    print(f"\nDetailed Results:")
    print(f"  Seeds:       {', '.join([str(s) for s in seeds[:len(all_preds)]])}")
    print(f"  Best K:      {', '.join([str(k) for k in all_k_values])}")
    print(f"  R2:          {', '.join([f'{x:.4f}' for x in all_r2_values])}")
    print(f"  RMSE:        {', '.join([f'{x:.4f}' for x in all_rmse_values])}")
    print(f"  MAE:         {', '.join([f'{x:.4f}' for x in all_mae_values])}")

    save_path = f"{result_dir}/{task}/predictions/OphPred_{task}_all_seeds.csv"
    
    result_dict = {
        "uniprot_id": test_ids,
        "true_value": np.round(y_test, 2)
    }
    
    for i, seed in enumerate(seeds[:len(all_preds)]):
        result_dict[f"pred_seed_{seed}"] = np.round(all_preds[i], 2)
    
    result_dict["pred_mean"] = np.round(pred_test_avg, 2)
    result_dict["pred_std"] = np.round(pred_test_std, 2)
    
    pd.DataFrame(result_dict).to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OphPred baseline pipeline with 5 seeds.")
    parser.add_argument('--task', type=str, choices=['pH', 'Tm', 'Topt'], required=True, help="Specify the dataset task.")
    args = parser.parse_args()
    run_oph_pred_pipeline(args.task)