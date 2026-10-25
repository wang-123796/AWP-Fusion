import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score

def load_seq2_and_get_true(task, seeds, result_dir):
    seq2_scores = []
    seq2_true_values = None
    seq2_ids = None

    for seed in seeds:
        seq_path = f"{result_dir}/{task}/predictions/Seq2_{task}_win3_seed{seed}.csv"
        if not os.path.exists(seq_path):
            print(f"Warning: Seq2 file not found -> {seq_path}")
            continue
            
        df = pd.read_csv(seq_path)
        
        
        if seq2_true_values is None:
            seq2_true_values = df['true'].values
            seq2_ids = df['id'].values if 'id' in df.columns else df['uniprot_id'].values
            
        r2 = r2_score(df['true'].values, df['pred'].values)
        seq2_scores.append(r2)
        
    return np.array(seq2_scores), seq2_true_values, seq2_ids

def load_other_model_scores(file_path, true_values, true_ids):
    if not os.path.exists(file_path):
        print(f"Warning: File not found -> {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    
    if not np.array_equal(df['uniprot_id'].values, true_ids):
        true_series = pd.Series(true_values, index=true_ids)
        try:
            y_true = true_series.loc[df['uniprot_id']].values
        except KeyError:
            print(f"Error: ID mismatch in {file_path}")
            return None
    else:
        y_true = true_values

    r2_scores = []
    pred_cols = [c for c in df.columns if 'seed_' in c or 'pred_seed' in c]
    pred_cols = sorted(pred_cols, key=lambda x: int(''.join(filter(str.isdigit, x))))

    for col in pred_cols:
        r2_scores.append(r2_score(y_true, df[col].values))
        
    return np.array(r2_scores)

def run_statistical_tests(task):
    print(f"\n{'='*70}")
    print(f"Statistical Tests | task={task}")
    print(f"{'='*70}")

    result_dir = "../results"
    seeds = [42, 123, 456, 789, 1011]


    seq2_scores, true_values, true_ids = load_seq2_and_get_true(task, seeds, result_dir)
    
    if seq2_scores is None or len(seq2_scores) == 0:
        print("Error: Seq2 results not found. Cannot establish ground truth.")
        return
        
    print(f"Loaded Seq2: {len(seq2_scores)} runs, Mean R2 = {np.mean(seq2_scores):.4f}")

    models = {'Seq2': seq2_scores}

    
    base_path = f"{result_dir}/{task}/predictions/AWP-Fusion_{task}_all_seeds.csv"
    base_scores = load_other_model_scores(base_path, true_values, true_ids)
    if base_scores is None:
        print("Error: Cannot find AWP-Fusion results. Aborting.")
        return
    print(f"Loaded AWP-Fusion: {len(base_scores)} runs, Mean R2 = {np.mean(base_scores):.4f}")

    
    awp_path = f"{result_dir}/{task}/predictions/AWP_{task}_all_seeds.csv"
    awp_scores = load_other_model_scores(awp_path, true_values, true_ids)
    if awp_scores is not None:
        models['AWP'] = awp_scores
        print(f"Loaded AWP: {len(awp_scores)} runs, Mean R2 = {np.mean(awp_scores):.4f}")

    
    oph_path = f"{result_dir}/{task}/predictions/OphPred_{task}_all_seeds.csv"
    oph_scores = load_other_model_scores(oph_path, true_values, true_ids)
    if oph_scores is not None:
        models['OphPred'] = oph_scores
        print(f"Loaded OphPred: {len(oph_scores)} runs, Mean R2 = {np.mean(oph_scores):.4f}")

    
    print(f"\n{'-'*70}")
    print(f"Paired t-tests vs AWP-Fusion")
    print(f"{'-'*70}")

    for name, comp_scores in models.items():
        if len(base_scores) != len(comp_scores):
            print(f"Warning: Length mismatch ({len(base_scores)} vs {len(comp_scores)}). Skipping.")
            continue

        t_stat, p_val = ttest_rel(base_scores, comp_scores)
        mean_diff = np.mean(base_scores) - np.mean(comp_scores)

        if p_val < 0.001:
            p_str, sig = "p < 0.001", "***"
        elif p_val < 0.01:
            p_str, sig = f"p = {p_val:.4f}", "**"
        elif p_val < 0.05:
            p_str, sig = f"p = {p_val:.4f}", "*"
        else:
            p_str, sig = f"p = {p_val:.4f}", "ns"

        print(f"AWP-Fusion vs {name:<10} | Mean diff: +{mean_diff:.4f} | t-stat: {t_stat:>7.3f} | {p_str} ({sig})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired t-tests for model comparison")
    parser.add_argument('--task', type=str, choices=['pH', 'Tm', 'Topt'], default='pH')
    args = parser.parse_args()
    run_statistical_tests(args.task)