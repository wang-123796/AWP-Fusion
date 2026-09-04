import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_figures(fig, base_path):
    fig.savefig(base_path, format='svg', bbox_inches='tight')
    png_path = base_path.replace('.svg', '.png')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {base_path} and {png_path}")
    import matplotlib.pyplot as plt
    plt.close(fig)  

def get_best_seed_for_task(task):
    seed_map = {
        'pH': 1011,
        'Topt': 1011,
        'Tm': 456
    }
    return seed_map.get(task, 1011) 

def plot_fusion_scatter(task):
    pred_path = f"../results/{task}/predictions/AWP-Fusion_{task}_all_seeds.csv"
    test_path = f"../data/raw/test_{task}.csv"
    
    if not os.path.exists(pred_path):
        print(f"Warning: Fusion results not found at {pred_path}. Skipping scatter plot.")
        return

    df_pred = pd.read_csv(pred_path)
    df_test = pd.read_csv(test_path)
    
    seed_cols = [col for col in df_pred.columns if col.startswith('seed_')]
    if not seed_cols:
        print(f"Warning: No seed columns found in {pred_path}. Available columns: {df_pred.columns.tolist()}")
        return
    
    print(f"Found seed columns: {seed_cols}")
    
    selected_seed = get_best_seed_for_task(task)
    selected_col = f"seed_{selected_seed}"
    
    if selected_col not in df_pred.columns:
        print(f"Warning: Selected seed column '{selected_col}' not found. Using first available seed column.")
        selected_col = seed_cols[0]
        selected_seed = int(selected_col.split('_')[1])
    
    print(f"Task: {task}, Selected seed: {selected_seed} (column: {selected_col})")
    
    merged = pd.merge(df_test, df_pred, on="uniprot_id")
    
    y_true = merged['true_value'].values if 'true_value' in merged.columns else None
    if y_true is None:
        target_col_map = {'pH': 'pHopt', 'Tm': 'tm', 'Topt': 'topt'}
        target_col = target_col_map[task]
        y_true = df_test[target_col].values
    
    y_pred = merged[selected_col].values
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    if 'mean_pred' in df_pred.columns:
        y_pred_mean = merged['mean_pred'].values
        r2_mean = r2_score(y_true, y_pred_mean)
        rmse_mean = np.sqrt(mean_squared_error(y_true, y_pred_mean))
        mae_mean = mean_absolute_error(y_true, y_pred_mean)
        print(f"Mean prediction - R2: {r2_mean:.4f}, RMSE: {rmse_mean:.4f}, MAE: {mae_mean:.4f}")
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20, color='#2E86AB', edgecolors='none')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'r--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Experimental Value', fontsize=12)
    ax.set_ylabel('Predicted Value', fontsize=12)
    ax.set_title(f'AWP-Fusion: {task} Prediction (Seed={selected_seed})', fontsize=13)
    
    text_str = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    ax.text(0.03, 0.97, text_str, 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    ax.set_xlim(min_val-margin, max_val+margin)
    ax.set_ylim(min_val-margin, max_val+margin)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    save_path = f"../results/{task}/figures/fusion_scatter_seed{selected_seed}.svg"
    ensure_dir(os.path.dirname(save_path))
    save_figures(fig, save_path)
    
    print(f"✅ Scatter plot saved with seed={selected_seed}")
    
def main():
    parser = argparse.ArgumentParser(description="Plotting utility for AWP-Fusion model.")
    parser.add_argument('--task', type=str, required=True, choices=['pH', 'Tm', 'Topt'], help="Task name.")
    parser.add_argument('--type', type=str, choices=['scatter', 'compare', 'all'], default='all', help="Type of figure to plot.")
    args = parser.parse_args()
    
    if args.type == 'scatter' or args.type == 'all':
        print(f"\nPlotting fusion scatter plot for task {args.task}...")
        plot_fusion_scatter(args.task)

if __name__ == "__main__":
    main()