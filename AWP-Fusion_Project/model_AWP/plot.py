import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from model import BalancedE2ERegressionNet
from utils import load_sequence_features


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

def plot_train_curve(task):
    csv_path = "epoch.csv" 
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping training curve plot.")
        return

    df = pd.read_csv(csv_path)
    T = df[f'{task}_T'].tolist()
    R2 = df[f'{task}_R2'].tolist()
    epoch = df[f'{task}_epoch'].tolist()
    
    sorted_idx = np.argsort(T)
    T_sorted = np.array(T)[sorted_idx]
    R2_sorted = np.array(R2)[sorted_idx]
    epoch_sorted = np.array(epoch)[sorted_idx]
    
    best_idx = np.argmax(R2)
    best_T, best_R2, best_epoch = T[best_idx], R2[best_idx], epoch[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_sorted, R2_sorted, color='#2E86AB', linewidth=2.5, marker='o', markersize=8)
    ax.scatter(best_T, best_R2, color='red', s=150, zorder=10, marker='*', edgecolors='white')
    
    ax.annotate(f'T={best_T:.2f}\nR²={best_R2:.4f}\nEpoch={best_epoch}', 
                xy=(best_T, best_R2), xytext=(best_T + 0.02, best_R2 + 0.01), 
                fontsize=9, color='red', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Temperature (T)'); ax.set_ylabel('Max Validation R²')
    ax.set_title(f'Training Process: {task}')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    save_path = f"../results/{task}/figures/training_curve.svg"
    ensure_dir(os.path.dirname(save_path))
    save_figures(fig, save_path)

def plot_weight_analysis(task, best_T):
    weight_path = f"../data/weights/{task}_T{best_T}.pt"
    data_path = f"../data/raw/train_{task}.csv"
    feat_path = f"../data/processed/{task}_sequence_train_feat"
    
    if not os.path.exists(weight_path):
        print(f"Warning: Weight file {weight_path} not found. Skipping weight plot.")
        return

    train_df = pd.read_csv(data_path)
    sequences = train_df['sequence'].values
    seq_data = load_sequence_features(train_df, feat_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    best_T = checkpoint.get('best_T', best_T)  
    
    model = BalancedE2ERegressionNet(temperature=best_T).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    

    all_weights = []
    with torch.no_grad():
        for x in seq_data:
            _, weights = model.pooling(x.to(device))
            all_weights.append(weights.cpu().numpy())
    weights_flat = np.concatenate([w for w in all_weights])
    uniform_weight = 1.0 / np.mean([len(w) for w in all_weights])
    
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    sns.kdeplot(weights_flat, fill=True, alpha=0.6, color='#2E86AB', linewidth=2.5, ax=ax1)
    ax1.axvline(x=uniform_weight, color='#D72638', linestyle='--', linewidth=2, label=f'Uniform = {uniform_weight:.5f}')
    ax1.set_xlabel('Residue Weight'); ax1.set_ylabel('Probability Density')
    ax1.legend()
    save_path1 = f"../results/{task}/figures/weight_density.svg"
    ensure_dir(os.path.dirname(save_path1))
    save_figures(fig1, save_path1)
    
    aa_weights = {aa: [] for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    for seq, ws in zip(sequences, all_weights):
        for aa, w in zip(seq[:len(ws)], ws):
            if aa in aa_weights:
                aa_weights[aa].append(w)
    
    aa_list = list(aa_weights.keys())
    aa_means = [np.mean(aa_weights[aa]) for aa in aa_list]
    aa_stds = [np.std(aa_weights[aa]) for aa in aa_list]
    
    sorted_idx = np.argsort(aa_means)[::-1]
    aa_sorted = [aa_list[i] for i in sorted_idx]
    means_sorted = [aa_means[i] for i in sorted_idx]
    stds_sorted = [aa_stds[i] for i in sorted_idx]
    
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.bar(aa_sorted, means_sorted, yerr=stds_sorted, color='skyblue', edgecolor='black', capsize=3)
    ax2.axhline(y=uniform_weight, color='red', linestyle='--')
    ax2.set_xlabel('Amino Acid'); ax2.set_ylabel('Average Residue Weight')
    
    save_path2 = f"../results/{task}/figures/20aa_weights.svg"
    save_figures(fig2, save_path2)

def main():
    parser = argparse.ArgumentParser(description="Plotting utility for AWP model.")
    parser.add_argument('--task', type=str, required=True, choices=['pH', 'Tm', 'Topt'], help="Task name.")
    parser.add_argument('--type', type=str, choices=['train', 'weight', 'all'], default='all', help="Type of figure to plot.")
    parser.add_argument('--best_T', type=float, required=True, help="Optimal temperature T (e.g., 0.5, 1.0)")
    args = parser.parse_args()
    
    if args.task == 'pH':
        print(f"Plotting all figures for pH task...")
        plot_train_curve(args.task)
        plot_weight_analysis(args.task, args.best_T)
        print("All figures for pH task plotted successfully.")
    else:
        print(f"Plotting training curve only for {args.task} task...")
        plot_train_curve(args.task)
        print(f"Training curve for {args.task} task plotted successfully.")
        
if __name__ == "__main__":
    main()