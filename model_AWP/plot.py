import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings("ignore")

# ==================== Font Settings ====================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==================== Model Definition ====================
class BalancedWeightedMeanPooling(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=32, dropout=0.6, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.dropout(h)
        scores = self.fc2(h).squeeze(-1)
        weights = torch.softmax(scores / self.temperature, dim=0)
        weighted_feat = torch.sum(weights.unsqueeze(-1) * x, dim=0)
        return weighted_feat, weights

class BalancedE2ERegressionNet(nn.Module):
    def __init__(self, input_dim=1280, pool_hidden=32, head_hidden=128, 
                 dropout=0.6, temperature=2.0):
        super().__init__()
        self.pooling = BalancedWeightedMeanPooling(input_dim, pool_hidden, dropout, temperature)
        self.head = nn.Sequential(
            nn.Linear(input_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1)
        )
    
    def forward(self, x):
        feat, weights = self.pooling(x)
        pred = self.head(feat).squeeze(-1)
        return pred, feat, weights

# ==================== Data & Utils ====================
def load_sequence_features(df, feat_dir):
    seq_feats = []
    print(f"  Loading {len(df)} sequences...")
    with torch.no_grad():
        for i, pid in enumerate(df["uniprot_id"]):
            x = torch.load(f"{feat_dir}/{pid}.pt", map_location="cpu")
            seq_feats.append(x)
            if (i + 1) % 500 == 0:
                print(f"    Loaded {i+1}/{len(df)}...")
    return seq_feats

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_figures(fig, base_path):
    ensure_dir(os.path.dirname(base_path))
    fig.savefig(base_path, format='svg', bbox_inches='tight')
    png_path = base_path.replace('.svg', '.png')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {base_path} and {png_path}")
    plt.close(fig)

# ==================== Plotting Functions ====================
def plot_train_curve(task):
    csv_path = "epoch.csv" 
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping training curve plot.")
        return

    df = pd.read_csv(csv_path)
    T = df[f'{task}_T'].tolist()
    R2 = df[f'{task}_R2'].tolist()
    
    sorted_idx = np.argsort(T)
    T_sorted = np.array(T)[sorted_idx]
    R2_sorted = np.array(R2)[sorted_idx]
    
    best_idx = np.argmax(R2)
    best_T, best_R2 = T[best_idx], R2[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_sorted, R2_sorted, color='#2E86AB', linewidth=2.5, marker='o', markersize=8)
    ax.scatter(best_T, best_R2, color='red', s=150, zorder=10, marker='*', edgecolors='white')
    
    ax.annotate(f'T={best_T:.2f}\nR²={best_R2:.4f}', 
                xy=(best_T, best_R2), xytext=(best_T + 0.02, best_R2 + 0.01), 
                fontsize=9, color='red', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Temperature (T)'); ax.set_ylabel('Max Validation R²')
    ax.set_title(f'Training Process: {task}')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    save_path = f"../results/{task}/figures/training_curve.svg"
    save_figures(fig, save_path)


def plot_weight_analysis(task, best_T):
    """Generates the 20-amino acid weight bar chart with significance testing."""
    weight_path = f"../data/weights/{task}_T{best_T}.pt"
    data_path = f"../data/raw/train_{task}.csv"
    feat_path = f"../data/processed/{task}_sequence_train_feat"
    
    if not os.path.exists(weight_path):
        print(f"Warning: Weight file {weight_path} not found. Skipping weight plot.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    best_T = checkpoint.get('best_T', best_T)  
    
    train_df = pd.read_csv(data_path)
    sequences = train_df['sequence'].values
    seq_data = load_sequence_features(train_df, feat_path)
    
    model = BalancedE2ERegressionNet(temperature=best_T).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Extracting weights...")
    all_weights = []
    with torch.no_grad():
        for i, x in enumerate(seq_data):
            _, weights = model.pooling(x.to(device))
            all_weights.append(weights.cpu().numpy())
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(seq_data)} samples...")
                
    uniform_weight = 1.0 / np.mean([len(w) for w in all_weights])
    
    # ==================== 20 AA Weight Bar Chart ====================
    print("Generating 20 AA weight chart...")
    aa_weights = {}
    for seq, ws in zip(sequences, all_weights):
        seq_len = min(len(seq), len(ws))
        for i in range(seq_len):
            aa = seq[i]
            if aa not in aa_weights:
                aa_weights[aa] = []
            aa_weights[aa].append(ws[i])
            
    aa_list = list(aa_weights.keys())
    aa_means = [np.mean(aa_weights[aa]) for aa in aa_list]
    aa_stds = [np.std(aa_weights[aa]) for aa in aa_list]
    
    sorted_idx = np.argsort(aa_means)[::-1]
    aa_sorted = [aa_list[i] for i in sorted_idx]
    means_sorted = [aa_means[i] for i in sorted_idx]
    stds_sorted = [aa_stds[i] for i in sorted_idx]
    
    # Color mapping
    category_colors = {
        'Acidic': '#D72638', 'Basic': '#2E86AB', 'Hydrophobic': '#6A994E', 
        'Polar': '#F4A261', 'Special': '#9B5DE5', 'Other': '#BC4B51'
    }
    def get_category(aa):
        if aa in set('DE'): return 'Acidic'
        elif aa in set('KRH'): return 'Basic'
        elif aa in set('STNQY'): return 'Polar'
        elif aa in set('AVLIMFW'): return 'Hydrophobic'
        elif aa in set('GP'): return 'Special'
        else: return 'Other'
    colors = [category_colors[get_category(aa)] for aa in aa_sorted]
    
    # Significance Testing (ANOVA + Tukey HSD)
    groups = []
    group_labels = []
    for aa in aa_sorted:
        groups.extend(aa_weights[aa])
        group_labels.extend([aa] * len(aa_weights[aa]))
    
    tukey = pairwise_tukeyhsd(groups, group_labels, alpha=0.01)
    tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
    
    sig_counts = {aa: 0 for aa in aa_sorted}
    total_comparisons = {aa: 0 for aa in aa_sorted}
    for _, row in tukey_df.iterrows():
        g1, g2, reject = row['group1'], row['group2'], row['reject']
        if reject:
            if np.mean(aa_weights[g1]) > np.mean(aa_weights[g2]):
                sig_counts[g1] += 1
            else:
                sig_counts[g2] += 1
        total_comparisons[g1] += 1
        total_comparisons[g2] += 1
        
    significant_aa = [aa for aa in aa_sorted if total_comparisons[aa] > 0 and (sig_counts[aa] / total_comparisons[aa]) > 0.5 and aa_means[aa_list.index(aa)] > uniform_weight]
    is_significant = [aa in significant_aa for aa in aa_sorted]
    sig_indices = [i for i, sig in enumerate(is_significant) if sig]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(aa_sorted, means_sorted, yerr=stds_sorted, color=colors, 
           edgecolor='black', linewidth=0.8, alpha=0.85, capsize=3)
    
    if sig_indices:
        start = min(sig_indices)
        end = max(sig_indices) + 1
        ax.axvspan(start - 0.5, end - 0.5, alpha=0.15, color='#F46036', zorder=0)
        first_sig_idx = sig_indices[0]
        ax.text(first_sig_idx, means_sorted[first_sig_idx] + stds_sorted[first_sig_idx] + 0.0003, 
                'p < 0.01', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#D72638')
                 
    ax.axhline(y=uniform_weight, color='#D72638', linestyle='--', linewidth=2,
               label=f'Uniform Pooling = {uniform_weight:.5f}')
    ax.set_xlabel('Amino Acid', fontsize=12)
    ax.set_ylabel('Average Residue Weight', fontsize=12)
    ax.set_title(f'{task}: Average Weight for 20 Amino Acids (T={best_T})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    
    save_path = f"../results/{task}/figures/20aa_weights.svg"
    save_figures(fig, save_path)
    
    print(f"\n>> Significant amino acids for {task}: {', '.join(significant_aa) if significant_aa else 'None'}")


def main():
    parser = argparse.ArgumentParser(description="Plotting utility for AWP model.")
    parser.add_argument('--task', type=str, required=True, choices=['pH', 'Tm', 'Topt'], help="Task name.")
    parser.add_argument('--type', type=str, choices=['train', 'weight', 'all'], default='all', help="Type of figure to plot.")
    parser.add_argument('--best_T', type=float, required=True, help="Optimal temperature T")
    args = parser.parse_args()
    
    print(f"Plotting for task: {args.task}, type: {args.type}...")
    
    if args.type in ['train', 'all']:
        plot_train_curve(args.task)
        
    if args.type in ['weight', 'all']:
        plot_weight_analysis(args.task, args.best_T)
        
    print(f"Plotting for {args.task} completed.")

if __name__ == "__main__":
    main()