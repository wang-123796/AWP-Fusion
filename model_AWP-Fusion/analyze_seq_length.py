import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

def analyze_sequence_lengths():
    data_dir = "../data/raw"
    result_dir = "../results"
    os.makedirs(result_dir, exist_ok=True)
    
    tasks = ['pH', 'Topt', 'Tm']
    summary_data = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    for i, task in enumerate(tasks):
        train_path = f"{data_dir}/train_{task}.csv"
        test_path = f"{data_dir}/test_{task}.csv"
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Warning: Data for {task} not found. Skipping.")
            continue
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        
        train_lengths = train_df['sequence'].str.len()
        
        test_lengths = test_df['sequence'].str.len()
        
        
        all_lengths = pd.concat([train_lengths, test_lengths])
        
        summary_data.append({
            'Task': task,
            'Min_Length': all_lengths.min(),
            'Max_Length': all_lengths.max(),
            'Median_Length': int(all_lengths.median()),
            'Mean_Length': round(all_lengths.mean(), 1),
            'Std_Length': round(all_lengths.std(), 1)
        })
        
        
        ax = axes[i]
        sns.histplot(all_lengths, bins=50, kde=True, ax=ax, color='#2E86AB', alpha=0.6)
        ax.set_title(f'{task} Sequence Length Distribution')
        ax.set_xlabel('Sequence Length (Amino Acids)')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        
        stats_text = f"Range: {all_lengths.min()} - {all_lengths.max()}\nMedian: {int(all_lengths.median())}\nMean: {all_lengths.mean():.1f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig_path = f"{result_dir}/sequence_length_distribution.svg"
    plt.savefig(fig_path, format='svg', bbox_inches='tight')
    plt.savefig(fig_path.replace('.svg', '.png'), format='png', dpi=300, bbox_inches='tight')
    print(f"✅ Distribution plot saved to: {fig_path}")

    
    summary_df = pd.DataFrame(summary_data)
    csv_path = f"{result_dir}/sequence_length_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print("Sequence Length Summary")
    print("="*50)
    print(summary_df.to_string(index=False))
    print("="*50)
    print(f"✅ Summary table saved to: {csv_path}")

if __name__ == "__main__":
    analyze_sequence_lengths()