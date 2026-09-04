import math
import numpy as np
import pandas as pd
import argparse
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from functions import *
from model import MultiAttModel
import os
import warnings
import random

def set_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_local_feat(uid, is_train=True, task="pH", feat_dir="../data/processed"):
    if is_train:
        path = f"{feat_dir}/{task}_sequence_train_feat/{uid}.pt"
    else:
        path = f"{feat_dir}/{task}_sequence_test_feat/{uid}.pt"
    
    feat = torch.load(path, map_location="cpu").float()  
    feat = feat.transpose(0, 1)    
    feat = feat.unsqueeze(0)       
    return feat

def pad_batch(feat_list):
    seq_lens = [f.shape[-1] for f in feat_list]
    max_L = max(seq_lens)
    B = len(feat_list)
    C = 1280

    batch_feat = torch.zeros(B, C, max_L)
    mask = torch.ones(B, max_L, dtype=torch.bool)

    for i in range(B):
        cur_len = seq_lens[i]
        batch_feat[i, :, :cur_len] = feat_list[i]
        mask[i, cur_len:] = False

    return batch_feat, mask

def split_stratified(df, val_ratio=0.1, seed=None, target_col="pHopt"):
    if seed is not None:
        np.random.seed(seed)
    
    df = df.reset_index(drop=True)
    df['group'] = df[target_col].apply(lambda x: 1 if 40 <= x <= 60 else 0)
    tr_idx, val_idx = [], []
    for g in [0, 1]:
        idx = df[df['group'] == g].index.to_numpy()
        np.random.shuffle(idx)
        val_size = int(len(idx) * val_ratio)
        val_idx.extend(idx[:val_size])
        tr_idx.extend(idx[val_size:])
    return df.iloc[tr_idx], df.iloc[val_idx]

def train_eval(model, train_df, val_df, device, lr, batch_size, lr_decay, decay_interval, num_epochs, win_size, task="pH", feat_dir="../data/processed"):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=lr_decay)
    criterion = F.mse_loss

    best_r2 = -float('inf')
    best_state = None

    min_size = 4
    accumulation_steps = batch_size // min_size

    for epoch in range(num_epochs):
        model.train()
        predictions, targets = [], []
        shuffled_df = train_df.sample(frac=1).reset_index(drop=True)
        optimizer.zero_grad()
        
        total_batches = math.ceil(len(shuffled_df) / min_size)

        for i in range(total_batches):
            batch = shuffled_df.iloc[i * min_size : (i + 1) * min_size]

            feats = [load_local_feat(uid, True, task, feat_dir) for uid in batch['uniprot_id']]
            feats, mask = pad_batch(feats)
            feats = feats.to(device)
            mask = mask.to(device)

            y = torch.FloatTensor(batch['target'].values).to(device)
            
            pred = model(feats, mask).squeeze(-1)
            
            loss = criterion(pred.float(), y.float())
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches:
                optimizer.step()
                optimizer.zero_grad()

            predictions.extend(pred.detach().cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())

        model.eval()
        va_pred, va_true = [], []
        
        val_feats_cache = {}
        for uid in val_df['uniprot_id']:
            val_feats_cache[uid] = load_local_feat(uid, True, task, feat_dir)
        
        with torch.no_grad():
            for i in range(math.ceil(len(val_df) / batch_size)):
                batch = val_df.iloc[i * batch_size : (i + 1) * batch_size]
                feats = [val_feats_cache[uid] for uid in batch['uniprot_id']]
                feats, mask = pad_batch(feats)
                feats = feats.to(device)
                mask = mask.to(device)
                y = torch.FloatTensor(batch['target'].values).to(device)
                
                pred = model(feats, mask).squeeze(-1)
                
                va_pred.extend(pred.cpu().numpy().tolist())
                va_true.extend(y.cpu().numpy().tolist())

        val_r2 = get_r2(np.array(va_true), np.array(va_pred))
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_state = model.state_dict().copy()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Val R2: {val_r2:.6f}")

        scheduler.step()

    model.load_state_dict(best_state)
    return model, best_r2

def test(model, test_df, batch_size, device, MAX, MIN, task="pH", feat_dir="../data/processed"):
    model.eval()
    preds, trues, ids = [], [], []
    
    test_feats_cache = {}
    for uid in test_df['uniprot_id']:
        test_feats_cache[uid] = load_local_feat(uid, False, task, feat_dir)
    
    with torch.no_grad():
        for i in range(math.ceil(len(test_df) / batch_size)):
            batch = test_df.iloc[i * batch_size : (i + 1) * batch_size]
            feats = [test_feats_cache[uid] for uid in batch['uniprot_id']]
            feats, mask = pad_batch(feats)
            feats = feats.to(device)
            mask = mask.to(device)
            
            p = model(feats, mask).squeeze(-1).cpu().numpy()
            
            preds.extend(p.tolist() if hasattr(p, 'tolist') else p)
            trues.extend(batch['target'].values)
            ids.extend(batch['uniprot_id'].values)

    trues = np.array(trues)
    preds = np.array(preds)
    
    trues_orig = trues * (MAX - MIN) + MIN
    preds_orig = preds * (MAX - MIN) + MIN

    r2 = get_r2(trues_orig, preds_orig)
    rmse = get_rmse(trues_orig, preds_orig)
    mae = get_mae(trues_orig, preds_orig)

    df = pd.DataFrame({"id": ids, "true": trues_orig, "pred": preds_orig})
    
    return df, r2, rmse, mae

def run_single_experiment(train_df, test_df, win, device, args, seed, MAX, MIN, task="pH", feat_dir="../data/processed"):
    set_random_seeds(seed)
    
    target_col = args.target_col
    train_split, val_split = split_stratified(train_df, 0.1, seed=seed, target_col=target_col)
    
    model = MultiAttModel(1280, win, 4, 4).to(device)
    
    model, best_val_r2 = train_eval(model, train_split, val_split, device,
                                    args.lr, args.batch, args.lr_decay,
                                    args.decay_interval, args.num_epoch, win,
                                    task=task, feat_dir=feat_dir)
    
    df, r2, rmse, mae = test(model, test_df, args.batch, device, MAX, MIN, task=task, feat_dir=feat_dir)
    
    os.makedirs(f'../results/{task}/predictions', exist_ok=True)
    df.to_csv(f'../results/{task}/predictionss/Seq2_{task}_win{win}_seed{seed}.csv', index=False)
    
    return {
        'seed': seed,
        'win': win,
        'val_r2': best_val_r2,
        'test_r2': r2,
        'test_rmse': rmse,
        'test_mae': mae
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='pH', choices=['pH', 'Tm', 'Topt'], help="Task name")
    parser.add_argument('--target_col', default='pHopt', help="Target column name (e.g., pHopt, tm, topt)")
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--lr_decay', default=0.5, type=float)
    parser.add_argument('--decay_interval', default=10, type=int)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--seeds', nargs='+', default=[42, 123, 456, 789, 1011], type=int, 
                        help='List of random seeds for multiple runs')
    # Default window size = 3 (optimal value found by the original author)
    parser.add_argument('--windows', nargs='+', default=3, type=int,
                        help='List of window sizes')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MIN, MAX = 0, 120

    data_dir = "../data/raw"
    feat_dir = "../data/processed"
    result_dir = "../results"

    train_df = pd.read_csv(f"{data_dir}/train_{args.task}.csv")
    test_df = pd.read_csv(f"{data_dir}/test_{args.task}.csv")
    train_df['target'] = rescale_targets(train_df[args.target_col], MAX, MIN)
    test_df['target'] = rescale_targets(test_df[args.target_col], MAX, MIN)

    all_results = []

    for win in args.windows:
        print(f"\n{'='*60}")
        print(f"Window Size: {win}")
        print(f"{'='*60}")
        
        win_results = []
        
        for seed in args.seeds:
            print(f"\n--- Seed: {seed} ---")
            
            result = run_single_experiment(train_df, test_df, win, device, args, seed, MAX, MIN, task=args.task, feat_dir=feat_dir)
            win_results.append(result)
            all_results.append(result)
            
            print(f"Val R2: {result['val_r2']:.6f}")
            print(f"Test - R2: {result['test_r2']:.6f}, RMSE: {result['test_rmse']:.6f}, MAE: {result['test_mae']:.6f}")
        
        if win_results:
            print(f"\n{'='*60}")
            print(f"Summary for Window {win} (over {len(args.seeds)} seeds)")
            print(f"{'='*60}")
            
            metrics = [
                ('val_r2', 'Val R2'),
                ('test_r2', 'Test R2'),
                ('test_rmse', 'Test RMSE'),
                ('test_mae', 'Test MAE')
            ]
            
            for key, name in metrics:
                values = [r[key] for r in win_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{name}: {mean_val:.6f} ± {std_val:.6f}")
            
            summary_df = pd.DataFrame(win_results)
            os.makedirs(f'{result_dir}/{args.task}/summary', exist_ok=True)
            summary_df.to_csv(f'{result_dir}/{args.task}/summary/Seq2_{args.task}_window_{win}_summary.csv', index=False)

    all_results_df = pd.DataFrame(all_results)
    os.makedirs(f'{result_dir}/{args.task}/summary', exist_ok=True)
    all_results_df.to_csv(f'{result_dir}/{args.task}/summary/Seq2_{args.task}_all_results.csv', index=False)
    
    print("\nFinal Results (Mean ± Std over 5 seeds):")
    for win in args.windows:
        win_df = all_results_df[all_results_df['win'] == win]
        print(f"\nWindow {win}:")
        print(f"  Val R2:   {win_df['val_r2'].mean():.6f} ± {win_df['val_r2'].std():.6f}")
        print(f"  Test R2:  {win_df['test_r2'].mean():.6f} ± {win_df['test_r2'].std():.6f}")
        print(f"  Test RMSE:{win_df['test_rmse'].mean():.6f} ± {win_df['test_rmse'].std():.6f}")
        print(f"  Test MAE: {win_df['test_mae'].mean():.6f} ± {win_df['test_mae'].std():.6f}")
    
    best_win = all_results_df.groupby('win')['test_r2'].mean().idxmax()
    best_r2 = all_results_df.groupby('win')['test_r2'].mean().max()
    print(f"\n🏆 Best window: {best_win} (Test R2 = {best_r2:.6f})")