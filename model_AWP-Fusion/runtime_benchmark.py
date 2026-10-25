import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import functools


_original_torch_load = torch.load
torch.load = functools.partial(_original_torch_load, weights_only=False)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


esm_main_path = os.path.join(_project_root, 'esm_main')
if esm_main_path not in sys.path:
    sys.path.insert(0, esm_main_path)

from esm_main import esm

from model_AWP.utils import load_sequence_features, extract_weighted_features
from model_AWP.model import BalancedE2ERegressionNet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

def run_runtime_benchmark(task, best_T, n_samples=50):
    print(f"\n{'='*70}")
    print(f"Runtime Benchmark | task={task}, T={best_T}, samples={n_samples}")
    print(f"{'='*70}")

    target_col_map = {'pH': 'pHopt', 'Tm': 'tm', 'Topt': 'topt'}
    target_col = target_col_map[task]

    data_dir = "../data/raw"
    feat_dir = "../data/processed"
    weight_dir = "../data/weights"
    result_dir = "../results"
    
    os.makedirs(f"{result_dir}/{task}", exist_ok=True)

    train_path = f"{data_dir}/train_{task}.csv"
    test_path = f"{data_dir}/test_{task}.csv"
    train_feat = f"{feat_dir}/{task}_sequence_train_feat"
    test_feat = f"{feat_dir}/{task}_sequence_test_feat"
    weight_path = f"{weight_dir}/{task}_T{best_T}.pt"
    
    esm_weight_path = f"{weight_dir}/esm2_t33_650M_UR50D.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_df = pd.read_csv(test_path)
    if len(test_df) > n_samples:
        test_df_sub = test_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    else:
        test_df_sub = test_df

    sequences = test_df_sub['sequence'].values

    
    print("\n[1/3] Measuring ESM-2 embedding time (loading local .pt)...")
    
    if not os.path.exists(esm_weight_path):
        print(f"Error: Cannot find ESM weight file at {esm_weight_path}.")
        return

    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_weight_path)
    esm_model = esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    dummy_data = [("seq", sequences[0])]
    _, _, dummy_tokens = batch_converter(dummy_data)
    dummy_tokens = dummy_tokens.to(device)
    with torch.no_grad():
        _ = esm_model(dummy_tokens, repr_layers=[33], return_contacts=False)
    if device.type == 'cuda': torch.cuda.synchronize()

    esm_times = []
    with torch.no_grad():
        for seq in sequences:
            batch_data = [("seq", seq)]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            if device.type == 'cuda': torch.cuda.synchronize()
            start_time = time.time()
            
            _ = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            end_time = time.time()
            esm_times.append(end_time - start_time)
            
    mean_esm_time = np.mean(esm_times)
    print(f"  ESM-2 embedding time: {mean_esm_time*1000:.2f} ms / sequence")
    
    
    del esm_model
    torch.cuda.empty_cache()

    
    print("\n[2/3] Measuring AWP + KNN prediction time...")
    
    X_train_seq = load_sequence_features(pd.read_csv(train_path), train_feat)
    X_test_seq = load_sequence_features(test_df_sub, test_feat)

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    model = BalancedE2ERegressionNet(input_dim=1280, temperature=best_T).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    
    start_time = time.time()
    X_test_w, _ = extract_weighted_features(model, X_test_seq, device)
    if device.type == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    mean_awp_time = (end_time - start_time) / len(sequences)

    
    scaler = StandardScaler()
    X_dummy_train = np.random.randn(100, 1280)  
    y_dummy_train = np.random.randn(100)
    knn = KNeighborsRegressor(n_neighbors=5, n_jobs=1).fit(X_dummy_train, y_dummy_train)
    
    start_time = time.time()
    _ = knn.predict(X_test_w)
    end_time = time.time()
    mean_knn_time = (end_time - start_time) / len(sequences)

    total_downstream_time = mean_awp_time + mean_knn_time
    print(f"  AWP pooling time: {mean_awp_time*1000:.4f} ms / sequence")
    print(f"  KNN inference time: {mean_knn_time*1000:.4f} ms / sequence")
    print(f"  Total downstream time: {total_downstream_time*1000:.4f} ms / sequence")

    
    full_time = mean_esm_time + total_downstream_time
    print("\n[3/3] Summary")
    print(f"  Full pipeline: {full_time*1000:.2f} ms / sequence")
    print(f"  ESM-2 dominates {mean_esm_time/full_time*100:.2f}% of the time.")
    
    result_df = pd.DataFrame({
        'task': [task],
        'n_samples': [len(sequences)],
        'esm2_time_ms': [mean_esm_time * 1000],
        'awp_knn_time_ms': [total_downstream_time * 1000],
        'full_pipeline_ms': [full_time * 1000]
    })
    save_path = f"{result_dir}/{task}/runtime_benchmark.csv"
    result_df.to_csv(save_path, index=False)
    print(f"\nTiming results saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['pH', 'Tm', 'Topt'], default='pH')
    parser.add_argument('--best_T', type=float, default=1.0)
    parser.add_argument('--n_samples', type=int, default=50)
    args = parser.parse_args()
    
    run_runtime_benchmark(args.task, args.best_T, args.n_samples)