import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure stdout uses UTF-8 if supported or safe ASCII
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import prepare_data
from models.mlp import SimpleMLP
from clients.honest import train_honest_client
from defenses.aggregators import fedavg_aggregate
from defenses.metrics import evaluate_model

def run_phase1_baseline(seed=42, n_clients=5, rounds=30, batch_size=32, local_epochs=5, lr=0.01, log_path="logs/baseline_fedavg.csv"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    print(f"=== Running Phase 1 Baseline FedAvg (Seed: {seed}) ===")
    print(f"1. Threading seed={seed} into data loading...")
    prepare_data(seed=seed, n_clients=n_clients, alpha=0.5, ref_size=250, save_dir="data")
    
    # Load dataset partitions
    client_data_list = []
    for i in range(n_clients):
        c_data = torch.load(f"data/client_{i}.pt")
        client_data_list.append(c_data)
        
    test_data = torch.load("data/test.pt")
    test_dataset = TensorDataset(test_data['x'], test_data['y'], test_data['sensitive'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_dim = test_data['x'].shape[1]
    
    print(f"2. Threading seed={seed} into model weight initialization...")
    torch.manual_seed(seed)
    global_model = SimpleMLP(input_dim)
    global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}
    
    print(f"3. Threading seed={seed} into training loops...")
    
    logs = []
    for r in range(1, rounds + 1):
        client_updates = []
        client_sizes = []
        
        for i in range(n_clients):
            # Thread seed per client round
            client_seed = seed + r * 100 + i
            update, num_samples = train_honest_client(
                SimpleMLP, input_dim, global_weights, client_data_list[i],
                batch_size=batch_size, local_epochs=local_epochs, lr=lr, seed=client_seed
            )
            client_updates.append(update)
            client_sizes.append(num_samples)
            
        global_weights, avg_update = fedavg_aggregate(global_weights, client_updates, client_sizes)
        global_model.load_state_dict(global_weights)
        
        eval_res = evaluate_model(global_model, test_loader)
        eval_res['round'] = r
        eval_res['seed'] = seed
        logs.append(eval_res)
        
        if r % 5 == 0 or r == 1 or r == rounds:
            print(f"Round {r:02d}/30 | Acc: {eval_res['accuracy']:.4f} | dDP: {eval_res['dp_gap']:.4f} | dEO: {eval_res['eo_gap']:.4f}")
        
    df_logs = pd.DataFrame(logs)
    df_logs.to_csv(log_path, index=False)
    print(f"Phase 1 baseline completed! Saved logs to {log_path}")
    return eval_res['accuracy']

if __name__ == "__main__":
    run_phase1_baseline(seed=42)
