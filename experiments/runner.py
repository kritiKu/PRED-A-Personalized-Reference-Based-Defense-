import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(4)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import prepare_data
from models.mlp import SimpleMLP
from clients.honest import train_honest_client
from attacks.lora_attack import train_malicious_client
from defenses.aggregators import fedavg_aggregate, krum_aggregate, fltrust_aggregate
from defenses.metrics import evaluate_model

def run_fl_experiment(strategy='fedavg', attack=True, seed=42, rounds=30,
                      batch_size=32, local_epochs=5, lr=0.01, log_path=None,
                      data_dir="data", n_clients=5, attacker_id=3):
    """
    Run a federated learning experiment with configurable client count.

    Args:
        n_clients  : total number of clients
        attacker_id: which client index is the malicious one (default 3)
    """
    if log_path and os.path.dirname(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"\n=======================================================")
    print(f"Running FL Experiment | Strategy: {strategy.upper()} | "
          f"Attack: {attack} | Seed: {seed} | Clients: {n_clients}")
    print(f"=======================================================")

    # 1. Prepare / load data
    prepare_data(seed=seed, n_clients=n_clients, alpha=0.5,
                 ref_size=250, save_dir=data_dir)

    client_data_list = []
    for i in range(n_clients):
        c_data = torch.load(f"{data_dir}/client_{i}.pt", weights_only=False)
        client_data_list.append(c_data)

    test_data  = torch.load(f"{data_dir}/test.pt",      weights_only=False)
    ref_data   = torch.load(f"{data_dir}/reference.pt", weights_only=False)

    test_dataset = TensorDataset(test_data['x'], test_data['y'], test_data['sensitive'])
    test_loader  = DataLoader(test_dataset, batch_size=512, shuffle=False)
    input_dim    = test_data['x'].shape[1]

    # 2. KRUM tolerance: assume 1 attacker, tolerate up to floor(n_clients/5)
    krum_f = max(1, n_clients // 5)

    # 3. Init global model
    torch.manual_seed(seed)
    global_model   = SimpleMLP(input_dim)
    global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

    logs = []
    last_median_norm = 1.0

    for r in range(1, rounds + 1):
        client_updates = [None] * n_clients
        client_sizes   = [None] * n_clients
        honest_norms   = []

        # Train honest clients first (all except attacker)
        for i in range(n_clients):
            if i == attacker_id:
                continue
            c_seed = seed + r * 100 + i
            update, num_samples = train_honest_client(
                SimpleMLP, input_dim, global_weights, client_data_list[i],
                batch_size=batch_size, local_epochs=local_epochs, lr=lr, seed=c_seed
            )
            client_updates[i] = update
            client_sizes[i]   = num_samples
            honest_norms.append(torch.norm(update).item())

        current_median_norm = float(np.median(honest_norms)) if honest_norms else last_median_norm
        last_median_norm    = current_median_norm

        # Train malicious / honest client at attacker_id
        c_seed = seed + r * 100 + attacker_id
        if attack:
            update, num_samples = train_malicious_client(
                SimpleMLP, input_dim, global_weights, client_data_list[attacker_id],
                batch_size=batch_size, local_epochs=local_epochs, lr=lr,
                seed=c_seed, target_norm=current_median_norm
            )
        else:
            update, num_samples = train_honest_client(
                SimpleMLP, input_dim, global_weights, client_data_list[attacker_id],
                batch_size=batch_size, local_epochs=local_epochs, lr=lr, seed=c_seed
            )
        client_updates[attacker_id] = update
        client_sizes[attacker_id]   = num_samples

        # Aggregation
        if strategy == 'fedavg':
            global_weights, avg_update = fedavg_aggregate(
                global_weights, client_updates, client_sizes)
        elif strategy == 'krum':
            global_weights, avg_update = krum_aggregate(
                global_weights, client_updates, client_sizes, f=krum_f)
        elif strategy == 'fltrust':
            ref_seed = seed + r * 1000
            global_weights, avg_update, g0, steps_taken = fltrust_aggregate(
                global_weights, client_updates, client_sizes, ref_data, SimpleMLP,
                input_dim, batch_size=batch_size, target_steps=200, seed=ref_seed
            )
            if r % 10 == 0 or r == 1:
                print(f"[FLTrust Round {r}] steps: {steps_taken}/200")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        global_model.load_state_dict(global_weights)
        eval_res = evaluate_model(global_model, test_loader)
        eval_res.update({'round': r, 'seed': seed, 'strategy': strategy,
                         'attack': attack, 'n_clients': n_clients})
        logs.append(eval_res)

        if r % 10 == 0 or r == 1 or r == rounds:
            print(f"Round {r:02d}/{rounds} | Acc: {eval_res['accuracy']:.4f} | "
                  f"dDP: {eval_res['dp_gap']:.4f} | dEO: {eval_res['eo_gap']:.4f}")

    df_logs = pd.DataFrame(logs)
    if log_path:
        df_logs.to_csv(log_path, index=False)
        print(f"Saved logs to {log_path}")

    return df_logs, logs[-1]
