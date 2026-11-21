import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import prepare_data
from models.mlp import SimpleMLP
from clients.honest import train_honest_client
from attacks.lora_attack import train_malicious_client
from defenses.aggregators import fltrust_aggregate
from defenses.signal import compute_raw_rank_signal
from defenses.metrics import evaluate_model
from torch.utils.data import DataLoader, TensorDataset

SEEDS = [42, 43, 44, 45, 46]

def run_phase8_9_defense():
    print("\n=======================================================")
    print("PHASE 8 & 9: PER-CLIENT BASELINE RELATIVE EWMA DEFENSE")
    print("=======================================================")
    
    all_seed_results = []
    
    for seed in SEEDS:
        print(f"\n---> Running Baseline-Relative EWMA Defense | Seed {seed}")
        prepare_data(seed=seed, n_clients=5, alpha=0.5, ref_size=250, save_dir="data")
        
        client_data_list = [torch.load(f"data/client_{i}.pt") for i in range(5)]
        ref_data = torch.load("data/reference.pt")
        test_data = torch.load("data/test.pt")
        test_loader = DataLoader(TensorDataset(test_data['x'], test_data['y'], test_data['sensitive']), batch_size=64, shuffle=False)
        input_dim = client_data_list[0]['x'].shape[1]
        
        torch.manual_seed(seed)
        global_model = SimpleMLP(input_dim)
        global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}
        
        # Round 0: Calculate server's self-consistency baseline (attacker-free reference baseline)
        ref_seed1 = seed + 10000
        ref_seed2 = seed + 20000
        _, _, g0_run1, _ = fltrust_aggregate(global_weights, [torch.zeros_like(p.view(-1)) for p in global_model.parameters()], [1]*5, ref_data, SimpleMLP, input_dim, seed=ref_seed1)
        _, _, g0_run2, _ = fltrust_aggregate(global_weights, [torch.zeros_like(p.view(-1)) for p in global_model.parameters()], [1]*5, ref_data, SimpleMLP, input_dim, seed=ref_seed2)
        
        base_sig1 = compute_raw_rank_signal(g0_run1, g0_run2, global_weights)
        baseline_signal = base_sig1
        
        ewma = np.zeros(5)
        decay = 0.9
        threshold = 1.2 # Baseline-relative threshold
        
        logs = []
        
        for r in range(1, 31):
            honest_norms = []
            for i in range(5):
                if i == 3:
                    continue
                client_seed = seed + r * 100 + i
                update, num_samples = train_honest_client(
                    SimpleMLP, input_dim, global_weights, client_data_list[i],
                    batch_size=32, local_epochs=5, lr=0.01, seed=client_seed
                )
                honest_norms.append(torch.norm(update).item())
                
            median_honest_norm = float(np.median(honest_norms))
            
            client_updates = [None] * 5
            client_sizes = [None] * 5
            
            for i in range(5):
                client_seed = seed + r * 100 + i
                if i == 3:
                    update, num_samples = train_malicious_client(
                        SimpleMLP, input_dim, global_weights, client_data_list[3],
                        batch_size=32, local_epochs=5, lr=0.01, seed=client_seed, target_norm=median_honest_norm
                    )
                else:
                    update, num_samples = train_honest_client(
                        SimpleMLP, input_dim, global_weights, client_data_list[i],
                        batch_size=32, local_epochs=5, lr=0.01, seed=client_seed
                    )
                client_updates[i] = update
                client_sizes[i] = num_samples
                
            # Reference direction
            ref_seed = seed + r * 1000
            _, _, g0, _ = fltrust_aggregate(
                global_weights, client_updates, client_sizes, ref_data, SimpleMLP,
                input_dim, batch_size=32, target_steps=200, seed=ref_seed
            )
            
            # Compute baseline-relative signal and EWMA downweighting
            weights = np.zeros(5)
            for i in range(5):
                raw_sig = compute_raw_rank_signal(client_updates[i], g0, global_weights)
                rel_sig = raw_sig / (baseline_signal + 1e-8)
                is_above = 1.0 if rel_sig > threshold else 0.0
                
                ewma[i] = decay * ewma[i] + (1.0 - decay) * is_above
                weights[i] = max(0.0, 1.0 - ewma[i])
                
            # Aggregate with EWMA downweighting
            total_weight = sum(weights)
            d = client_updates[0].numel()
            avg_update = torch.zeros(d, dtype=torch.float32)
            
            if total_weight > 1e-8:
                for i in range(5):
                    avg_update += (weights[i] / total_weight) * client_updates[i]
            else:
                avg_update = g0
                
            # Update global model
            offset = 0
            for key, p in global_weights.items():
                numel = p.numel()
                global_weights[key] = p + avg_update[offset:offset + numel].view_as(p)
                offset += numel
                
            global_model.load_state_dict(global_weights)
            
            eval_res = evaluate_model(global_model, test_loader)
            eval_res['round'] = r
            eval_res['seed'] = seed
            logs.append(eval_res)
            
        df_seed_log = pd.DataFrame(logs)
        df_seed_log.to_csv(f"logs/our_defense_seed{seed}.csv", index=False)
        
        final_res = logs[-1]
        all_seed_results.append({
            'strategy': 'OUR_DEFENSE',
            'seed': seed,
            'accuracy': final_res['accuracy'],
            'dp_gap': final_res['dp_gap'],
            'eo_gap': final_res['eo_gap']
        })
        
    df_defense_results = pd.DataFrame(all_seed_results)
    df_defense_results.to_csv("logs/our_defense_multiseed.csv", index=False)
    print("Completed Phase 8 & 9 Defense across 5 seeds!")
    return df_defense_results

if __name__ == "__main__":
    run_phase8_9_defense()
