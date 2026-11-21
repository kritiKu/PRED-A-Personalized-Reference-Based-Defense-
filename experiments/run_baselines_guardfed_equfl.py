import os, sys, time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import prepare_data
from models.mlp import SimpleMLP
from clients.honest import train_honest_client
from attacks.lora_attack import train_malicious_client
from defenses.metrics import evaluate_model
from defenses.guardfed import guardfed_aggregate
from defenses.equfl import equfl_aggregate

SEEDS = [43, 44, 45, 46, 47]
LR = 0.01; BATCH = 32; EPOCHS = 5
ROUNDS = 30
CONFIGS = {10: [3, 8], 20: [3, 8, 13, 18]}

def run_seed(seed, n_clients, attackers):
    tag = f"n{n_clients}"
    log_dir = f"logs/unified_{tag}"
    os.makedirs(log_dir, exist_ok=True)
    data_dir = f"data/seed_{seed}_{tag}"
    
    # Check if data exists, if not prepare it
    if not os.path.exists(f"{data_dir}/client_0.pt"):
        prepare_data(seed=seed, n_clients=n_clients, alpha=0.5, ref_size=250, save_dir=data_dir)
        
    cd = [torch.load(f"{data_dir}/client_{i}.pt", weights_only=False) for i in range(n_clients)]
    ref = torch.load(f"{data_dir}/reference.pt", weights_only=False)
    td = torch.load(f"{data_dir}/test.pt", weights_only=False)
    tl = DataLoader(TensorDataset(td['x'], td['y'], td['sensitive']), batch_size=64, shuffle=False)
    idim = cd[0]['x'].shape[1]
    
    torch.manual_seed(seed)
    im = SimpleMLP(idim)
    igw = {k: v.clone() for k, v in im.state_dict().items()}

    for sn in ["GuardFed", "EquFL"]:
        csv = f"{log_dir}/{sn.lower()}_seed{seed}.csv"
        print(f"  {sn}...", end="", flush=True)
        t0 = time.time()
        gw = {k: v.clone() for k, v in igw.items()}
        model = SimpleMLP(idim)
        logs = []
        
        # State for EquFL
        equfl_syn_data = None
        equfl_checkpoints = []
        
        for r in range(1, ROUNDS + 1):
            updates = [None] * n_clients
            sizes = [None] * n_clients
            hn = []
            
            for i in range(n_clients):
                if i in attackers: continue
                u, n = train_honest_client(SimpleMLP, idim, gw, cd[i], batch_size=BATCH, local_epochs=EPOCHS, lr=LR, seed=seed+r*100+i)
                updates[i] = u
                sizes[i] = n
                hn.append(u.norm().item())
                
            mn = float(np.median(hn))
            
            for a in attackers:
                u, n = train_malicious_client(SimpleMLP, idim, gw, cd[a], batch_size=BATCH, local_epochs=EPOCHS, lr=LR, seed=seed+r*100+a, target_norm=mn)
                updates[a] = u
                sizes[a] = n
                
            excl = []
            
            if sn == "GuardFed":
                # GuardFed: tau=5.0 is default from paper, gamma_threshold=0.1
                gw, excl = guardfed_aggregate(SimpleMLP, idim, gw, updates, sizes, ref, tau=5.0, gamma_threshold=0.1, ref_steps=50, seed=seed+r*555)
            elif sn == "EquFL":
                # EquFL: first 15 rounds collect, then synthetic distillation
                gw, equfl_syn_data, equfl_checkpoints = equfl_aggregate(
                    SimpleMLP, idim, gw, updates, sizes, ref,
                    gamma=1.0, syn_data=equfl_syn_data, checkpoints=equfl_checkpoints,
                    current_round=r-1, warmup_rounds=15, seed=seed+r*888
                )
                
            model.load_state_dict(gw)
            met = evaluate_model(model, tl)
            logs.append({
                'seed': seed, 'round': r, 'strategy': sn, 'excluded': str(excl),
                'accuracy': met['accuracy'], 'dp_gap': met['dp_gap'], 'eo_gap': met['eo_gap']
            })
            if r % 5 == 0:
                print(f"    Round {r} done.", flush=True)
            
        pd.DataFrame(logs).to_csv(csv, index=False)
        f = logs[-1]
        print(f" done ({time.time()-t0:.0f}s) Acc={f['accuracy']:.4f} DP={f['dp_gap']:.4f}", flush=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        seeds_to_run = [int(sys.argv[1])]
    else:
        seeds_to_run = SEEDS
    
    for n_clients, attackers in CONFIGS.items():
        print(f"{'='*50}\nRunning GuardFed and EquFL for N={n_clients}\n{'='*50}")
        for s in seeds_to_run:
            print(f"\nSeed {s}")
            run_seed(s, n_clients, attackers)
