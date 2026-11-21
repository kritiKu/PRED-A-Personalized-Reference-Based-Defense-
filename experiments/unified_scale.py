"""
Unified PRED-v2 Pipeline: N=5, N=10, N=20 (seeds 43-46, no seed 42)
Same pipeline everywhere:
  - Reference-prediction estimator (no ground truth needed)
  - IW-reweighted rank-residual
  - Dynamic MAD Z-score outlier exclusion (z > 3.0)
  - Fairness repair (8 steps, lambda=3.0)
Baselines: FedAvg, KRUM, FLTrust
Attackers: 20% in each case
"""
import os, sys, time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(8)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import prepare_data
from models.mlp import SimpleMLP
from clients.honest import train_honest_client
from attacks.lora_attack import train_malicious_client
from defenses.signal import compute_raw_rank_signal
from defenses.metrics import evaluate_model

SEEDS = [43, 44, 45, 46]   # No seed 42
LR = 0.01; BATCH = 32; EPOCHS = 5
IW_STEPS = 100
REPAIR_STEPS = 8; REPAIR_LR = 0.001; REPAIR_LAMBDA = 3.0
ROUNDS = 30

# Scale configurations: (n_clients, attacker_list)
CONFIGS = {
    5:  [3],              # 1 attacker = 20%
    10: [3, 8],           # 2 attackers = 20%
    20: [3, 8, 13, 18],   # 4 attackers = 20%
}

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm Helpers (same as before)
# ─────────────────────────────────────────────────────────────────────────────
def fedavg(gw, updates, sizes):
    total = sum(sizes)
    avg = sum((s/total)*u for u,s in zip(updates, sizes))
    offset = 0; new_gw = {}
    for key, par in gw.items():
        n = par.numel()
        new_gw[key] = par + avg[offset:offset+n].view_as(par)
        offset += n
    return new_gw

def fedavg_subset(gw, updates, sizes, excluded_indices):
    rem_u = [u for i,u in enumerate(updates) if i not in excluded_indices]
    rem_s = [s for i,s in enumerate(sizes) if i not in excluded_indices]
    if not rem_u:
        return {k: v.clone() for k, v in gw.items()}
    return fedavg(gw, rem_u, rem_s)

def krum(gw, updates, f=1):
    n = len(updates)
    k = max(1, n - f - 2)
    scores = []
    for i in range(n):
        dists = sorted([torch.norm(updates[i] - updates[j]).item()**2
                        for j in range(n) if j != i])
        scores.append(sum(dists[:k]))
    best = int(np.argmin(scores))
    offset = 0; new_gw = {}
    for key, par in gw.items():
        numel = par.numel()
        new_gw[key] = par + updates[best][offset:offset+numel].view_as(par)
        offset += numel
    return new_gw

def fltrust(gw, updates, ref_data, input_dim):
    ref_model = SimpleMLP(input_dim)
    ref_model.load_state_dict(gw)
    opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    for _ in range(200):
        opt.zero_grad()
        crit(ref_model(ref_data['x']), ref_data['y']).backward()
        opt.step()
    gv = torch.cat([p.data.view(-1) for p in gw.values()])
    rv = torch.cat([p.data.view(-1) for p in ref_model.parameters()])
    g0 = rv - gv
    g0_norm = torch.norm(g0) + 1e-9
    ts_list = [max(0.0, (torch.dot(u, g0) / ((torch.norm(u)+1e-9)*g0_norm)).item())
               for u in updates]
    ts_sum = sum(ts_list)
    if ts_sum < 1e-9:
        return {k: v.clone() for k, v in gw.items()}
    agg = sum((ts/ts_sum) * u * (g0_norm/(torch.norm(u)+1e-9))
              for ts, u in zip(ts_list, updates))
    offset = 0; new_gw = {}
    for key, par in gw.items():
        numel = par.numel()
        new_gw[key] = par + agg[offset:offset+numel].view_as(par)
        offset += numel
    return new_gw

def estimate_pos_rate(update_vec, global_weights, ref_data, input_dim):
    cw = {}; offset = 0
    for key, par in global_weights.items():
        numel = par.numel()
        cw[key] = par + update_vec[offset:offset+numel].view_as(par)
        offset += numel
    m = SimpleMLP(input_dim); m.load_state_dict(cw); m.eval()
    with torch.no_grad():
        probs = torch.softmax(m(ref_data['x']), dim=1)[:, 1]
    return float(np.clip(probs.mean().item(), 0.01, 0.99))

def compute_iw_g0(gw, ref_data, input_dim, target_pos, seed):
    torch.manual_seed(seed)
    ref_x, ref_y = ref_data['x'], ref_data['y']
    n_ref = len(ref_x)
    ref_pos = ref_y.float().mean().item()
    weights = torch.ones(n_ref)
    if 1e-4 < target_pos < 1-1e-4 and 1e-4 < ref_pos < 1-1e-4:
        for j in range(n_ref):
            weights[j] = (target_pos/ref_pos if ref_y[j].item()==1
                          else (1-target_pos)/(1-ref_pos))
    weights = weights * (n_ref / weights.sum())
    rm = SimpleMLP(input_dim); rm.load_state_dict(gw)
    opt = torch.optim.SGD(rm.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss(reduction='none')
    gen = torch.Generator().manual_seed(seed)
    steps = 0
    while steps < IW_STEPS:
        perm = torch.randperm(n_ref, generator=gen)
        for s in range(0, n_ref, 32):
            if steps >= IW_STEPS: break
            idx = perm[s:s+32]
            bw = weights[idx]; bw = bw / (bw.sum()+1e-8)
            opt.zero_grad()
            (crit(rm(ref_x[idx]), ref_y[idx]) * bw).sum().backward()
            opt.step(); steps += 1
    gv = torch.cat([p.data.view(-1) for p in gw.values()])
    rv = torch.cat([p.data.view(-1) for p in rm.parameters()])
    return rv - gv

def pred_v2_detect(updates, gw, ref_data, input_dim, seed):
    signals = []
    for i, u in enumerate(updates):
        est = estimate_pos_rate(u, gw, ref_data, input_dim)
        g0_i = compute_iw_g0(gw, ref_data, input_dim, est, seed + i*97)
        signals.append(compute_raw_rank_signal(u, g0_i, gw))
    scores = np.array(signals)
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    if mad < 1e-6: mad = 1e-4
    z = (scores - median) / (mad * 1.4826)
    excluded = np.where(z > 3.0)[0].tolist()
    return signals, excluded

def fairness_repair(gw, ref_data, input_dim, seed=None):
    if seed is not None: torch.manual_seed(seed)
    m = SimpleMLP(input_dim); m.load_state_dict(gw)
    ref_x, ref_y, ref_s = ref_data['x'], ref_data['y'], ref_data['sensitive']
    opt = torch.optim.SGD(m.parameters(), lr=REPAIR_LR)
    for _ in range(REPAIR_STEPS):
        opt.zero_grad()
        out = m(ref_x)
        ce = nn.CrossEntropyLoss()(out, ref_y)
        pr = torch.softmax(out, dim=1)[:,1]
        pm = pr[ref_s==1].mean() if (ref_s==1).sum()>0 else pr.mean()
        pf = pr[ref_s==0].mean() if (ref_s==0).sum()>0 else pr.mean()
        (ce + REPAIR_LAMBDA * torch.abs(pm - pf)).backward()
        opt.step()
    return {k: v.clone() for k, v in m.state_dict().items()}

# ─────────────────────────────────────────────────────────────────────────────
# Unified Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_config(n_clients, attackers, seed):
    tag = f"n{n_clients}"
    log_dir = f"logs/unified_{tag}"
    os.makedirs(log_dir, exist_ok=True)

    data_dir = f"data/seed_{seed}_{tag}"
    prepare_data(seed=seed, n_clients=n_clients, alpha=0.5, ref_size=250, save_dir=data_dir)
    client_data = [torch.load(f"{data_dir}/client_{i}.pt", weights_only=False)
                   for i in range(n_clients)]
    ref_data  = torch.load(f"{data_dir}/reference.pt", weights_only=False)
    test_data = torch.load(f"{data_dir}/test.pt", weights_only=False)
    test_loader = DataLoader(TensorDataset(test_data['x'], test_data['y'],
                                            test_data['sensitive']),
                              batch_size=64, shuffle=False)
    input_dim = client_data[0]['x'].shape[1]
    torch.manual_seed(seed)
    init_model = SimpleMLP(input_dim)
    init_gw = {k: v.clone() for k, v in init_model.state_dict().items()}
    n_attackers = len(attackers)
    honest = [i for i in range(n_clients) if i not in attackers]

    for strat_name in ["FedAvg", "KRUM", "FLTrust", "PRED-v2"]:
        csv_path = f"{log_dir}/{strat_name.lower()}_seed{seed}.csv"
        if os.path.exists(csv_path):
            print(f"    SKIP {strat_name} (exists)", flush=True)
            continue
        print(f"    {strat_name}...", end="", flush=True)
        t0 = time.time()
        gw = {k: v.clone() for k, v in init_gw.items()}
        model = SimpleMLP(input_dim)
        logs = []
        for r in range(1, ROUNDS+1):
            updates = [None]*n_clients; sizes = [None]*n_clients
            honest_norms = []
            for i in range(n_clients):
                if i in attackers: continue
                upd, n = train_honest_client(SimpleMLP, input_dim, gw, client_data[i],
                                             batch_size=BATCH, local_epochs=EPOCHS,
                                             lr=LR, seed=seed+r*100+i)
                updates[i]=upd; sizes[i]=n; honest_norms.append(upd.norm().item())
            med_norm = float(np.median(honest_norms))
            for a in attackers:
                upd, n = train_malicious_client(SimpleMLP, input_dim, gw, client_data[a],
                                                batch_size=BATCH, local_epochs=EPOCHS,
                                                lr=LR, seed=seed+r*100+a,
                                                target_norm=med_norm)
                updates[a]=upd; sizes[a]=n

            excluded_list = []
            if strat_name == "FedAvg":
                gw = fedavg(gw, updates, sizes)
            elif strat_name == "KRUM":
                gw = krum(gw, updates, f=n_attackers)
            elif strat_name == "FLTrust":
                gw = fltrust(gw, updates, ref_data, input_dim)
            elif strat_name == "PRED-v2":
                _, excluded_list = pred_v2_detect(updates, gw, ref_data, input_dim,
                                                  seed=seed+r*911)
                gw = fedavg_subset(gw, updates, sizes, excluded_list)
                gw = fairness_repair(gw, ref_data, input_dim, seed=seed+r*777)

            model.load_state_dict(gw)
            metrics = evaluate_model(model, test_loader)
            logs.append({'seed': seed, 'round': r, 'strategy': strat_name,
                         'excluded': str(excluded_list),
                         'accuracy': metrics['accuracy'],
                         'dp_gap': metrics['dp_gap'],
                         'eo_gap': metrics['eo_gap']})

        pd.DataFrame(logs).to_csv(csv_path, index=False)
        elapsed = time.time() - t0
        final = logs[-1]
        print(f" done ({elapsed:.0f}s) DP={final['dp_gap']:.4f}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for nc in [5, 10, 20]:
        attackers = CONFIGS[nc]
        print(f"\n{'='*60}")
        print(f"  N={nc} clients, {len(attackers)} attackers ({attackers}), seeds {SEEDS}")
        print(f"{'='*60}")
        for seed in SEEDS:
            print(f"  Seed {seed}:", flush=True)
            run_config(nc, attackers, seed)
    print("\n*** ALL CONFIGURATIONS COMPLETE ***")
