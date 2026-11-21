import os, sys, time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)  # Prevent hanging with multiple defenses
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import prepare_data
from models.mlp import SimpleMLP
from clients.honest import train_honest_client
from attacks.lora_attack import train_malicious_client
from defenses.signal import compute_raw_rank_signal
from defenses.metrics import evaluate_model
from defenses.guardfed import guardfed_aggregate
from defenses.dpg_fairfl import dpg_fairfl_aggregate

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helper functions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    while steps < 100:
        perm = torch.randperm(n_ref, generator=gen)
        for s in range(0, n_ref, 32):
            if steps >= 100: break
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
    opt = torch.optim.SGD(m.parameters(), lr=0.001)
    for _ in range(8):
        opt.zero_grad()
        out = m(ref_x)
        ce = nn.CrossEntropyLoss()(out, ref_y)
        pr = torch.softmax(out, dim=1)[:,1]
        pm = pr[ref_s==1].mean() if (ref_s==1).sum()>0 else pr.mean()
        pf = pr[ref_s==0].mean() if (ref_s==0).sum()>0 else pr.mean()
        (ce + 3.0 * torch.abs(pm - pf)).backward()
        opt.step()
    return {k: v.clone() for k, v in m.state_dict().items()}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Runner
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SEEDS = [43, 44, 45, 46, 47]
LR = 0.01; BATCH = 32; EPOCHS = 5; ROUNDS = 30
N_CLIENTS = 5
ATTACKERS = [3]

def run_seed_all6(seed, r_val):
    tag = f"unified_n{N_CLIENTS}"
    log_dir = f"logs/ablation_pred/r{r_val}"
    os.makedirs(log_dir, exist_ok=True)
    
    data_dir = f"data/seed_{seed}_{tag}"
    if not os.path.exists(f"{data_dir}/client_0.pt"):
        prepare_data(seed=seed, n_clients=N_CLIENTS, alpha=0.5, ref_size=250, save_dir=data_dir)
        
    cd = [torch.load(f"{data_dir}/client_{i}.pt", weights_only=False) for i in range(N_CLIENTS)]
    ref = torch.load(f"{data_dir}/reference.pt", weights_only=False)
    td = torch.load(f"{data_dir}/test.pt", weights_only=False)
    tl = DataLoader(TensorDataset(td['x'], td['y'], td['sensitive']), batch_size=64, shuffle=False)
    idim = cd[0]['x'].shape[1]
    
    torch.manual_seed(seed)
    im = SimpleMLP(idim)
    igw = {k: v.clone() for k, v in im.state_dict().items()}
    n_attackers = len(ATTACKERS)

    strategies = ["FedAvg", "PRED-v2", "PRED-v2-DetectOnly", "PRED-v2-RepairOnly"]
    
    for sn in strategies:
        csv = f"{log_dir}/{sn.lower()}_seed{seed}.csv"
        if os.path.exists(csv):
            print(f"  SKIP {sn} (exists)", flush=True)
            continue
            
        print(f"  {sn}...", end="", flush=True)
        t0 = time.time()
        gw = {k: v.clone() for k, v in igw.items()}
        model = SimpleMLP(idim)
        logs = []
        
        # State for DPG-FairFL
        dpg_generator = None
        
        for r in range(1, ROUNDS + 1):
            updates = [None] * N_CLIENTS
            sizes = [None] * N_CLIENTS
            hn = []
            
            for i in range(N_CLIENTS):
                if i in ATTACKERS: continue
                u, n = train_honest_client(SimpleMLP, idim, gw, cd[i], batch_size=BATCH, local_epochs=EPOCHS, lr=LR, seed=seed+r*100+i)
                updates[i] = u
                sizes[i] = n
                hn.append(u.norm().item())
                
            mn = float(np.median(hn))
            
            for a in ATTACKERS:
                u, n = train_malicious_client(SimpleMLP, idim, gw, cd[a], batch_size=BATCH, local_epochs=EPOCHS, lr=LR, seed=seed+r*100+a, target_norm=mn, r=r_val)
                updates[a] = u
                sizes[a] = n
                
            excl = []
            
            if sn == "FedAvg":
                gw = fedavg(gw, updates, sizes)
            elif sn == "KRUM":
                gw = krum(gw, updates, f=n_attackers)
            elif sn == "FLTrust":
                gw = fltrust(gw, updates, ref, idim)
            elif sn == "GuardFed":
                gw, excl = guardfed_aggregate(SimpleMLP, idim, gw, updates, sizes, ref, tau=5.0, gamma_threshold=0.1, ref_steps=50, seed=seed+r*555)
            elif sn == "DPG-FairFL":
                gw, dpg_generator = dpg_fairfl_aggregate(
                    SimpleMLP, idim, gw, updates, sizes, ref,
                    generator=dpg_generator, gamma=5.0, seed=seed+r*888
                )
            elif sn == "PRED-v2":
                _, excl = pred_v2_detect(updates, gw, ref, idim, seed=seed+r*911)
                gw = fedavg_subset(gw, updates, sizes, excl)
                gw = fairness_repair(gw, ref, idim, seed=seed+r*777)
            elif sn == "PRED-v2-DetectOnly":
                _, excl = pred_v2_detect(updates, gw, ref, idim, seed=seed+r*911)
                gw = fedavg_subset(gw, updates, sizes, excl)
            elif sn == "PRED-v2-RepairOnly":
                excl = []
                gw = fedavg_subset(gw, updates, sizes, excl)
                gw = fairness_repair(gw, ref, idim, seed=seed+r*777)
                
            model.load_state_dict(gw)
            met = evaluate_model(model, tl)
            logs.append({
                'seed': seed, 'round': r, 'strategy': sn, 'excluded': str(excl),
                'accuracy': met['accuracy'], 'dp_gap': met['dp_gap'], 'eo_gap': met['eo_gap']
            })
            
        pd.DataFrame(logs).to_csv(csv, index=False)
        f = logs[-1]
        print(f" done ({time.time()-t0:.0f}s) Acc={f['accuracy']:.4f} DP={f['dp_gap']:.4f}", flush=True)

if __name__ == "__main__":
    ranks = [2, 4, 8, 16]
    print(f"{'='*60}\nRunning Adult Dataset Ablation - PRED-v2\n{'='*60}")
    for r_val in ranks:
        print(f"\n=================== RANK {r_val} ===================")
        for s in SEEDS:
            print(f"\nSeed {s}")
            run_seed_all6(s, r_val)
