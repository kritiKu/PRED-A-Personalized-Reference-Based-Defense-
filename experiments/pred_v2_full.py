"""
PRED-v2: Importance-Weighted Rank-Residual + Fairness Repair
=============================================================
Combines two simultaneous fixes:

Step (1) -- Corrected detection:
  For each client i, compute a client-specific reference g0^i by training
  the 250-sample reference set with importance weights that match client i's
  local class balance (target_pos_rate_i), so skewed-data honest clients
  (e.g., C1 with 6.7% positive in seed 42) are NOT mistaken for attackers.

Step (2) -- Fairness repair (unconditional):
  After each round's FedAvg aggregation, run REPAIR_STEPS extra SGD steps
  on the balanced reference set with loss = CE + lambda * |DP_gap|.
  Applied every round regardless of who was excluded.

Comparison baselines (cached logs):
  - FedAvg        (logs/sweep_fedavg_seed*.csv)
  - KRUM          (logs/sweep_krum_seed*.csv)
  - FLTrust       (logs/sweep_fltrust_seed*.csv)
  - PRED-v1       (logs/pred_seed*.csv)

Outputs:
  logs/pred_v2_seed*.csv
  logs/pred_v2_final.csv
  results/pred_v2_exclusion_accuracy.png
  results/pred_v2_vs_baselines.png
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(8)
if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.preprocess import prepare_data
from models.mlp import SimpleMLP
from clients.honest import train_honest_client
from attacks.lora_attack import train_malicious_client
from defenses.aggregators import fedavg_aggregate
from defenses.signal import compute_raw_rank_signal
from defenses.metrics import evaluate_model

os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

SEEDS     = [42, 43, 44, 45, 46]
N_CLIENTS = 5
ATTACKER  = 3
ROUNDS    = 30
LR        = 0.01
BATCH     = 32
EPOCHS    = 5

# Detection
IW_STEPS = 100     # steps for importance-weighted g0 (per client) -- 5x100=500/round

# Fairness repair
REPAIR_STEPS  = 8
REPAIR_LR     = 0.001
REPAIR_LAMBDA = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Importance-weighted FLTrust reference
# ─────────────────────────────────────────────────────────────────────────────
def compute_iw_g0(global_weights, ref_data, input_dim,
                   target_pos_rate, seed, target_steps=IW_STEPS, batch_size=32):
    """
    Train reference model on ref_data with per-sample importance weights
    that rebalance class proportions to match target_pos_rate.
    Returns g0 = ref_model_weights - global_weights (flattened update vector).
    """
    torch.manual_seed(seed)
    ref_x = ref_data['x']
    ref_y = ref_data['y']
    n_ref = len(ref_x)

    ref_pos_rate = ref_y.float().mean().item()

    # Compute per-sample importance weights
    weights = torch.ones(n_ref, dtype=torch.float32)
    if 1e-4 < target_pos_rate < 1 - 1e-4 and 1e-4 < ref_pos_rate < 1 - 1e-4:
        for j in range(n_ref):
            if ref_y[j].item() == 1:
                weights[j] = target_pos_rate / ref_pos_rate
            else:
                weights[j] = (1.0 - target_pos_rate) / (1.0 - ref_pos_rate)
    # Normalize: weights sum to n_ref to keep gradient scale stable
    weights = weights * (n_ref / weights.sum())

    ref_model = SimpleMLP(input_dim)
    ref_model.load_state_dict(global_weights)
    optimizer  = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    criterion  = nn.CrossEntropyLoss(reduction='none')

    generator = torch.Generator().manual_seed(seed)
    steps = 0
    while steps < target_steps:
        perm = torch.randperm(n_ref, generator=generator)
        for start in range(0, n_ref, batch_size):
            if steps >= target_steps:
                break
            idx = perm[start:start + batch_size]
            bx  = ref_x[idx]
            by  = ref_y[idx]
            bw  = weights[idx]
            bw  = bw / (bw.sum() + 1e-8)  # normalize batch weights

            optimizer.zero_grad()
            per_loss = criterion(ref_model(bx), by)
            loss     = (per_loss * bw).sum()
            loss.backward()
            optimizer.step()
            steps += 1

    global_vec = torch.cat([p.data.view(-1) for p in global_weights.values()])
    ref_vec    = torch.cat([p.data.view(-1) for p in ref_model.parameters()])
    return ref_vec - global_vec


# ─────────────────────────────────────────────────────────────────────────────
# PRED-v2 signal: per-client importance-weighted rank-residual
# ─────────────────────────────────────────────────────────────────────────────
def compute_pred_v2_signals(client_updates, client_data_list, global_weights,
                             ref_data, input_dim, seed):
    """
    For each client i:
      1. Compute target_pos_rate from client i's local labels
      2. Compute g0^i via importance-weighted reference training
      3. rank_signal_i = compute_raw_rank_signal(u_i, g0^i, global_weights)
    Most suspicious = argmax(rank_signals).
    """
    rank_signals = []
    for i, (u, cd) in enumerate(zip(client_updates, client_data_list)):
        target_pos = float(cd['y'].float().mean().item())
        g0_i = compute_iw_g0(
            global_weights, ref_data, input_dim,
            target_pos_rate=target_pos,
            seed=seed + i * 97,
            target_steps=IW_STEPS,
            batch_size=32
        )
        rs = compute_raw_rank_signal(u, g0_i, global_weights)
        rank_signals.append(rs)

    excluded = int(np.argmax(rank_signals))
    return rank_signals, excluded


# ─────────────────────────────────────────────────────────────────────────────
# Fairness repair step (unconditional, every round)
# ─────────────────────────────────────────────────────────────────────────────
def fairness_repair(global_weights, ref_data, input_dim,
                    n_steps=REPAIR_STEPS, lr=REPAIR_LR, lam=REPAIR_LAMBDA,
                    seed=None):
    """
    Fine-tune global model with fairness-penalized loss on reference set.
    loss = CrossEntropy + lam * |P(y=1|male) - P(y=1|female)|
    Returns updated global_weights dict.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = SimpleMLP(input_dim)
    model.load_state_dict(global_weights)
    model.train()

    ref_x = ref_data['x']
    ref_y = ref_data['y']
    ref_s = ref_data['sensitive']

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss()

    for _ in range(n_steps):
        optimizer.zero_grad()
        out   = model(ref_x)
        ce    = criterion(out, ref_y)
        probs = torch.softmax(out, dim=1)[:, 1]

        m_mask = (ref_s == 1)
        f_mask = (ref_s == 0)
        p_m = probs[m_mask].mean() if m_mask.sum() > 0 else probs.mean()
        p_f = probs[f_mask].mean() if f_mask.sum() > 0 else probs.mean()

        (ce + lam * torch.abs(p_m - p_f)).backward()
        optimizer.step()

    return {k: v.clone() for k, v in model.state_dict().items()}


# ─────────────────────────────────────────────────────────────────────────────
# FedAvg over a subset of clients
# ─────────────────────────────────────────────────────────────────────────────
def fedavg_subset(global_weights, all_updates, all_sizes, excluded_idx):
    remaining_updates = [u for i, u in enumerate(all_updates) if i != excluded_idx]
    remaining_sizes   = [s for i, s in enumerate(all_sizes)   if i != excluded_idx]
    total = sum(remaining_sizes)
    d     = remaining_updates[0].numel()
    avg   = torch.zeros(d)
    for u, s in zip(remaining_updates, remaining_sizes):
        avg += (s / total) * u
    new_weights = {}
    offset = 0
    for key, p in global_weights.items():
        numel = p.numel()
        new_weights[key] = p + avg[offset:offset + numel].view_as(p)
        offset += numel
    return new_weights


# ─────────────────────────────────────────────────────────────────────────────
# Main PRED-v2 experiment
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("PRED-v2: IW-Rank-Residual + Fairness Repair")
print(f"  IW_STEPS={IW_STEPS}/client | Repair: {REPAIR_STEPS} steps, "
      f"lr={REPAIR_LR}, lam={REPAIR_LAMBDA}")
print("="*65)

pred_v2_final = []
pred_v2_round_logs = []

for seed in SEEDS:
    print(f"\n-- Seed {seed} --", flush=True)
    data_dir = f"data/seed_{seed}"
    prepare_data(seed=seed, n_clients=N_CLIENTS, alpha=0.5,
                 ref_size=250, save_dir=data_dir)

    client_data = [
        torch.load(f"{data_dir}/client_{i}.pt", weights_only=False)
        for i in range(N_CLIENTS)
    ]
    ref_data  = torch.load(f"{data_dir}/reference.pt", weights_only=False)
    test_data = torch.load(f"{data_dir}/test.pt",      weights_only=False)
    test_loader = DataLoader(
        TensorDataset(test_data['x'], test_data['y'], test_data['sensitive']),
        batch_size=64, shuffle=False
    )
    input_dim = client_data[0]['x'].shape[1]

    # Print partition info for reference
    for i in range(N_CLIENTS):
        cd = client_data[i]
        pos = cd['y'].float().mean().item()
        tag = " (ATTACKER)" if i == ATTACKER else ""
        print(f"  C{i}{tag}: n={len(cd['y'])}, pos_rate={pos:.3f}", flush=True)

    torch.manual_seed(seed)
    global_model   = SimpleMLP(input_dim)
    global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

    c3_correct    = 0
    honest_false  = 0
    round_logs    = []

    for r in range(1, ROUNDS + 1):
        # Train all clients
        honest_norms = []
        updates = [None] * N_CLIENTS
        sizes   = [None] * N_CLIENTS

        for i in range(N_CLIENTS):
            if i == ATTACKER: continue
            c_seed = seed + r * 100 + i
            upd, n = train_honest_client(
                SimpleMLP, input_dim, global_weights, client_data[i],
                batch_size=BATCH, local_epochs=EPOCHS, lr=LR, seed=c_seed
            )
            updates[i] = upd
            sizes[i]   = n
            honest_norms.append(upd.norm().item())

        med_norm = float(np.median(honest_norms))
        att_seed = seed + r * 100 + ATTACKER
        att_upd, att_n = train_malicious_client(
            SimpleMLP, input_dim, global_weights, client_data[ATTACKER],
            batch_size=BATCH, local_epochs=EPOCHS, lr=LR,
            seed=att_seed, target_norm=med_norm
        )
        updates[ATTACKER] = att_upd
        sizes[ATTACKER]   = att_n

        # ── Step 1: IW rank-residual detection ───────────────────────────────
        det_seed = seed + r * 911
        rank_signals, excluded = compute_pred_v2_signals(
            updates, client_data, global_weights, ref_data, input_dim, seed=det_seed
        )

        correct = (excluded == ATTACKER)
        if correct: c3_correct   += 1
        else:       honest_false += 1

        # ── Aggregate remaining 4 clients ────────────────────────────────────
        global_weights = fedavg_subset(global_weights, updates, sizes, excluded)

        # ── Step 2: Fairness repair (unconditional) ───────────────────────────
        repair_seed = seed + r * 777
        global_weights = fairness_repair(
            global_weights, ref_data, input_dim,
            n_steps=REPAIR_STEPS, lr=REPAIR_LR, lam=REPAIR_LAMBDA,
            seed=repair_seed
        )

        global_model.load_state_dict(global_weights)
        metrics = evaluate_model(global_model, test_loader)

        arr = np.array(rank_signals)
        mu, sig = arr.mean(), arr.std() + 1e-8
        z_c3 = float((arr[ATTACKER] - mu) / sig)

        round_logs.append({
            'seed': seed, 'round': r,
            'excluded_client': excluded,
            'excluded_correct': int(correct),
            'rank_signal_c3': rank_signals[ATTACKER],
            'z_score_c3': z_c3,
            **metrics
        })

        if r % 10 == 0:
            print(f"  R{r:02d}: excl=C{excluded} "
                  f"({'OK' if correct else 'MISS'})  "
                  f"DP={metrics['dp_gap']:.4f}", flush=True)

    df_seed = pd.DataFrame(round_logs)
    df_seed.to_csv(f"logs/pred_v2_seed{seed}.csv", index=False)
    pred_v2_round_logs.append(df_seed)

    final = round_logs[-1]
    pred_v2_final.append({
        'strategy': 'PRED-v2', 'seed': seed,
        'accuracy':  final['accuracy'],
        'dp_gap':    final['dp_gap'],
        'eo_gap':    final['eo_gap'],
        'c3_excl_rounds':    c3_correct,
        'honest_excl_rounds': honest_false,
    })
    print(f"  C3 correct: {c3_correct}/30  |  Honest false: {honest_false}/30  "
          f"|  Final DP={final['dp_gap']:.4f}  Acc={final['accuracy']:.4f}")

df_final = pd.DataFrame(pred_v2_final)
df_final.to_csv("logs/pred_v2_final.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Load cached baselines + PRED-v1
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- Loading cached baselines --")
baselines = {}
for name, pattern in [('FEDAVG','fedavg'), ('KRUM','krum'), ('FLTRUST','fltrust')]:
    rows = []
    for seed in SEEDS:
        p = f"logs/sweep_{pattern}_seed{seed}.csv"
        if os.path.exists(p):
            df_b = pd.read_csv(p)
            r = df_b.iloc[-1]
            rows.append({'strategy': name, 'seed': seed,
                         'accuracy': r['accuracy'], 'dp_gap': r['dp_gap'],
                         'eo_gap': r['eo_gap']})
    if rows:
        baselines[name] = pd.DataFrame(rows)
        print(f"  {name}: {len(rows)} seeds")

# PRED-v1
predv1_rows = []
for seed in SEEDS:
    p = f"logs/pred_seed{seed}.csv"
    if os.path.exists(p):
        df_p = pd.read_csv(p)
        r = df_p.iloc[-1]
        predv1_rows.append({'strategy': 'PRED-v1', 'seed': seed,
                            'accuracy': r['accuracy'], 'dp_gap': r['dp_gap'],
                            'eo_gap': r['eo_gap'],
                            'c3_excl_rounds': df_p['excluded_correct'].sum()})
if predv1_rows:
    baselines['PRED-v1'] = pd.DataFrame(predv1_rows)
    print(f"  PRED-v1: {len(predv1_rows)} seeds")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*65)
print("EXCLUSION ACCURACY: PRED-v2")
print("="*65)
total_c3 = sum(r['c3_excl_rounds'] for r in pred_v2_final)
total_ho = sum(r['honest_excl_rounds'] for r in pred_v2_final)
for row in pred_v2_final:
    print(f"  Seed {row['seed']}: C3 excl={row['c3_excl_rounds']:2d}/30  "
          f"Honest false={row['honest_excl_rounds']:2d}/30")
print(f"  Pooled: C3={total_c3}/150 ({total_c3/150*100:.1f}%)  "
      f"Honest={total_ho}/150 ({total_ho/150*100:.1f}%)")
bp = binomtest(total_c3, 150, 0.2, alternative='greater').pvalue
print(f"  Binomial test vs 20% chance: p={bp:.4f}")

print()
print("="*65)
print("FINAL METRICS -- ALL STRATEGIES")
print("="*65)
all_rows = [{'strategy': r['strategy'], 'seed': r['seed'],
             'accuracy': r['accuracy'], 'dp_gap': r['dp_gap'],
             'eo_gap': r['eo_gap']} for r in pred_v2_final]
for df_b in baselines.values():
    all_rows.extend(df_b[['strategy','seed','accuracy','dp_gap','eo_gap']].to_dict('records'))
df_all = pd.DataFrame(all_rows)

for strat in ['PRED-v2', 'PRED-v1', 'FEDAVG', 'KRUM', 'FLTRUST']:
    sub = df_all[df_all['strategy'] == strat]
    if sub.empty: continue
    print(f"  {strat:10s}: Acc={sub['accuracy'].mean():.4f}+/-{sub['accuracy'].std():.4f}  "
          f"DP={sub['dp_gap'].mean():.4f}+/-{sub['dp_gap'].std():.4f}  "
          f"EO={sub['eo_gap'].mean():.4f}+/-{sub['eo_gap'].std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Paired sign tests (PRED-v2 vs each baseline on DP)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*65)
print("PAIRED SIGN TESTS (PRED-v2 vs Baselines on DP gap)")
print("  H1: PRED-v2 achieves lower DP than baseline")
print("="*65)

v2_dp = {r['seed']: r['dp_gap'] for r in pred_v2_final}

for name, df_b in baselines.items():
    wins = losses = ties = 0
    diffs = []
    for _, row in df_b.iterrows():
        s = row['seed']
        if s not in v2_dp: continue
        d = v2_dp[s] - row['dp_gap']
        diffs.append(d)
        if d < -1e-6:   wins   += 1
        elif d > 1e-6:  losses += 1
        else:           ties   += 1

    ne = wins + losses
    pv = binomtest(wins, ne, 0.5, alternative='less').pvalue if ne > 0 else 1.0
    sig = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.1 else ""))
    mean_d = np.mean(diffs) if diffs else 0.0
    print(f"\n  PRED-v2 vs {name}:")
    print(f"    Diffs: {[f'{v:+.4f}' for v in diffs]}")
    print(f"    Better: {wins} | Tied: {ties} | Worse: {losses}")
    print(f"    Sign test p={pv:.4f}{' '+sig if sig else ''}  |  Mean diff: {mean_d:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {'PRED-v2':'#e74c3c', 'PRED-v1':'#e67e22',
          'FEDAVG':'#95a5a6', 'KRUM':'#3498db', 'FLTRUST':'#27ae60'}

# Plot 1: Exclusion accuracy per seed
fig1, ax1 = plt.subplots(figsize=(11, 5))
x  = np.arange(len(SEEDS))
bw = 0.3
c3c = [r['c3_excl_rounds']    for r in pred_v2_final]
hof = [r['honest_excl_rounds'] for r in pred_v2_final]

# Also load PRED-v1 c3 counts for comparison
v1c3 = [baselines['PRED-v1'][baselines['PRED-v1']['seed']==s]['c3_excl_rounds'].values[0]
        if 'PRED-v1' in baselines and not baselines['PRED-v1'][baselines['PRED-v1']['seed']==s].empty
        else 0 for s in SEEDS] if 'PRED-v1' in baselines else [0]*5

ax1.bar(x - bw, v1c3, bw, label='PRED-v1 C3 correct',
        color='#e67e22', alpha=0.65, edgecolor='black', lw=0.7)
ax1.bar(x,      c3c,  bw, label='PRED-v2 C3 correct',
        color='#e74c3c', alpha=0.82, edgecolor='black', lw=0.7)
ax1.bar(x + bw, hof,  bw, label='PRED-v2 Honest false excl.',
        color='#95a5a6', alpha=0.70, edgecolor='black', lw=0.7)
ax1.axhline(6, color='navy', linestyle='--', lw=1.5, label='20% chance (6 rounds)')

for i, (v, c, h) in enumerate(zip(v1c3, c3c, hof)):
    ax1.text(x[i]-bw, v+0.3, str(int(v)), ha='center', va='bottom', fontsize=9, color='#c0392b')
    ax1.text(x[i],    c+0.3, str(c),      ha='center', va='bottom', fontsize=10,
             fontweight='bold', color='#922b21')
    ax1.text(x[i]+bw, h+0.3, str(h),      ha='center', va='bottom', fontsize=9, color='#555')

ax1.set_xticks(x)
ax1.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=11)
ax1.set_ylabel("Rounds excluded (out of 30)", fontsize=11)
ax1.set_ylim(0, 33)
ax1.set_title(
    f"PRED-v2 Exclusion Accuracy: IW-Rank-Residual vs Original\n"
    f"Pooled C3: PRED-v2={total_c3}/150 ({total_c3/150*100:.1f}%)  |  "
    f"Binom p={bp:.4f}",
    fontsize=11, fontweight='bold'
)
ax1.legend(fontsize=9)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("results/pred_v2_exclusion_accuracy.png", dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved results/pred_v2_exclusion_accuracy.png")

# Plot 2: Fairness + accuracy comparison
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 6))
strats = ['PRED-v2', 'PRED-v1', 'FEDAVG', 'KRUM', 'FLTRUST']
met_cfg = [('dp_gap', 'DP Gap (lower=better)', True),
           ('eo_gap', 'EO Gap (lower=better)', True),
           ('accuracy', 'Accuracy (higher=better)', False)]

for ax, (metric, ylabel, lower_better) in zip(axes2, met_cfg):
    for j, strat in enumerate(strats):
        sub = df_all[df_all['strategy'] == strat]
        if sub.empty: continue
        vals = sub[metric].values
        bp_  = ax.boxplot(vals, positions=[j], widths=0.5, patch_artist=True,
                          medianprops=dict(color='black', linewidth=2),
                          flierprops=dict(marker='o', markersize=3, alpha=0.4))
        bp_['boxes'][0].set_facecolor(COLORS.get(strat, '#888'))
        bp_['boxes'][0].set_alpha(0.78)
        ax.text(j, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
                f"{vals.mean():.3f}", ha='center', va='top', fontsize=8.5, color='#333')

    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels([s.replace('-', '\n') for s in strats], fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

legend_patches = [plt.Rectangle((0,0),1,1, fc=COLORS.get(s,'#888'),
                                 alpha=0.78, label=s) for s in strats]
fig2.legend(handles=legend_patches, loc='lower center', ncol=5,
            fontsize=10, bbox_to_anchor=(0.5, 0.01))
fig2.suptitle(
    "PRED-v2 vs Baselines -- Final Round Metrics (5 seeds x 30 rounds)\n"
    "Attack: norm-matched LoRA | PRED-v2: IW-rank-residual + fairness repair",
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("results/pred_v2_vs_baselines.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved results/pred_v2_vs_baselines.png")

print()
print("="*65)
print("ALL DONE")
print("="*65)
