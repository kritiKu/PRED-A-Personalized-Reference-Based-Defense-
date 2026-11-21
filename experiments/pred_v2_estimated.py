"""
PRED-v2-Est: Reference-Prediction-Based Class Balance Estimation
=================================================================
Real-world variant of PRED-v2: instead of using the TRUE known local
class balance (pos_rate_i = mean(y) from client_data[i]), we ESTIMATE
each client's class balance using only information the server already has:
the submitted update Δ_i and the server's own balanced reference set D_ref.

Estimator v1 FAILED (bias-shift approach):
  softmax(out.bias_shift)[1] → all estimates cluster at ~0.50
  Reason: norm-matching clips the entire update to median norm, making
  the 2-dimensional bias difference negligible. Confirmed: all estimates
  were 0.48-0.52 regardless of true pos_rate. Seed 42 C3 detection: 6/30.

Estimator v2 (this script) — Reference Prediction:
  1. Server reconstructs client i's local model: w_i = w_global + Δ_i
  2. Run w_i on server's balanced reference set (250 samples)
  3. pos_rate_est = mean(P(ŷ=1 | X_ref, w_i))  = predicted pos rate
  Rationale:
  - A client trained on 6.7% positive data (C1) has a model biased toward
    predicting 0 → low predicted pos_rate on balanced ref
  - A client trained on 93.1% positive data (C3) has a model biased toward
    predicting 1 → high predicted pos_rate on balanced ref
  - Fully server-side: only uses update + reference set (no client data revealed)
  - Invariant to norm-matching: prediction is from model output, not gradient magnitude

Comparison target: PRED-v2 ground-truth (logs/pred_v2_seed*.csv)
                   Detection accuracy = 142/150 (94.7%)
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
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
IW_STEPS  = 100
REPAIR_STEPS  = 8
REPAIR_LR     = 0.001
REPAIR_LAMBDA = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Estimator v2: pos_rate from reference-set predictions of client's model
# ─────────────────────────────────────────────────────────────────────────────
def estimate_pos_rate_from_prediction(update_vec, global_weights, ref_data, input_dim):
    """
    Server reconstructs client i's local model from the submitted update,
    then runs it on the server's balanced reference set to get a predicted
    positive rate. This correlates with local class balance without the client
    revealing any data.

    update_vec: flattened 1D tensor (global_weights + Δ_i = local model)
    Returns: predicted positive class rate in (0.01, 0.99)
    """
    # Reconstruct client's local model weights: w_i = w_global + Δ_i
    client_weights = {}
    offset = 0
    for key, param in global_weights.items():
        numel = param.numel()
        delta = update_vec[offset:offset + numel].view_as(param)
        client_weights[key] = param + delta
        offset += numel

    model = SimpleMLP(input_dim)
    model.load_state_dict(client_weights)
    model.eval()

    with torch.no_grad():
        out   = model(ref_data['x'])
        probs = torch.softmax(out, dim=1)[:, 1]   # P(ŷ=1) for each ref sample
        pos_rate_est = float(probs.mean().item())

    return float(np.clip(pos_rate_est, 0.01, 0.99))





# ─────────────────────────────────────────────────────────────────────────────
# IW reference computation (same as PRED-v2, but uses estimated pos_rate)
# ─────────────────────────────────────────────────────────────────────────────
def compute_iw_g0(global_weights, ref_data, input_dim,
                   target_pos_rate, seed, target_steps=IW_STEPS, batch_size=32):
    torch.manual_seed(seed)
    ref_x = ref_data['x']
    ref_y = ref_data['y']
    n_ref = len(ref_x)
    ref_pos_rate = ref_y.float().mean().item()

    weights = torch.ones(n_ref, dtype=torch.float32)
    if 1e-4 < target_pos_rate < 1 - 1e-4 and 1e-4 < ref_pos_rate < 1 - 1e-4:
        for j in range(n_ref):
            if ref_y[j].item() == 1:
                weights[j] = target_pos_rate / ref_pos_rate
            else:
                weights[j] = (1.0 - target_pos_rate) / (1.0 - ref_pos_rate)
    weights = weights * (n_ref / weights.sum())

    ref_model = SimpleMLP(input_dim)
    ref_model.load_state_dict(global_weights)
    optimizer  = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    criterion  = nn.CrossEntropyLoss(reduction='none')
    generator  = torch.Generator().manual_seed(seed)
    steps = 0
    while steps < target_steps:
        perm = torch.randperm(n_ref, generator=generator)
        for start in range(0, n_ref, batch_size):
            if steps >= target_steps: break
            idx = perm[start:start + batch_size]
            bw  = weights[idx]; bw = bw / (bw.sum() + 1e-8)
            optimizer.zero_grad()
            per_loss = criterion(ref_model(ref_x[idx]), ref_y[idx])
            (per_loss * bw).sum().backward()
            optimizer.step()
            steps += 1

    global_vec = torch.cat([p.data.view(-1) for p in global_weights.values()])
    ref_vec    = torch.cat([p.data.view(-1) for p in ref_model.parameters()])
    return ref_vec - global_vec


def compute_pred_v2_est_signals(client_updates, global_weights, ref_data,
                                 input_dim, seed):
    """PRED-v2-Est: use reference-prediction to estimate pos_rate per client."""
    rank_signals = []
    estimated_rates = []
    for i, u in enumerate(client_updates):
        # Estimate local class balance from client model's predictions on ref set
        pos_est = estimate_pos_rate_from_prediction(
            u, global_weights, ref_data, input_dim)
        estimated_rates.append(pos_est)
        g0_i = compute_iw_g0(global_weights, ref_data, input_dim,
                              target_pos_rate=pos_est,
                              seed=seed + i * 97,
                              target_steps=IW_STEPS, batch_size=32)
        rs = compute_raw_rank_signal(u, g0_i, global_weights)
        rank_signals.append(rs)
    excluded = int(np.argmax(rank_signals))
    return rank_signals, excluded, estimated_rates


def fairness_repair(global_weights, ref_data, input_dim, seed=None):
    if seed is not None: torch.manual_seed(seed)
    model = SimpleMLP(input_dim)
    model.load_state_dict(global_weights)
    ref_x = ref_data['x']; ref_y = ref_data['y']; ref_s = ref_data['sensitive']
    opt = torch.optim.SGD(model.parameters(), lr=REPAIR_LR)
    for _ in range(REPAIR_STEPS):
        opt.zero_grad()
        out = model(ref_x)
        ce  = nn.CrossEntropyLoss()(out, ref_y)
        probs = torch.softmax(out, dim=1)[:, 1]
        p_m = probs[ref_s==1].mean() if (ref_s==1).sum()>0 else probs.mean()
        p_f = probs[ref_s==0].mean() if (ref_s==0).sum()>0 else probs.mean()
        (ce + REPAIR_LAMBDA * torch.abs(p_m - p_f)).backward()
        opt.step()
    return {k: v.clone() for k, v in model.state_dict().items()}


def fedavg_subset(global_weights, all_updates, all_sizes, excluded_idx):
    remaining_updates = [u for i, u in enumerate(all_updates) if i != excluded_idx]
    remaining_sizes   = [s for i, s in enumerate(all_sizes)   if i != excluded_idx]
    total  = sum(remaining_sizes)
    avg    = sum((s/total)*u for u, s in zip(remaining_updates, remaining_sizes))
    offset = 0
    new_weights = {}
    for key, p in global_weights.items():
        numel = p.numel()
        new_weights[key] = p + avg[offset:offset+numel].view_as(p)
        offset += numel
    return new_weights


# ═════════════════════════════════════════════════════════════════════════════
# Main experiment
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PRED-v2-Est: Bias-Estimated Class Balance (Real-World Variant)")
print(f"  Estimator: softmax(out.bias_shift)[1]")
print(f"  IW_STEPS={IW_STEPS} | Repair: {REPAIR_STEPS} steps, lam={REPAIR_LAMBDA}")
print("=" * 65)

est_final   = []
calib_records = []   # true vs estimated pos_rate across all rounds

for seed in SEEDS:
    print(f"\n-- Seed {seed} --", flush=True)
    data_dir = f"data/seed_{seed}"
    prepare_data(seed=seed, n_clients=N_CLIENTS, alpha=0.5,
                 ref_size=250, save_dir=data_dir)

    client_data = [torch.load(f"{data_dir}/client_{i}.pt", weights_only=False)
                   for i in range(N_CLIENTS)]
    ref_data  = torch.load(f"{data_dir}/reference.pt", weights_only=False)
    test_data = torch.load(f"{data_dir}/test.pt",      weights_only=False)
    test_loader = DataLoader(
        TensorDataset(test_data['x'], test_data['y'], test_data['sensitive']),
        batch_size=64, shuffle=False
    )
    input_dim = client_data[0]['x'].shape[1]

    true_rates = [float(client_data[i]['y'].float().mean().item())
                  for i in range(N_CLIENTS)]
    print(f"  True pos rates: " +
          " | ".join([f"C{i}={'[ATT]' if i==ATTACKER else ''}{true_rates[i]:.3f}"
                      for i in range(N_CLIENTS)]), flush=True)

    torch.manual_seed(seed)
    gm = SimpleMLP(input_dim)
    gw = {k: v.clone() for k, v in gm.state_dict().items()}

    c3_correct = 0; honest_false = 0
    round_logs = []

    for r in range(1, ROUNDS + 1):
        updates = [None]*N_CLIENTS; sizes = [None]*N_CLIENTS
        honest_norms = []
        for i in range(N_CLIENTS):
            if i == ATTACKER: continue
            upd, n = train_honest_client(SimpleMLP, input_dim, gw, client_data[i],
                                         batch_size=BATCH, local_epochs=EPOCHS,
                                         lr=LR, seed=seed+r*100+i)
            updates[i] = upd; sizes[i] = n
            honest_norms.append(upd.norm().item())

        med_norm = float(np.median(honest_norms))
        att_upd, att_n = train_malicious_client(
            SimpleMLP, input_dim, gw, client_data[ATTACKER],
            batch_size=BATCH, local_epochs=EPOCHS, lr=LR,
            seed=seed+r*100+ATTACKER, target_norm=med_norm)
        updates[ATTACKER] = att_upd; sizes[ATTACKER] = att_n

        # Detection with ESTIMATED pos_rate
        det_seed = seed + r * 911
        rank_signals, excluded, est_rates = compute_pred_v2_est_signals(
            updates, gw, ref_data, input_dim, seed=det_seed)

        correct = (excluded == ATTACKER)
        if correct: c3_correct   += 1
        else:       honest_false += 1

        # Log calibration (true vs estimated pos_rate)
        for i in range(N_CLIENTS):
            calib_records.append({
                'seed': seed, 'round': r, 'client': i,
                'true_pos_rate': true_rates[i],
                'est_pos_rate':  est_rates[i],
                'error':         est_rates[i] - true_rates[i],
                'is_attacker':   int(i == ATTACKER)
            })

        # Aggregate + repair
        gw = fedavg_subset(gw, updates, sizes, excluded)
        gw = fairness_repair(gw, ref_data, input_dim, seed=seed+r*777)

        gm.load_state_dict(gw)
        metrics = evaluate_model(gm, test_loader)
        round_logs.append({'seed': seed, 'round': r,
                           'excluded_client': excluded,
                           'excluded_correct': int(correct), **metrics})

        if r % 10 == 0:
            est_str = " | ".join([f"C{i}:{est_rates[i]:.2f}" for i in range(N_CLIENTS)])
            print(f"  R{r:02d}: excl=C{excluded} "
                  f"({'OK' if correct else 'MISS'})  "
                  f"DP={metrics['dp_gap']:.4f} | Est: [{est_str}]", flush=True)

    df_seed = pd.DataFrame(round_logs)
    df_seed.to_csv(f"logs/pred_v2_est_seed{seed}.csv", index=False)

    final = round_logs[-1]
    est_final.append({'seed': seed, 'strategy': 'PRED-v2-Est',
                      'accuracy':  final['accuracy'],
                      'dp_gap':    final['dp_gap'],
                      'eo_gap':    final['eo_gap'],
                      'c3_correct': c3_correct,
                      'honest_false': honest_false})
    print(f"  C3 correct: {c3_correct}/30  Honest false: {honest_false}/30  "
          f"DP={final['dp_gap']:.4f}  Acc={final['accuracy']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Calibration analysis
# ─────────────────────────────────────────────────────────────────────────────
df_calib = pd.DataFrame(calib_records)
df_calib.to_csv("logs/pred_v2_est_calibration.csv", index=False)

print()
print("=" * 65)
print("CALIBRATION: True vs Estimated pos_rate")
print("=" * 65)
# Per-seed, per-client: average true and estimated
summary = df_calib.groupby(['seed', 'client']).agg(
    true_mean=('true_pos_rate', 'mean'),
    est_mean=('est_pos_rate',  'mean'),
    error_mae=('error', lambda x: np.abs(x).mean()),
    error_bias=('error', 'mean')
).reset_index()

for seed in SEEDS:
    sub = summary[summary['seed'] == seed]
    true_rates_here = [float(torch.load(f"data/seed_{seed}/client_{i}.pt",
                             weights_only=False)['y'].float().mean().item())
                       for i in range(N_CLIENTS)]
    print(f"\n  Seed {seed}:")
    print(f"  {'Client':>8}  {'True pos':>10}  {'Est mean':>10}  "
          f"{'MAE':>8}  {'Bias':>8}  {'Attacker?':>10}")
    for _, row in sub.iterrows():
        i = int(row['client'])
        att = "ATTACKER" if i == ATTACKER else "honest"
        print(f"  {'C'+str(i):>8}  {row['true_mean']:>10.4f}  "
              f"{row['est_mean']:>10.4f}  {row['error_mae']:>8.4f}  "
              f"{row['error_bias']:>+8.4f}  {att:>10}")

# Correlation: true vs estimated across all clients/rounds
r_pearson, p_pearson = pearsonr(df_calib['true_pos_rate'], df_calib['est_pos_rate'])
r_spear,   p_spear   = spearmanr(df_calib['true_pos_rate'], df_calib['est_pos_rate'])
mae_all = np.abs(df_calib['error']).mean()

print(f"\n  Overall:")
print(f"    Pearson r  = {r_pearson:.4f}  (p={p_pearson:.4e})")
print(f"    Spearman r = {r_spear:.4f}  (p={p_spear:.4e})")
print(f"    MAE        = {mae_all:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Compare detection accuracy: PRED-v2 (ground truth) vs PRED-v2-Est (estimated)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("DETECTION ACCURACY: PRED-v2 (True) vs PRED-v2-Est (Estimated)")
print("=" * 65)

# Load PRED-v2 ground truth
v2_gt = {}
for seed in SEEDS:
    p = f"logs/pred_v2_seed{seed}.csv"
    if os.path.exists(p):
        df_p = pd.read_csv(p)
        c3_excl = df_p['excluded_correct'].sum()
        final = df_p.iloc[-1]
        v2_gt[seed] = {'c3_correct': c3_excl,
                       'accuracy': final['accuracy'],
                       'dp_gap': final['dp_gap'],
                       'eo_gap': final['eo_gap']}

print(f"\n  {'Seed':>5}  {'GT C3/30':>10}  {'Est C3/30':>10}  "
      f"{'GT DP':>8}  {'Est DP':>8}  {'GT Acc':>8}  {'Est Acc':>8}")
print(f"  {'─'*70}")
for row in est_final:
    s = row['seed']
    gt = v2_gt.get(s, {})
    print(f"  {s:>5}  {gt.get('c3_correct','?'):>10}  {row['c3_correct']:>10}  "
          f"{gt.get('dp_gap',0):>8.4f}  {row['dp_gap']:>8.4f}  "
          f"{gt.get('accuracy',0):>8.4f}  {row['accuracy']:>8.4f}")

gt_total  = sum(v2_gt[s]['c3_correct'] for s in SEEDS if s in v2_gt)
est_total = sum(r['c3_correct'] for r in est_final)
gt_false  = 150 - gt_total
est_false = sum(r['honest_false'] for r in est_final)

print(f"\n  {'POOLED':>5}  {gt_total:>10}  {est_total:>10}  "
      f"{np.mean([v2_gt[s]['dp_gap'] for s in SEEDS if s in v2_gt]):>8.4f}  "
      f"{np.mean([r['dp_gap'] for r in est_final]):>8.4f}  "
      f"{np.mean([v2_gt[s]['accuracy'] for s in SEEDS if s in v2_gt]):>8.4f}  "
      f"{np.mean([r['accuracy'] for r in est_final]):>8.4f}")
print(f"\n  Ground truth PRED-v2: {gt_total}/150 ({gt_total/150*100:.1f}%) correct")
print(f"  Estimated  PRED-v2-Est: {est_total}/150 ({est_total/150*100:.1f}%) correct")
print(f"  Detection loss from estimation: {gt_total-est_total} rounds "
      f"({(gt_total-est_total)/150*100:.1f}pp)")

# ─────────────────────────────────────────────────────────────────────────────
# Calibration plot + detection comparison plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6))
fig.suptitle("PRED-v2-Est: Bias-Estimated Class Balance vs Ground Truth",
             fontsize=12, fontweight='bold')

# Plot 1: Calibration scatter (true vs estimated)
ax0 = axes[0]
for seed in SEEDS:
    sub = df_calib[df_calib['seed'] == seed]
    colors_pts = ['#e74c3c' if r==1 else '#3498db'
                  for r in sub['is_attacker']]
    ax0.scatter(sub['true_pos_rate'], sub['est_pos_rate'],
                c=colors_pts, alpha=0.15, s=8)
ax0.plot([0,1],[0,1],'k--', lw=1.5, label='Perfect calibration')
ax0.set_xlabel("True positive rate", fontsize=10)
ax0.set_ylabel("Estimated positive rate (bias shift)", fontsize=10)
ax0.set_title(f"Calibration: True vs Estimated\n"
              f"Pearson r={r_pearson:.3f}, MAE={mae_all:.3f}", fontsize=10)
ax0.legend(fontsize=9)
ax0.grid(True, linestyle='--', alpha=0.4)

from matplotlib.lines import Line2D
handles = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#e74c3c',
                   markersize=8,label='Attacker C3'),
           Line2D([0],[0],marker='o',color='w',markerfacecolor='#3498db',
                   markersize=8,label='Honest clients')]
ax0.legend(handles=handles, fontsize=9)

# Plot 2: Detection accuracy bar chart
ax1 = axes[1]
x  = np.arange(len(SEEDS))
bw = 0.35
gt_bars  = [v2_gt[s]['c3_correct'] for s in SEEDS if s in v2_gt]
est_bars = [r['c3_correct'] for r in est_final]

ax1.bar(x - bw/2, gt_bars,  bw, label='PRED-v2 (true class balance)',
        color='#27ae60', alpha=0.8, edgecolor='black', lw=0.8)
ax1.bar(x + bw/2, est_bars, bw, label='PRED-v2-Est (estimated from bias)',
        color='#e67e22', alpha=0.8, edgecolor='black', lw=0.8)
for i, (g, e) in enumerate(zip(gt_bars, est_bars)):
    ax1.text(x[i]-bw/2, g+0.2, str(g), ha='center', va='bottom', fontsize=9)
    ax1.text(x[i]+bw/2, e+0.2, str(e), ha='center', va='bottom', fontsize=9)

ax1.set_xticks(x)
ax1.set_xticklabels([f"Seed {s}" for s in SEEDS])
ax1.set_ylabel("C3 correctly excluded (out of 30)")
ax1.set_ylim(0, 33)
ax1.set_title(f"Detection Accuracy\n"
              f"GT={gt_total}/150 ({gt_total/150*100:.1f}%) | "
              f"Est={est_total}/150 ({est_total/150*100:.1f}%)",
              fontsize=10)
ax1.axhline(6, color='navy', ls='--', lw=1.2, label='20% chance baseline')
ax1.legend(fontsize=9)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# Plot 3: Final DP comparison
ax2 = axes[2]
gt_dp  = [v2_gt[s]['dp_gap'] for s in SEEDS if s in v2_gt]
est_dp = [r['dp_gap'] for r in est_final]

ax2.bar(x - bw/2, gt_dp,  bw, label='PRED-v2 (true)',
        color='#27ae60', alpha=0.8, edgecolor='black', lw=0.8)
ax2.bar(x + bw/2, est_dp, bw, label='PRED-v2-Est (estimated)',
        color='#e67e22', alpha=0.8, edgecolor='black', lw=0.8)
for i, (g, e) in enumerate(zip(gt_dp, est_dp)):
    ax2.text(x[i]-bw/2, g+0.002, f"{g:.3f}", ha='center', va='bottom', fontsize=8)
    ax2.text(x[i]+bw/2, e+0.002, f"{e:.3f}", ha='center', va='bottom', fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels([f"Seed {s}" for s in SEEDS])
ax2.set_ylabel("Final ΔDP (lower = better)")
ax2.set_title(f"Final Fairness Gap\n"
              f"GT mean={np.mean(gt_dp):.4f} | Est mean={np.mean(est_dp):.4f}",
              fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("results/pred_v2_est_comparison.png", dpi=180, bbox_inches='tight')
plt.close()
print("\nSaved results/pred_v2_est_comparison.png")
print("\nALL DONE")
