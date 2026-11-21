"""
Per-Round Rank-Exclusion Detector (PRED)
=========================================
Signal: Rank-residual — stable rank of client's update residual after projecting
        onto FLTrust reference g0 (from defenses/signal.py / Phase 8-9 detector).
        Higher rank-residual = more suspicious (update has more low-rank structure
        NOT aligned with the honest reference direction).
Rule:   Each round, exclude the SINGLE client with the HIGHEST rank-residual signal.
        Average the remaining 4 clients with standard FedAvg (re-weighted by size).
Memory: NONE — g0 is recomputed fresh each round, every decision is independent.
        All 5 clients receive the updated global model each round regardless.

NOTE: plain cosine-similarity (Phase 7 gate) is defeated by the norm-matched attacker
      (attacker achieves MAXIMUM cos_sim by construction). The rank-residual from
      signal.py was the original detector that gave 30/30 and 80/80 correct rounds.

Comparison: loads cached per-round results for FedAvg / KRUM / FLTrust from
            logs/sweep_*_seed*.csv so no re-training of baselines is needed.

Outputs:
  logs/pred_seed*.csv                  -- per-round exclusion log per seed
  logs/pred_final_results.csv          -- final round metrics
  results/pred_exclusion_accuracy.png  -- per-seed bar chart (C3 excl. rate)
  results/pred_vs_baselines.png        -- fairness/accuracy comparison + sign test
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from defenses.aggregators import fedavg_aggregate, fltrust_aggregate
from defenses.signal import compute_raw_rank_signal
from defenses.metrics import evaluate_model

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS     = [42, 43, 44, 45, 46]
N_CLIENTS = 5
ATTACKER  = 3
ROUNDS    = 30
LR        = 0.01
BATCH     = 32
EPOCHS    = 5
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PRED Signal: rank-residual (the original working detector from signal.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_pred_signals(client_updates, global_weights, ref_data,
                         input_dim, seed):
    """
    Per-round rank-residual signal (Eqs from signal.py / Phase 8-9):
      1. Get reference update g0 via FLTrust server training on ref_data (200 steps).
      2. For each client: rank_signal = compute_raw_rank_signal(client_i, g0, W).
         Higher rank_signal = more suspicious (attacker's residual is less rank-1).
      3. Most suspicious = argmax(rank_signals).

    This is exactly the signal that achieved 30/30 and 80/80 detection rounds
    in the earlier tests — memoryless, fresh g0 each round.
    """
    # Fresh FLTrust g0 computation — fast (200 steps on 250 reference samples)
    _, _, g0, _ = fltrust_aggregate(
        global_weights, client_updates, [1]*len(client_updates),
        ref_data, SimpleMLP, input_dim,
        batch_size=32, target_steps=200, seed=seed
    )

    rank_signals = []
    for u in client_updates:
        rs = compute_raw_rank_signal(u, g0, global_weights)
        rank_signals.append(rs)

    excluded = int(np.argmax(rank_signals))   # most suspicious = highest rank-residual
    return rank_signals, excluded


def pred_aggregate(global_weights, client_updates, client_sizes, excluded_idx):
    """
    FedAvg over the 4 remaining clients (excluding excluded_idx).
    """
    remaining_updates = [u for i, u in enumerate(client_updates) if i != excluded_idx]
    remaining_sizes   = [s for i, s in enumerate(client_sizes)   if i != excluded_idx]
    return fedavg_aggregate(global_weights, remaining_updates, remaining_sizes)

# ─────────────────────────────────────────────────────────────────────────────
# Main PRED experiment
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("PER-ROUND RANK-EXCLUSION DETECTOR (PRED)")
print("Signal: rank-residual vs FLTrust g0 | No memory | Fresh decision each round")
print("="*65)

pred_final = []    # final-round metrics per seed
pred_round_logs = []

for seed in SEEDS:
    print(f"\n── Seed {seed} ──", flush=True)
    data_dir = f"data/seed_{seed}"
    prepare_data(seed=seed, n_clients=N_CLIENTS, alpha=0.5,
                 ref_size=250, save_dir=data_dir)

    client_data = [
        torch.load(f"{data_dir}/client_{i}.pt", weights_only=False)
        for i in range(N_CLIENTS)
    ]
    ref_data  = torch.load(f"{data_dir}/reference.pt", weights_only=False)
    test_data = torch.load(f"{data_dir}/test.pt", weights_only=False)
    test_loader = DataLoader(
        TensorDataset(test_data['x'], test_data['y'], test_data['sensitive']),
        batch_size=64, shuffle=False
    )
    input_dim = client_data[0]['x'].shape[1]

    torch.manual_seed(seed)
    global_model   = SimpleMLP(input_dim)
    global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

    c3_excl_count    = 0
    honest_excl_count = 0
    round_logs        = []

    for r in range(1, ROUNDS + 1):
        # ── Train all 5 clients ──────────────────────────────────────────────
        honest_norms = []
        updates = [None] * N_CLIENTS
        sizes   = [None] * N_CLIENTS

        # First pass: train honest clients to get median norm
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

        # Train attacker with norm-matching
        att_seed = seed + r * 100 + ATTACKER
        att_upd, att_n = train_malicious_client(
            SimpleMLP, input_dim, global_weights, client_data[ATTACKER],
            batch_size=BATCH, local_epochs=EPOCHS, lr=LR,
            seed=att_seed, target_norm=med_norm
        )
        updates[ATTACKER] = att_upd
        sizes[ATTACKER]   = att_n

        # ── Per-round signal & exclusion decision (rank-residual) ─────────────
        ref_seed = seed + r * 999
        rank_signals, excluded = compute_pred_signals(
            updates, global_weights, ref_data, input_dim, seed=ref_seed
        )

        arr = np.array(rank_signals)
        mu, sig = arr.mean(), arr.std() + 1e-8
        z_scores = ((arr - mu) / sig).tolist()  # for logging

        correct = (excluded == ATTACKER)
        if correct:
            c3_excl_count += 1
        else:
            honest_excl_count += 1

        # ── Aggregate remaining 4 clients ────────────────────────────────────
        new_weights, _ = pred_aggregate(global_weights, updates, sizes, excluded)
        global_weights = new_weights
        global_model.load_state_dict(global_weights)

        # ── Evaluate ─────────────────────────────────────────────────────────
        metrics = evaluate_model(global_model, test_loader)
        round_logs.append({
            'seed': seed, 'round': r,
            'excluded_client': excluded,
            'excluded_correct': int(correct),
            'rank_signal_c3': rank_signals[ATTACKER],
            'z_score_c3': z_scores[ATTACKER],
            **metrics
        })

    # Per-round log
    df_seed = pd.DataFrame(round_logs)
    df_seed.to_csv(f"logs/pred_seed{seed}.csv", index=False)
    pred_round_logs.append(df_seed)

    # Final round metrics
    final = round_logs[-1]
    pred_final.append({
        'strategy': 'PRED', 'seed': seed,
        'accuracy': final['accuracy'],
        'dp_gap': final['dp_gap'],
        'eo_gap': final['eo_gap'],
        'c3_excl_rounds': c3_excl_count,
        'honest_excl_rounds': honest_excl_count,
    })

    print(f"  C3 correctly excluded: {c3_excl_count}/30 rounds  "
          f"| Honest falsely excluded: {honest_excl_count}/30 rounds")
    print(f"  Final → Acc={final['accuracy']:.4f}  "
          f"ΔDP={final['dp_gap']:.4f}  ΔEO={final['eo_gap']:.4f}")

df_pred_final = pd.DataFrame(pred_final)
df_pred_final.to_csv("logs/pred_final_results.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Load baseline results from cached sweep logs
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Loading cached baseline results ──")
baselines = {}
for strategy in ['fedavg', 'krum', 'fltrust']:
    rows = []
    for seed in SEEDS:
        path = f"logs/sweep_{strategy}_seed{seed}.csv"
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping")
            continue
        df_b = pd.read_csv(path)
        # Get final round
        final_row = df_b.iloc[-1]
        rows.append({
            'strategy': strategy.upper(), 'seed': seed,
            'accuracy': final_row['accuracy'],
            'dp_gap':   final_row['dp_gap'],
            'eo_gap':   final_row['eo_gap'],
        })
    if rows:
        baselines[strategy.upper()] = pd.DataFrame(rows)
        print(f"  {strategy.upper()}: loaded {len(rows)} seeds")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("EXCLUSION ACCURACY PER SEED")
print("="*65)
for row in pred_final:
    print(f"  Seed {row['seed']}: C3 excl. {row['c3_excl_rounds']:2d}/30 rounds  "
          f"| Honest excl. {row['honest_excl_rounds']:2d}/30 rounds")

total_c3   = sum(r['c3_excl_rounds']   for r in pred_final)
total_hon  = sum(r['honest_excl_rounds'] for r in pred_final)
print(f"\n  POOLED (5 seeds × 30 rounds = 150 total):")
print(f"  C3 correctly excluded: {total_c3}/150  ({total_c3/150*100:.1f}%)")
print(f"  Honest falsely excl. : {total_hon}/150  ({total_hon/150*100:.1f}%)")

# Binomial test: is C3 excl. rate significantly above 20% chance?
binom_p = binomtest(total_c3, 150, p=0.2, alternative='greater').pvalue
print(f"  Binomial test (vs 20% chance): p = {binom_p:.4f}")

print("\n" + "="*65)
print("FINAL ROUND METRICS — ALL STRATEGIES")
print("="*65)
all_rows = [{'strategy': 'PRED', 'seed': r['seed'],
             'accuracy': r['accuracy'], 'dp_gap': r['dp_gap'],
             'eo_gap': r['eo_gap']} for r in pred_final]
for df_b in baselines.values():
    all_rows.extend(df_b.to_dict('records'))

df_all = pd.DataFrame(all_rows)
for strat in ['PRED', 'FEDAVG', 'KRUM', 'FLTRUST']:
    sub = df_all[df_all['strategy'] == strat]
    if sub.empty: continue
    print(f"  {strat:8s}: Acc={sub['accuracy'].mean():.4f}±{sub['accuracy'].std():.4f}  "
          f"ΔDP={sub['dp_gap'].mean():.4f}±{sub['dp_gap'].std():.4f}  "
          f"ΔEO={sub['eo_gap'].mean():.4f}±{sub['eo_gap'].std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Paired sign tests (same-seed: PRED vs each baseline on ΔDP)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PAIRED SIGN TESTS (PRED vs Baselines on ΔDP)")
print("  H1: PRED achieves lower ΔDP than baseline (one-sided, per-seed sign)")
print("="*65)

pred_dp = {r['seed']: r['dp_gap'] for r in pred_final}

for baseline_name, df_b in baselines.items():
    wins, ties, losses = 0, 0, 0
    pairs = []
    for _, row in df_b.iterrows():
        seed = row['seed']
        if seed not in pred_dp: continue
        diff = pred_dp[seed] - row['dp_gap']
        pairs.append(diff)
        if diff < -1e-6:  wins   += 1   # PRED lower ΔDP = better
        elif diff > 1e-6: losses += 1
        else:             ties   += 1

    n_effective = wins + losses
    if n_effective > 0:
        p_val = binomtest(wins, n_effective, p=0.5, alternative='less').pvalue
    else:
        p_val = 1.0
    sig = " ***" if p_val < 0.01 else (" **" if p_val < 0.05 else (" *" if p_val < 0.1 else ""))
    print(f"\n  PRED vs {baseline_name}:")
    print(f"    Per-seed ΔDP diffs (PRED − baseline): {[f'{v:+.4f}' for v in pairs]}")
    print(f"    PRED better (lower ΔDP): {wins} | Tied: {ties} | PRED worse: {losses}")
    print(f"    Sign test p-value (H1: PRED lower): p={p_val:.4f}{sig}")
    mean_diff = np.mean(pairs) if pairs else 0.0
    print(f"    Mean ΔDP diff: {mean_diff:+.4f} ({'PRED reduces ΔDP' if mean_diff < 0 else 'PRED increases ΔDP'})")

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
COLORS_STRAT = {'PRED':'#e74c3c', 'FEDAVG':'#95a5a6',
                'KRUM':'#3498db', 'FLTRUST':'#27ae60'}

# Plot 1: Per-seed exclusion accuracy
fig1, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(SEEDS))
bw = 0.35
c3_counts     = [r['c3_excl_rounds']    for r in pred_final]
honest_counts = [r['honest_excl_rounds'] for r in pred_final]

ax1.bar(x - bw/2, c3_counts,     bw, label='C3(A) correctly excluded',
        color='#e74c3c', alpha=0.82, edgecolor='black', linewidth=0.7)
ax1.bar(x + bw/2, honest_counts, bw, label='Honest client falsely excluded',
        color='#95a5a6', alpha=0.82, edgecolor='black', linewidth=0.7)
ax1.axhline(30 * 0.2, color='navy', linestyle='--', linewidth=1.5,
            label='Chance baseline (20% = 6 rounds)')

for i, (c, h) in enumerate(zip(c3_counts, honest_counts)):
    ax1.text(x[i] - bw/2, c + 0.3, str(c), ha='center', va='bottom',
             fontsize=11, fontweight='bold', color='#c0392b')
    ax1.text(x[i] + bw/2, h + 0.3, str(h), ha='center', va='bottom',
             fontsize=11, color='#555')

ax1.set_xticks(x)
ax1.set_xticklabels([f"Seed {s}" for s in SEEDS], fontsize=11)
ax1.set_ylabel("Rounds excluded (out of 30)", fontsize=11)
ax1.set_ylim(0, 33)
ax1.set_title(
    f"PRED: Per-Seed Exclusion Accuracy\n"
    f"Pooled: C3 excl. {total_c3}/150 ({total_c3/150*100:.1f}%)  "
    f"| Honest excl. {total_hon}/150 ({total_hon/150*100:.1f}%)  "
    f"| Chance=20%  | binom p={binom_p:.3f}",
    fontsize=11, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("results/pred_exclusion_accuracy.png", dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved results/pred_exclusion_accuracy.png")

# Plot 2: ΔDP and Accuracy comparison across all strategies
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6))
strategies = ['PRED', 'FEDAVG', 'KRUM', 'FLTRUST']
metrics    = [('dp_gap', 'ΔDP (Demographic Parity Gap)', True),
              ('eo_gap', 'ΔEO (Equalized Odds Gap)',     True),
              ('accuracy','Accuracy',                    False)]

for ax, (metric, ylabel, lower_better) in zip(axes2, metrics):
    positions = []
    for j, strat in enumerate(strategies):
        sub = df_all[df_all['strategy'] == strat]
        if sub.empty: continue
        vals = sub[metric].values
        bp   = ax.boxplot(
            vals, positions=[j], widths=0.5, patch_artist=True, notch=False,
            medianprops=dict(color='black', linewidth=2.0),
            flierprops=dict(marker='o', markersize=3, alpha=0.4),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2)
        )
        bp['boxes'][0].set_facecolor(COLORS_STRAT.get(strat, '#888'))
        bp['boxes'][0].set_alpha(0.78)
        positions.append(j)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    better = "lower=better" if lower_better else "higher=better"
    ax.set_title(f"{ylabel}\n({better})", fontsize=10, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Annotate mean
    for j, strat in enumerate(strategies):
        sub = df_all[df_all['strategy'] == strat]
        if sub.empty: continue
        mean_val = sub[metric].mean()
        ax.text(j, ax.get_ylim()[0] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.07,
                f"{mean_val:.3f}", ha='center', va='top', fontsize=8, color='#333')

legend_patches = [plt.Rectangle((0,0),1,1,fc=COLORS_STRAT.get(s,'#888'),
                                 alpha=0.78, label=s) for s in strategies]
fig2.legend(handles=legend_patches, loc='lower center', ncol=4,
            fontsize=10, bbox_to_anchor=(0.5, 0.01))
fig2.suptitle(
    "PRED vs FedAvg / KRUM / FLTrust — Final Round Metrics\n"
    "(5 seeds × 30 rounds | Attacker=C3 | norm-matched LoRA attack)",
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("results/pred_vs_baselines.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved results/pred_vs_baselines.png")

print("\n" + "="*65)
print("ALL DONE")
print("="*65)
