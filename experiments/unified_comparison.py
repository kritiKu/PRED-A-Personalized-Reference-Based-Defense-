"""
Unified Comparison: N=5 vs N=10 vs N=20 (seeds 43-46 only)
Detection accuracy, fairness metrics, paired sign tests + magnitude reports
"""
import os, sys, ast
import numpy as np
import pandas as pd
from scipy.stats import binomtest

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

SEEDS = [43, 44, 45, 46, 47]
ROUNDS = 30
CONFIGS = {
    5:  {'attackers': [3],           'honest': [0,1,2,4]},
    10: {'attackers': [3, 8],        'honest': [0,1,2,4,5,6,7,9]},
    20: {'attackers': [3, 8, 13, 18],'honest': [i for i in range(20) if i not in [3,8,13,18]]},
}
STRATEGIES = ['fedavg', 'krum', 'fltrust', 'pred-v2']
STRAT_LABELS = {'fedavg':'FedAvg', 'krum':'KRUM', 'fltrust':'FLTrust', 'pred-v2':'PRED-v2'}
METRICS = [('accuracy','Accuracy',+1), ('dp_gap','Delta-DP',-1), ('eo_gap','Delta-EO',-1)]

def load_all():
    data = {}
    for nc in [5, 10, 20]:
        tag = f"unified_n{nc}"
        data[nc] = {}
        for strat in STRATEGIES:
            data[nc][strat] = {}
            for s in SEEDS:
                path = f"logs/{tag}/{strat}_seed{s}.csv"
                if os.path.exists(path):
                    data[nc][strat][s] = pd.read_csv(path)
    return data

data = load_all()

for nc in [5, 10, 20]:
    attackers = CONFIGS[nc]['attackers']
    honest = CONFIGS[nc]['honest']
    n_att = len(attackers)

    print(f"\n{'#'*75}")
    print(f"#  N = {nc} CLIENTS  |  {n_att} ATTACKERS ({attackers})  |  Seeds {SEEDS}")
    print(f"{'#'*75}")

    # ── Detection ──────────────────────────────────────────────────────
    print(f"\n  DETECTION ACCURACY (PRED-v2, Dynamic MAD Z>3.0)")
    print(f"  {'Seed':>5}", end="")
    for a in attackers:
        print(f"  {'C'+str(a)+' excl':>10}", end="")
    print(f"  {'Honest FP':>12}  {'Total TP':>10}")
    print(f"  {'─'*(15 + 12*n_att + 24)}")

    total_tp = 0; total_fp = 0; total_possible = 0
    per_seed_tp = {}
    for s in SEEDS:
        if s not in data[nc]['pred-v2']: continue
        df = data[nc]['pred-v2'][s]
        att_excl = {a: 0 for a in attackers}
        fp = 0
        for _, row in df.iterrows():
            excl = ast.literal_eval(row['excluded'])
            for a in attackers:
                if a in excl: att_excl[a] += 1
            fp += sum(1 for c in excl if c in honest)
        tp = sum(att_excl.values())
        total_tp += tp; total_fp += fp
        total_possible += ROUNDS * n_att
        per_seed_tp[s] = tp

        print(f"  {s:>5}", end="")
        for a in attackers:
            print(f"  {att_excl[a]:>7}/30", end="")
        print(f"  {fp:>12}  {tp:>7}/{ROUNDS*n_att}")

    det_rate = total_tp / total_possible * 100 if total_possible > 0 else 0
    print(f"  {'─'*(15 + 12*n_att + 24)}")
    print(f"  POOLED: {total_tp}/{total_possible} ({det_rate:.1f}%)  |  FP: {total_fp}")

    # ── Final Metrics ──────────────────────────────────────────────────
    print(f"\n  FINAL METRICS (round 30)")
    for metric_key, metric_label, better_dir in METRICS:
        print(f"\n  [{metric_label}]  ({'higher' if better_dir>0 else 'lower'} is better)")
        print(f"  {'Seed':>5}", end="")
        for strat in STRATEGIES:
            print(f"  {STRAT_LABELS[strat]:>10}", end="")
        print()
        print(f"  {'─'*55}")

        for s in SEEDS:
            print(f"  {s:>5}", end="")
            for strat in STRATEGIES:
                if s in data[nc][strat]:
                    val = data[nc][strat][s].iloc[-1][metric_key]
                    print(f"  {val:>10.4f}", end="")
                else:
                    print(f"  {'N/A':>10}", end="")
            print()

        print(f"  {'─'*55}")
        print(f"  {'Mean':>5}", end="")
        for strat in STRATEGIES:
            vals = [data[nc][strat][s].iloc[-1][metric_key]
                    for s in SEEDS if s in data[nc][strat]]
            print(f"  {np.mean(vals):>10.4f}" if vals else f"  {'N/A':>10}", end="")
        print()
        print(f"  {'Std':>5}", end="")
        for strat in STRATEGIES:
            vals = [data[nc][strat][s].iloc[-1][metric_key]
                    for s in SEEDS if s in data[nc][strat]]
            print(f"  {np.std(vals):>10.4f}" if vals else f"  {'N/A':>10}", end="")
        print()

    # ── Sign Tests ─────────────────────────────────────────────────────
    print(f"\n  PAIRED SIGN TESTS (PRED-v2 vs each baseline) + MAGNITUDES")
    for metric_key, metric_label, better_dir in METRICS:
        print(f"\n  --- {metric_label} ---")
        for baseline in ['fedavg', 'krum', 'fltrust']:
            diffs = []
            for s in SEEDS:
                if s in data[nc]['pred-v2'] and s in data[nc][baseline]:
                    pv = data[nc]['pred-v2'][s].iloc[-1][metric_key]
                    bv = data[nc][baseline][s].iloc[-1][metric_key]
                    diffs.append(pv - bv)

            if not diffs: continue
            pred_wins = sum(1 for d in diffs if d * better_dir > 1e-6)
            base_wins = sum(1 for d in diffs if d * better_dir < -1e-6)
            tied = len(diffs) - pred_wins - base_wins
            ne = pred_wins + base_wins
            p2 = min(1.0, 2*binomtest(min(pred_wins,base_wins), ne, 0.5,
                                       alternative='less').pvalue) if ne>0 else 1.0

            print(f"  vs {STRAT_LABELS[baseline]:<8}: ", end="")
            print(f"PRED wins {pred_wins}/{len(diffs)} | "
                  f"Base wins {base_wins}/{len(diffs)} | "
                  f"Tied {tied} | p={p2:.4f}")
            print(f"             Diffs: " +
                  ", ".join([f"s{SEEDS[i]}:{diffs[i]:+.4f}" for i in range(len(diffs))]))
            max_mag = max(abs(d) for d in diffs)
            driving = SEEDS[[abs(d) for d in diffs].index(max_mag)]
            print(f"             Largest gap: seed {driving} = {max_mag:.4f}")

# ── Cross-Scale Summary ───────────────────────────────────────────────
print(f"\n{'#'*75}")
print(f"#  CROSS-SCALE SUMMARY (Seeds 43-46, no seed 42)")
print(f"{'#'*75}")
print(f"\n  {'N':>4}  {'Attackers':>10}  {'Det %':>8}  {'FP':>5}  "
      f"{'DP mean':>8}  {'DP std':>8}  {'EO mean':>8}  {'Acc mean':>9}")
print(f"  {'─'*75}")

for nc in [5, 10, 20]:
    attackers = CONFIGS[nc]['attackers']
    honest = CONFIGS[nc]['honest']
    tp = 0; fp = 0; total = 0
    dp_vals = []; eo_vals = []; acc_vals = []
    for s in SEEDS:
        if s not in data[nc]['pred-v2']: continue
        df = data[nc]['pred-v2'][s]
        for _, row in df.iterrows():
            excl = ast.literal_eval(row['excluded'])
            tp += sum(1 for a in attackers if a in excl)
            fp += sum(1 for c in excl if c in honest)
        total += ROUNDS * len(attackers)
        final = df.iloc[-1]
        dp_vals.append(final['dp_gap'])
        eo_vals.append(final['eo_gap'])
        acc_vals.append(final['accuracy'])

    det = tp/total*100 if total>0 else 0
    print(f"  {nc:>4}  {str(attackers):>10}  {det:>7.1f}%  {fp:>5}  "
          f"{np.mean(dp_vals):>8.4f}  {np.std(dp_vals):>8.4f}  "
          f"{np.mean(eo_vals):>8.4f}  {np.mean(acc_vals):>9.4f}")

# Also show baselines at each scale for context
print(f"\n  Baseline FedAvg comparison:")
print(f"  {'N':>4}  {'DP mean':>8}  {'DP std':>8}  {'EO mean':>8}  {'Acc mean':>9}")
print(f"  {'─'*45}")
for nc in [5, 10, 20]:
    dp_vals = []; eo_vals = []; acc_vals = []
    for s in SEEDS:
        if s not in data[nc]['fedavg']: continue
        final = data[nc]['fedavg'][s].iloc[-1]
        dp_vals.append(final['dp_gap'])
        eo_vals.append(final['eo_gap'])
        acc_vals.append(final['accuracy'])
    if dp_vals:
        print(f"  {nc:>4}  {np.mean(dp_vals):>8.4f}  {np.std(dp_vals):>8.4f}  "
              f"{np.mean(eo_vals):>8.4f}  {np.mean(acc_vals):>9.4f}")

print("\nALL DONE")
