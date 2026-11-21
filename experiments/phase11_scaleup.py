import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(8)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.runner import run_fl_experiment

# Phase 11 sweep settings
SEEDS      = [42, 43, 44]          # 3 seeds each (enough for significance)
STRATEGIES = ['fedavg', 'krum', 'fltrust']
N_CLIENTS  = [10, 20]

def run_phase11_scaleup():
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs",    exist_ok=True)

    print("\n=======================================================")
    print("PHASE 11: SCALE-UP EXPERIMENT  (n=10 and n=20 clients)")
    print("=======================================================")
    print("Gate FAILED → testing whether separation improves at larger scale")

    all_results = []

    for n in N_CLIENTS:
        print(f"\n{'='*55}")
        print(f"  N_CLIENTS = {n}")
        print(f"{'='*55}")

        for strat in STRATEGIES:
            for seed in SEEDS:
                log_path = f"logs/scaleup_n{n}_{strat}_seed{seed}.csv"

                if os.path.exists(log_path):
                    print(f"---> SKIP  n={n} | {strat.upper()} | seed {seed} (cached)")
                    df = pd.read_csv(log_path)
                    last = df.iloc[-1]
                    all_results.append({
                        'n_clients': n, 'strategy': strat, 'seed': seed,
                        'accuracy': float(last['accuracy']),
                        'dp_gap':   float(last['dp_gap']),
                        'eo_gap':   float(last['eo_gap'])
                    })
                    continue

                print(f"\n---> n={n} | {strat.upper()} | seed {seed}")
                data_dir = f"data/n{n}_seed_{seed}"

                df_logs, final = run_fl_experiment(
                    strategy=strat, attack=True, seed=seed, rounds=30,
                    batch_size=32, local_epochs=5, lr=0.01,
                    log_path=log_path,
                    data_dir=data_dir,
                    n_clients=n,
                    attacker_id=3
                )
                all_results.append({
                    'n_clients': n, 'strategy': strat, 'seed': seed,
                    'accuracy': final['accuracy'],
                    'dp_gap':   final['dp_gap'],
                    'eo_gap':   final['eo_gap']
                })
                print(f"    Done: Acc={final['accuracy']:.4f} | "
                      f"dDP={final['dp_gap']:.4f} | dEO={final['eo_gap']:.4f}")

    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv("results/scaleup_raw.csv", index=False)

    # ── Summary table ────────────────────────────────────────────────────────
    summary_rows = []
    for n in N_CLIENTS:
        for strat in STRATEGIES:
            sub = df_raw[(df_raw['n_clients'] == n) & (df_raw['strategy'] == strat)]
            if sub.empty:
                continue
            summary_rows.append({
                'N': n,
                'Strategy': strat.upper(),
                'Accuracy': f"{sub['accuracy'].mean():.4f}+/-{sub['accuracy'].std():.4f}",
                'dDP':      f"{sub['dp_gap'].mean():.4f}+/-{sub['dp_gap'].std():.4f}",
                'dEO':      f"{sub['eo_gap'].mean():.4f}+/-{sub['eo_gap'].std():.4f}",
                'dp_mean':  sub['dp_gap'].mean(),
                'dp_std':   sub['dp_gap'].std(),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/scaleup_comparison.csv", index=False)

    print("\n=======================================================")
    print("PHASE 11 SCALE-UP COMPARISON RESULTS")
    print("=======================================================")
    print(df_summary[['N', 'Strategy', 'Accuracy', 'dDP', 'dEO']].to_string(index=False))

    # ── Bar plot: dDP by N and Strategy ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = {'FEDAVG': '#3498db', 'KRUM': '#e67e22', 'FLTRUST': '#e74c3c'}

    for ax, n in zip(axes, N_CLIENTS):
        sub = df_summary[df_summary['N'] == n]
        strategies = sub['Strategy'].tolist()
        dp_means   = sub['dp_mean'].tolist()
        dp_stds    = sub['dp_std'].tolist()
        x = range(len(strategies))

        bars = ax.bar(x, dp_means, yerr=dp_stds, capsize=5, width=0.5,
                      color=[colors.get(s, '#95a5a6') for s in strategies],
                      edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(strategies, fontsize=11)
        ax.set_title(f"n = {n} Clients", fontsize=12, fontweight='bold')
        ax.set_ylabel("Demographic Parity Gap (dDP)" if n == N_CLIENTS[0] else "")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Value labels on bars
        for bar, mean, std in zip(bars, dp_means, dp_stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                    f"{mean:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle(
        "Phase 11 Scale-Up: Fairness Attack Impact (dDP) under n=10 and n=20 Clients\n"
        "(Higher dDP = Worse Fairness, Attack More Effective)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("results/phase11_scaleup_plot.png", dpi=200)
    plt.close()
    print("Saved bar plot to results/phase11_scaleup_plot.png")

    # Compare n=5 (Phase 6) baseline vs n=10, n=20
    try:
        df_n5 = pd.read_csv("results/baseline_sweep_raw.csv")
        n5_rows = []
        for strat in STRATEGIES:
            sub = df_n5[df_n5['strategy'] == strat]
            n5_rows.append({
                'N': 5, 'Strategy': strat.upper(),
                'dp_mean': sub['dp_gap'].mean(),
                'dp_std':  sub['dp_gap'].std()
            })
        df_n5_sum = pd.DataFrame(n5_rows)
        df_all = pd.concat([df_n5_sum, df_summary[['N','Strategy','dp_mean','dp_std']]], ignore_index=True)
        df_all.to_csv("results/full_scaleup_comparison.csv", index=False)
        print("\n--- Full Scale Comparison (n=5 baseline vs n=10 vs n=20) ---")
        print(df_all.sort_values(['Strategy','N'])[['Strategy','N','dp_mean','dp_std']].to_string(index=False))
    except Exception as e:
        print(f"(Could not merge n=5 baseline: {e})")

    return df_summary, df_raw

if __name__ == "__main__":
    run_phase11_scaleup()
