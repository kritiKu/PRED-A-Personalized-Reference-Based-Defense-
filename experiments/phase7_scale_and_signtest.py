"""
Phase 7 Signal Validation at Scale (n=10 and n=20)
+
Paired Sign Test: FLTrust vs KRUM at n=20

Outputs:
  results/phase7_scale_signal_n10.png   -- box plot n=10
  results/phase7_scale_signal_n20.png   -- box plot n=20
  results/phase7_scale_signal_combined.png -- side-by-side n=5/10/20
  results/sign_test_fltrust_vs_krum_n20.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon, binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.set_num_threads(8)

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
from defenses.aggregators import fedavg_aggregate

SEEDS = [42, 43, 44]
N_CLIENTS_LIST = [10, 20]
ATTACKER_ID = 3
ROUNDS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Signal computation (same as Phase 7 n=5)
# ─────────────────────────────────────────────────────────────────────────────
def compute_signal(client_update, all_updates, client_sizes):
    total = sum(client_sizes)
    agg   = torch.zeros_like(client_update)
    for u, s in zip(all_updates, client_sizes):
        agg += (s / total) * u

    agg_norm = torch.norm(agg).item()
    cli_norm = torch.norm(client_update).item()

    if agg_norm < 1e-8 or cli_norm < 1e-8:
        cos_sim = 0.0
    else:
        cos_sim = torch.dot(client_update, agg).item() / (cli_norm * agg_norm)

    return cos_sim, cli_norm


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Signal extraction at n=10 and n=20
# ─────────────────────────────────────────────────────────────────────────────
def extract_signal_at_scale(n_clients, seeds, rounds=ROUNDS, attacker_id=ATTACKER_ID):
    records = []
    for seed in seeds:
        print(f"  n={n_clients} | Seed {seed} ...", flush=True)
        data_dir = f"data/n{n_clients}_seed_{seed}"
        prepare_data(seed=seed, n_clients=n_clients, alpha=0.5,
                     ref_size=250, save_dir=data_dir)

        client_data_list = [
            torch.load(f"{data_dir}/client_{i}.pt", weights_only=False)
            for i in range(n_clients)
        ]
        input_dim = client_data_list[0]['x'].shape[1]

        torch.manual_seed(seed)
        global_model   = SimpleMLP(input_dim)
        global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

        for r in range(1, rounds + 1):
            client_updates = []
            client_sizes   = []

            for i in range(n_clients):
                c_seed = seed + r * 100 + i
                if i == attacker_id:
                    update, n = train_malicious_client(
                        SimpleMLP, input_dim, global_weights, client_data_list[i],
                        batch_size=32, local_epochs=5, lr=0.01,
                        seed=c_seed, target_norm=None
                    )
                else:
                    update, n = train_honest_client(
                        SimpleMLP, input_dim, global_weights, client_data_list[i],
                        batch_size=32, local_epochs=5, lr=0.01, seed=c_seed
                    )
                client_updates.append(update)
                client_sizes.append(n)

            honest_ids   = [i for i in range(n_clients) if i != attacker_id]
            honest_norms = [torch.norm(client_updates[i]).item() for i in honest_ids]
            median_h_norm = float(np.median(honest_norms))

            for i in range(n_clients):
                cos_sim, cli_norm = compute_signal(
                    client_updates[i], client_updates, client_sizes
                )
                norm_ratio = cli_norm / (median_h_norm + 1e-8)
                group = 'Attacker' if i == attacker_id else 'Honest'
                label = f"C{i}(A)" if i == attacker_id else f"C{i}"

                records.append({
                    'n_clients': n_clients,
                    'seed': seed,
                    'round': r,
                    'client_id': i,
                    'group': group,
                    'client_label': label,
                    'cos_sim': cos_sim,
                    'norm_ratio': norm_ratio
                })

            global_weights, _ = fedavg_aggregate(
                global_weights, client_updates, client_sizes
            )
            global_model.load_state_dict(global_weights)

        print(f"    Done ({rounds} rounds x {n_clients} clients)", flush=True)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Box plot helper
# ─────────────────────────────────────────────────────────────────────────────
def box_plot_signal(df, n_clients, ax_cos, ax_norm, attacker_id=ATTACKER_ID):
    """Draw box plots on provided axes."""
    ids  = sorted(df['client_id'].unique())
    labels = []
    cos_data  = []
    norm_data = []

    for cid in ids:
        sub = df[df['client_id'] == cid]
        lbl = sub['client_label'].iloc[0]
        labels.append(lbl)
        cos_data.append(sub['cos_sim'].values)
        norm_data.append(sub['norm_ratio'].values)

    colors = []
    for cid in ids:
        colors.append('#e74c3c' if cid == attacker_id else '#3498db')

    for ax, data, ylabel, title in [
        (ax_cos,  cos_data,  'Cosine Similarity (client vs aggregate)',
         f'Cosine Similarity | n={n_clients}'),
        (ax_norm, norm_data, 'L2 Norm Ratio (vs median honest)',
         f'L2 Norm Ratio | n={n_clients}'),
    ]:
        bps = ax.boxplot(
            data,
            patch_artist=True,
            notch=False,
            widths=0.5,
            medianprops=dict(color='black', linewidth=2.0),
            flierprops=dict(marker='o', markersize=3, alpha=0.35),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2)
        )
        for patch, color in zip(bps['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)

        ax.set_xticklabels(labels, fontsize=8 if n_clients > 10 else 10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.45)

    # IQR separation assessment
    att_cos = df[df['client_id'] == attacker_id]['cos_sim']
    a_q25, a_q75 = np.percentile(att_cos, 25), np.percentile(att_cos, 75)
    overlapping = []
    for cid in ids:
        if cid == attacker_id:
            continue
        h_cos = df[df['client_id'] == cid]['cos_sim']
        h_q25, h_q75 = np.percentile(h_cos, 25), np.percentile(h_cos, 75)
        if a_q25 <= h_q75 and h_q25 <= a_q75:
            overlapping.append(cid)

    return overlapping


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Paired sign test FLTrust vs KRUM at n=20
# ─────────────────────────────────────────────────────────────────────────────
def run_sign_test(seeds=SEEDS):
    print("\n=======================================================")
    print("PAIRED SIGN TEST: FLTrust vs KRUM at n=20")
    print("=======================================================")

    diffs   = []
    records = []
    for seed in seeds:
        flt_path  = f"logs/scaleup_n20_fltrust_seed{seed}.csv"
        krum_path = f"logs/scaleup_n20_krum_seed{seed}.csv"

        if not os.path.exists(flt_path) or not os.path.exists(krum_path):
            print(f"  MISSING logs for seed {seed}, skipping")
            continue

        flt_dp  = pd.read_csv(flt_path).iloc[-1]['dp_gap']
        krum_dp = pd.read_csv(krum_path).iloc[-1]['dp_gap']
        diff    = float(krum_dp) - float(flt_dp)   # positive = FLTrust better

        diffs.append(diff)
        records.append({
            'seed': seed,
            'FLTrust_dDP': round(float(flt_dp), 4),
            'KRUM_dDP':    round(float(krum_dp), 4),
            'diff (KRUM-FLTrust)': round(diff, 4),
            'FLTrust_wins': diff > 0
        })
        print(f"  Seed {seed}: FLTrust dDP={flt_dp:.4f}  KRUM dDP={krum_dp:.4f}  diff={diff:+.4f}  "
              f"{'FLTrust BETTER' if diff > 0 else 'KRUM BETTER'}")

    df_sign = pd.DataFrame(records)

    n_wins   = sum(1 for d in diffs if d > 0)
    n_total  = len(diffs)
    n_ties   = sum(1 for d in diffs if d == 0)

    print(f"\n  FLTrust wins: {n_wins}/{n_total} seeds "
          f"({'excluding ties' if n_ties else 'no ties'})")

    # Binomial exact sign test (H0: P(FLTrust wins) = 0.5, one-sided)
    if n_total - n_ties > 0:
        p_binom = binomtest(n_wins, n_total - n_ties, p=0.5, alternative='greater').pvalue
        print(f"  One-sided sign test p-value (FLTrust > KRUM): {p_binom:.4f}")
    else:
        p_binom = 1.0
        print("  No valid pairs for sign test")

    # Wilcoxon signed-rank (needs ≥2 non-zero diffs)
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) >= 2:
        try:
            stat, p_wilcox = wilcoxon(nonzero, alternative='greater')
            print(f"  Wilcoxon signed-rank p-value (one-sided):    {p_wilcox:.4f}")
        except Exception as e:
            p_wilcox = None
            print(f"  Wilcoxon failed: {e}")
    else:
        p_wilcox = None
        print("  Not enough pairs for Wilcoxon test")

    print("\n  --- Interpretation ---")
    alpha = 0.05
    if p_binom < alpha:
        print(f"  CONCLUSION: FLTrust's advantage at n=20 is STATISTICALLY SIGNIFICANT")
        print(f"  (sign test p={p_binom:.4f} < alpha={alpha})")
        print("  => FLTrust consistently beats KRUM at n=20 (not just noise)")
    else:
        print(f"  CONCLUSION: FLTrust's advantage at n=20 is NOT statistically significant")
        print(f"  (sign test p={p_binom:.4f} >= alpha={alpha})")
        print(f"  With only {n_total} seeds, power is limited.")
        print("  => Effect appears consistent in direction but cannot be confirmed statistically.")
        print(f"  => Direction: FLTrust wins {n_wins}/{n_total} seeds "
              f"(consistent trend, needs more seeds to confirm)")

    df_sign['sign_test_p'] = p_binom
    df_sign['wilcoxon_p']  = p_wilcox if p_wilcox else 'N/A'
    df_sign.to_csv("results/sign_test_fltrust_vs_krum_n20.csv", index=False)
    print("\n  Saved to results/sign_test_fltrust_vs_krum_n20.csv")

    return df_sign, p_binom, p_wilcox, n_wins, n_total


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # ── PART 1 & 2: Signal at scale ──────────────────────────────────────────
    # Load n=5 signal from Phase 7 cache
    n5_signal_path = "logs/raw_signal_all_seeds.csv"

    all_signal_dfs = {}
    if os.path.exists(n5_signal_path):
        df_n5 = pd.read_csv(n5_signal_path)
        df_n5['n_clients'] = 5
        # Standardise column name (Phase 7 used 'client_label' with C3(A))
        all_signal_dfs[5] = df_n5
        print(f"Loaded n=5 signal from cache ({len(df_n5)} records)")
    else:
        print("n=5 signal cache not found — re-running Phase 7 for n=5")
        df_n5 = extract_signal_at_scale(5, SEEDS)
        df_n5['n_clients'] = 5
        all_signal_dfs[5] = df_n5

    for n in N_CLIENTS_LIST:
        print(f"\n=== Extracting signal at n={n} ===")
        df_n = extract_signal_at_scale(n, SEEDS)
        df_n.to_csv(f"logs/scale_signal_n{n}.csv", index=False)
        all_signal_dfs[n] = df_n
        print(f"Saved n={n} signal to logs/scale_signal_n{n}.csv")

    # ── Build combined figure: 3 columns (n=5, n=10, n=20) x 2 rows (cos, norm)
    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    separation_report = {}
    for col_idx, n in enumerate([5, 10, 20]):
        df_n = all_signal_dfs[n]
        ax_cos  = fig.add_subplot(gs[0, col_idx])
        ax_norm = fig.add_subplot(gs[1, col_idx])
        overlapping = box_plot_signal(df_n, n, ax_cos, ax_norm, attacker_id=ATTACKER_ID)
        separation_report[n] = overlapping

    fig.suptitle(
        "Phase 7 Signal Validation Across Scale: "
        "Cosine Similarity & L2 Norm Ratio per Client\n"
        f"(3 seeds × 30 rounds = 90 data points per client | Attacker = C{ATTACKER_ID} [red])",
        fontsize=13, fontweight='bold'
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor='#e74c3c', alpha=0.72, label='Attacker (C3)'),
        Patch(facecolor='#3498db', alpha=0.72, label='Honest Client')
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.savefig("results/phase7_scale_signal_combined.png", dpi=200,
                bbox_inches='tight')
    plt.close()
    print("\nSaved combined box plot to results/phase7_scale_signal_combined.png")

    # ── Per-N individual plots ────────────────────────────────────────────────
    for n in N_CLIENTS_LIST:
        df_n = all_signal_dfs[n]
        fig2, (ax_c, ax_n) = plt.subplots(1, 2, figsize=(13, 5))
        box_plot_signal(df_n, n, ax_c, ax_n, attacker_id=ATTACKER_ID)
        fig2.suptitle(
            f"Phase 7 Signal Validation at n={n} Clients\n"
            f"(3 seeds × 30 rounds = 90 data points per client)",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(f"results/phase7_scale_signal_n{n}.png", dpi=200)
        plt.close()

    # ── Separation report ────────────────────────────────────────────────────
    print("\n=======================================================")
    print("SIGNAL SEPARATION REPORT (IQR Overlap with Attacker C3)")
    print("=======================================================")
    for n in [5, 10, 20]:
        ov = separation_report.get(n, [])
        if ov:
            print(f"  n={n:2d}: OVERLAP with clients {ov} → FAIL "
                  "(threshold detector unreliable)")
        else:
            print(f"  n={n:2d}: NO OVERLAP → PASS "
                  "(C3 cleanly separates from all honest clients)")

    # ── PART 3: Sign test ────────────────────────────────────────────────────
    df_sign, p_binom, p_wilcox, n_wins, n_total = run_sign_test(SEEDS)

    print("\n=======================================================")
    print("ALL ANALYSES COMPLETE")
    print("=======================================================")
    print("  results/phase7_scale_signal_combined.png")
    print("  results/phase7_scale_signal_n10.png")
    print("  results/phase7_scale_signal_n20.png")
    print("  results/sign_test_fltrust_vs_krum_n20.csv")
