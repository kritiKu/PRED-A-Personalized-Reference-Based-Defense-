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

from data.preprocess import prepare_data
from models.mlp import SimpleMLP
from clients.honest import train_honest_client
from attacks.lora_attack import train_malicious_client
from defenses.aggregators import fedavg_aggregate

SEEDS = [42, 43, 44, 45, 46]

def compute_signal(client_update, all_updates, client_sizes):
    """
    Lightweight signal (no FLTrust needed):
    1. Compute weighted FedAvg aggregate of all updates
    2. Signal = cosine similarity of each client with aggregate
    3. Also record L2 norm ratio (client / median honest norm)
    """
    total = sum(client_sizes)
    agg = torch.zeros_like(client_update)
    for u, s in zip(all_updates, client_sizes):
        agg += (s / total) * u

    agg_norm = torch.norm(agg).item()
    cli_norm = torch.norm(client_update).item()

    if agg_norm < 1e-8 or cli_norm < 1e-8:
        cos_sim = 0.0
    else:
        cos_sim = torch.dot(client_update, agg).item() / (cli_norm * agg_norm)

    return cos_sim, cli_norm

def run_phase7_gate():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\n=======================================================")
    print("PHASE 7 (FAST): CRITICAL GATE -- MULTI-SEED SIGNAL CHECK")
    print("=======================================================")
    print("Signal: Cosine similarity with FedAvg aggregate + L2 norm ratio")
    print("5 seeds x 30 rounds x 5 clients = 750 data points (NO FLTrust)")

    records = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        data_dir = f"data/seed_{seed}"
        prepare_data(seed=seed, n_clients=5, alpha=0.5, ref_size=250, save_dir=data_dir)

        client_data_list = [
            torch.load(f"{data_dir}/client_{i}.pt", weights_only=False) for i in range(5)
        ]
        input_dim = client_data_list[0]['x'].shape[1]

        torch.manual_seed(seed)
        global_model = SimpleMLP(input_dim)
        global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

        for r in range(1, 31):
            client_updates = []
            client_sizes = []

            for i in range(5):
                c_seed = seed + r * 100 + i
                if i == 3:
                    update, n = train_malicious_client(
                        SimpleMLP, input_dim, global_weights, client_data_list[3],
                        batch_size=32, local_epochs=5, lr=0.01, seed=c_seed,
                        target_norm=None
                    )
                else:
                    update, n = train_honest_client(
                        SimpleMLP, input_dim, global_weights, client_data_list[i],
                        batch_size=32, local_epochs=5, lr=0.01, seed=c_seed
                    )
                client_updates.append(update)
                client_sizes.append(n)

            # Compute honest median norm for relative norm signal
            honest_norms = [torch.norm(client_updates[i]).item() for i in [0,1,2,4]]
            median_h_norm = float(np.median(honest_norms))

            for i in range(5):
                cos_sim, cli_norm = compute_signal(client_updates[i], client_updates, client_sizes)
                norm_ratio = cli_norm / (median_h_norm + 1e-8)

                records.append({
                    'seed': seed,
                    'round': r,
                    'client_id': i,
                    'group': 'Attacker' if i == 3 else 'Honest',
                    'client_label': f"C{i}(A)" if i == 3 else f"C{i}",
                    'cos_sim': cos_sim,
                    'norm_ratio': norm_ratio
                })

            # Evolve global model with FedAvg
            global_weights, _ = fedavg_aggregate(global_weights, client_updates, client_sizes)
            global_model.load_state_dict(global_weights)

        print(f"  Seed {seed}: 30 rounds done", flush=True)

    df = pd.DataFrame(records)
    df.to_csv("logs/raw_signal_all_seeds.csv", index=False)
    print(f"\nSaved {len(df)} records to logs/raw_signal_all_seeds.csv")

    # ─── Plot ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    order = ['C0', 'C1', 'C2', 'C3(A)', 'C4']
    colors = {
        'C0': '#2980b9', 'C1': '#27ae60', 'C2': '#8e44ad',
        'C3(A)': '#e74c3c', 'C4': '#f39c12'
    }

    for ax, metric, ylabel, title in zip(
        axes,
        ['cos_sim', 'norm_ratio'],
        ['Cosine Similarity with FedAvg Aggregate', 'L2 Norm Ratio (vs Median Honest)'],
        ['Cosine Similarity Signal', 'L2 Norm Ratio Signal']
    ):
        for label in order:
            sub = df[df['client_label'] == label][metric]
            bp = ax.boxplot(
                sub.values,
                positions=[order.index(label)],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=colors[label], alpha=0.7),
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4)
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(
        "Phase 7 Critical Gate: Raw Signal Distribution\n"
        "(5 seeds × 30 rounds = 150 data points per client)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("results/phase7_raw_signal_distribution.png", dpi=200)
    plt.close()
    print("Saved box plot to results/phase7_raw_signal_distribution.png")

    # ─── Gate Decision ───────────────────────────────────────────────────────
    print("\n--- Signal Distribution Summary ---")
    for label in order:
        sub_cos = df[df['client_label'] == label]['cos_sim']
        sub_norm = df[df['client_label'] == label]['norm_ratio']
        print(f"  {label}: cos_sim={sub_cos.mean():.4f}+/-{sub_cos.std():.4f} | "
              f"norm_ratio={sub_norm.mean():.4f}+/-{sub_norm.std():.4f}")

    att_cos = df[df['client_id'] == 3]['cos_sim']
    att_q25 = np.percentile(att_cos, 25)
    att_q75 = np.percentile(att_cos, 75)

    overlap = False
    for i in [0, 1, 2, 4]:
        h_cos = df[df['client_id'] == i]['cos_sim']
        h_q25 = np.percentile(h_cos, 25)
        h_q75 = np.percentile(h_cos, 75)
        if att_q25 <= h_q75 and h_q25 <= att_q75:
            overlap = True
            print(f"  [IQR OVERLAP] C3 overlaps with C{i}")

    print("\n=======================================================")
    if not overlap:
        verdict = "PASS"
        print("GATE DECISION: PASS")
        print("Attacker (C3) signal cleanly separates from all honest clients.")
        print("Action: Proceed to Phase 8 & 9 (Per-client EWMA defense).")
    else:
        verdict = "FAIL"
        print("GATE DECISION: FAIL (IQR overlap detected)")
        print("Fixed-threshold detector NOT reliable on this signal family.")
        print("Action: Proceed to Phase 11 (Scale-up n=10, n=20).")
    print("=======================================================\n")

    df.to_csv("results/phase7_gate_result.csv", index=False)
    with open("results/phase7_verdict.txt", "w") as f:
        f.write(verdict)

    return verdict, df

if __name__ == "__main__":
    run_phase7_gate()
