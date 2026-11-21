import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.runner import run_fl_experiment

def run_phase4(seed=42):
    log_path = "logs/attack_krum.csv"
    print(f"=== Running Phase 4 Multi-KRUM Defense (Seed: {seed}) ===")
    df_logs, final_metrics = run_fl_experiment(
        strategy='krum', attack=True, seed=seed, rounds=30,
        batch_size=32, local_epochs=5, lr=0.01, log_path=log_path
    )
    print(f"\nPhase 4 Multi-KRUM Summary (Seed {seed}):")
    print(f"Final Acc: {final_metrics['accuracy']:.4f} | dDP: {final_metrics['dp_gap']:.4f} | dEO: {final_metrics['eo_gap']:.4f}")
    return df_logs, final_metrics

if __name__ == "__main__":
    run_phase4(seed=42)
