import os
import sys
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.runner import run_fl_experiment

SEEDS = [42, 43, 44, 45, 46]
STRATEGIES = ['fedavg', 'krum', 'fltrust']

def run_phase6_sweep():
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\n=======================================================")
    print("PHASE 6: MULTI-SEED BASELINE SWEEP (5 SEEDS: 42-46)")
    print("=======================================================")
    
    all_results = []
    
    for strat in STRATEGIES:
        for seed in SEEDS:
            # Skip already completed runs
            log_path = f"logs/sweep_{strat}_seed{seed}.csv"
            if os.path.exists(log_path):
                print(f"---> SKIPPING {strat.upper()} | Seed {seed} (already done)")
                df = pd.read_csv(log_path)
                last = df.iloc[-1]
                all_results.append({
                    'strategy': strat,
                    'seed': seed,
                    'accuracy': float(last['accuracy']),
                    'dp_gap': float(last['dp_gap']),
                    'eo_gap': float(last['eo_gap'])
                })
                continue
                
            print(f"\n---> Running {strat.upper()} | Seed {seed}")
            data_dir = f"data/seed_{seed}"
            df_logs, final_metrics = run_fl_experiment(
                strategy=strat, attack=True, seed=seed, rounds=30,
                batch_size=32, local_epochs=5, lr=0.01,
                log_path=log_path,
                data_dir=data_dir
            )
            res = {
                'strategy': strat,
                'seed': seed,
                'accuracy': final_metrics['accuracy'],
                'dp_gap': final_metrics['dp_gap'],
                'eo_gap': final_metrics['eo_gap']
            }
            all_results.append(res)
            print(f"    Done: Acc={final_metrics['accuracy']:.4f} | dDP={final_metrics['dp_gap']:.4f} | dEO={final_metrics['eo_gap']:.4f}")
            
    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv("results/baseline_sweep_raw.csv", index=False)
    
    summary_rows = []
    for strat in STRATEGIES:
        sub = df_raw[df_raw['strategy'] == strat]
        acc_mean, acc_std = sub['accuracy'].mean(), sub['accuracy'].std()
        dp_mean, dp_std = sub['dp_gap'].mean(), sub['dp_gap'].std()
        eo_mean, eo_std = sub['eo_gap'].mean(), sub['eo_gap'].std()
        
        summary_rows.append({
            'Strategy': strat.upper(),
            'Accuracy': f"{acc_mean:.4f}+/-{acc_std:.4f}",
            'dDP': f"{dp_mean:.4f}+/-{dp_std:.4f}",
            'dEO': f"{eo_mean:.4f}+/-{eo_std:.4f}",
            'acc_mean': acc_mean, 'acc_std': acc_std,
            'dp_mean': dp_mean, 'dp_std': dp_std,
            'eo_mean': eo_mean, 'eo_std': eo_std
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/baseline_comparison.csv", index=False)
    
    print("\n=======================================================")
    print("PHASE 6 BASELINE COMPARISON RESULTS")
    print("=======================================================")
    print(df_summary[['Strategy', 'Accuracy', 'dDP', 'dEO']].to_string(index=False))
    
    return df_summary, df_raw

if __name__ == "__main__":
    run_phase6_sweep()
