import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_phase10_evaluation(df_baseline_raw, df_defense_raw):
    os.makedirs("results", exist_ok=True)
    
    # Combine baseline raw results with defense raw results
    df_combined = pd.concat([df_baseline_raw, df_defense_raw], ignore_index=True)
    df_combined.to_csv("results/all_seeds_raw.csv", index=False)
    
    strategies = df_combined['strategy'].unique()
    summary_rows = []
    
    for strat in strategies:
        sub = df_combined[df_combined['strategy'] == strat]
        acc_mean, acc_std = sub['accuracy'].mean(), sub['accuracy'].std()
        dp_mean, dp_std = sub['dp_gap'].mean(), sub['dp_gap'].std()
        eo_mean, eo_std = sub['eo_gap'].mean(), sub['eo_gap'].std()
        
        summary_rows.append({
            'Strategy': strat.upper(),
            'Accuracy (Mean ± Std)': f"{acc_mean:.4f} ± {acc_std:.4f}",
            'ΔDP (Mean ± Std)': f"{dp_mean:.4f} ± {dp_std:.4f}",
            'ΔEO (Mean ± Std)': f"{eo_mean:.4f} ± {eo_std:.4f}",
            'acc_mean': acc_mean, 'acc_std': acc_std,
            'dp_mean': dp_mean, 'dp_std': dp_std,
            'eo_mean': eo_mean, 'eo_std': eo_std
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/final_comparison.csv", index=False)
    
    print("\n=======================================================")
    print("PHASE 10: FINAL MULTI-SEED COMPARISON TABLE (results/final_comparison.csv)")
    print("=======================================================")
    print(df_summary[['Strategy', 'Accuracy (Mean ± Std)', 'ΔDP (Mean ± Std)', 'ΔEO (Mean ± Std)']].to_string(index=False))
    
    # Paired Sign Tests (Same-seed differences)
    print("\n=======================================================")
    print("PHASE 10: PAIRED SIGN TESTS (Same-Seed Pairings Across 5 Seeds)")
    print("=======================================================")
    
    seeds = sorted(df_combined['seed'].unique())
    def_sub = df_combined[df_combined['strategy'] == 'OUR_DEFENSE'].set_index('seed')
    
    sign_test_results = []
    
    for base_strat in ['fedavg', 'krum', 'fltrust']:
        base_sub = df_combined[df_combined['strategy'] == base_strat].set_index('seed')
        
        for metric in ['accuracy', 'dp_gap', 'eo_gap']:
            # Calculate difference: Defense - Baseline
            diffs = [def_sub.loc[s, metric] - base_sub.loc[s, metric] for s in seeds]
            signs = [np.sign(d) for d in diffs if d != 0]
            
            # For accuracy: higher is better (+ sign is win)
            # For dp_gap & eo_gap: lower is better (- sign is win)
            if metric == 'accuracy':
                consistent_win = all(s > 0 for s in signs) and len(signs) == 5
            else:
                consistent_win = all(s < 0 for s in signs) and len(signs) == 5
                
            sign_str = ", ".join([f"{d:+.4f}" for d in diffs])
            
            if consistent_win:
                verdict = f"BEATS {base_strat.upper()} (Consistent sign across all 5 seeds)"
            else:
                verdict = f"Not statistically distinguishable at n=5 seeds (Signs flip or inconsistent)"
                
            sign_test_results.append({
                'Metric': metric,
                'Comparison': f"OUR_DEFENSE vs {base_strat.upper()}",
                'Same-Seed Diffs': sign_str,
                'Verdict': verdict
            })
            
    df_signs = pd.DataFrame(sign_test_results)
    df_signs.to_csv("results/paired_sign_tests.csv", index=False)
    print(df_signs.to_string(index=False))
    return df_summary, df_signs

if __name__ == "__main__":
    df_b = pd.read_csv("results/baseline_sweep_raw.csv")
    df_d = pd.read_csv("logs/our_defense_multiseed.csv")
    run_phase10_evaluation(df_b, df_d)
