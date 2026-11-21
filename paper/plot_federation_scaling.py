import os, ast, glob, pandas as pd, numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. EXTRACT DATA DIRECTLY FROM EXPERIMENT LOGS
# -------------------------------------------------------------
scales = [5, 10, 20]
seeds = [43, 44, 45, 46, 47]
attackers_map = {
    5: [3],
    10: [3, 8],
    20: [3, 8, 13, 18]
}

pred_rows = []
fedavg_rows = []

for n in scales:
    att_list = attackers_map[n]
    n_att = len(att_list)
    folder = f"C:/Users/HP/Desktop/pred/logs/unified_n{n}"
    
    # Process PRED-v2
    for s in seeds:
        fpath = os.path.join(folder, f"pred-v2_seed{s}.csv")
        assert os.path.exists(fpath), f"Missing {fpath}"
        df = pd.read_csv(fpath)
        last = df.iloc[-1]
        
        # Calculate detection stats
        tp = 0
        fp = 0
        total_att = len(df) * n_att
        total_honest = len(df) * (n - n_att)
        for excl_str in df['excluded']:
            try:
                excl = ast.literal_eval(excl_str)
            except:
                excl = []
            for c in excl:
                if c in att_list:
                    tp += 1
                else:
                    fp += 1
        tpr = tp / total_att
        fpr = fp / total_honest
        
        pred_rows.append({
            'Strategy': 'PRED',
            'N': n,
            'Seed': s,
            'Rounds': len(df),
            'Attackers': n_att,
            'Fraction': f"{n_att/n:.0%}",
            'Rank': 4,
            'Accuracy': float(last['accuracy']),
            'Delta_DP': float(last['dp_gap']),
            'Delta_EO': float(last['eo_gap']),
            'Detection_Rate_TPR': tpr,
            'FPR': fpr,
            'TP_Count': tp,
            'Total_Att_Opps': total_att,
            'FP_Count': fp,
            'Total_Honest_Opps': total_honest,
            'Source_File': fpath
        })
        
    # Process FedAvg
    for s in seeds:
        fpath = os.path.join(folder, f"fedavg_seed{s}.csv")
        assert os.path.exists(fpath), f"Missing {fpath}"
        df = pd.read_csv(fpath)
        last = df.iloc[-1]
        
        fedavg_rows.append({
            'Strategy': 'FedAvg',
            'N': n,
            'Seed': s,
            'Rounds': len(df),
            'Attackers': n_att,
            'Fraction': f"{n_att/n:.0%}",
            'Rank': 4,
            'Accuracy': float(last['accuracy']),
            'Delta_DP': float(last['dp_gap']),
            'Delta_EO': float(last['eo_gap']),
            'Detection_Rate_TPR': np.nan,
            'Source_File': fpath
        })

df_pred = pd.DataFrame(pred_rows)
df_fed = pd.DataFrame(fedavg_rows)

# -------------------------------------------------------------
# 2. SAVE INDEPENDENT VERIFICATION TABLES
# -------------------------------------------------------------
out_paper_dir = "C:/Users/HP/Desktop/pred/paper"
out_brain_dir = "C:/Users/HP/.gemini/antigravity-ide/brain/18414c5f-eed7-43cb-9a90-f4a0cbab355b"

# Save full per-seed CSV table
df_pred.to_csv(os.path.join(out_paper_dir, "table_federation_scaling_raw.csv"), index=False)
df_pred.to_csv(os.path.join(out_brain_dir, "table_federation_scaling_raw.csv"), index=False)

# Compute summary stats (sample std, ddof=1)
summary_records = []
for n in scales:
    sub = df_pred[df_pred['N'] == n]
    sub_fed = df_fed[df_fed['N'] == n]
    summary_records.append({
        'Federation_Size_N': n,
        'Malicious_Clients': sub['Attackers'].iloc[0],
        'Attacker_Fraction': '20%',
        'Attack_Rank': 4,
        'Evaluation_Rounds': 30,
        'Num_Seeds': len(sub),
        'Seeds': str(sub['Seed'].tolist()),
        'Detection_Rate_Mean': sub['Detection_Rate_TPR'].mean(),
        'Detection_Rate_Std': sub['Detection_Rate_TPR'].std(),
        'Pooled_Detection_Rate': sub['TP_Count'].sum() / sub['Total_Att_Opps'].sum(),
        'DP_Gap_Mean': sub['Delta_DP'].mean(),
        'DP_Gap_Std': sub['Delta_DP'].std(),
        'EO_Gap_Mean': sub['Delta_EO'].mean(),
        'EO_Gap_Std': sub['Delta_EO'].std(),
        'Accuracy_Mean': sub['Accuracy'].mean(),
        'Accuracy_Std': sub['Accuracy'].std(),
        'FedAvg_DP_Mean': sub_fed['Delta_DP'].mean(),
        'FedAvg_DP_Std': sub_fed['Delta_DP'].std(),
        'FedAvg_EO_Mean': sub_fed['Delta_EO'].mean(),
        'FedAvg_EO_Std': sub_fed['Delta_EO'].std(),
        'FedAvg_Acc_Mean': sub_fed['Accuracy'].mean(),
        'FedAvg_Acc_Std': sub_fed['Accuracy'].std(),
    })

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv(os.path.join(out_paper_dir, "table_federation_scaling_summary.csv"), index=False)
df_summary.to_csv(os.path.join(out_brain_dir, "table_federation_scaling_summary.csv"), index=False)

# -------------------------------------------------------------
# 3. GENERATE PUBLICATION-QUALITY IEEE TNSM FIGURE
# -------------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11.5
plt.rcParams['xtick.labelsize'] = 10.5
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['figure.titlesize'] = 12

fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))

x_indices = np.array([5, 10, 20])
x_labels = [r'$N=5$', r'$N=10$', r'$N=20$']

# Jitter offsets to prevent error bar collision
jitter_pred = -0.3
jitter_fed = 0.3

# Colors
c_pred = '#006699'       # Deep Navy / Slate Blue
c_fed = '#777777'        # Neutral Charcoal Gray
c_det = '#1b9e77'        # Emerald Green

# Panel (a): Detection Rate (TPR)
ax_det = axes[0, 0]
det_means = df_summary['Detection_Rate_Mean'].values * 100
det_stds = df_summary['Detection_Rate_Std'].values * 100
ax_det.errorbar(x_indices, det_means, yerr=det_stds, fmt='o-', color=c_det, 
                ecolor=c_det, elinewidth=1.6, capsize=4.5, capthick=1.4, 
                markersize=7, linewidth=1.8, label='PRED Detection (TPR)')
for x, y in zip(x_indices, det_means):
    ax_det.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 9),
                    ha='center', fontsize=9.5, fontweight='bold', color=c_det)
ax_det.set_title('(a) Malicious Client Detection Rate', fontweight='bold', pad=8)
ax_det.set_ylabel('Detection Rate TPR (%)')
ax_det.set_ylim(80, 106)
ax_det.set_xlim(3, 22)
ax_det.set_xticks(x_indices)
ax_det.set_xticklabels(x_labels)
ax_det.grid(True, linestyle='--', alpha=0.5)

# Panel (b): Demographic Parity Gap
ax_dp = axes[0, 1]
dp_means = df_summary['DP_Gap_Mean'].values
dp_stds = df_summary['DP_Gap_Std'].values
fed_dp_means = df_summary['FedAvg_DP_Mean'].values
fed_dp_stds = df_summary['FedAvg_DP_Std'].values

ax_dp.errorbar(x_indices + jitter_pred, dp_means, yerr=dp_stds, fmt='s-', color=c_pred, 
               ecolor=c_pred, elinewidth=1.6, capsize=4.5, capthick=1.4, 
               markersize=6.5, linewidth=1.8, label='PRED')
ax_dp.errorbar(x_indices + jitter_fed, fed_dp_means, yerr=fed_dp_stds, fmt='^--', color=c_fed, 
               ecolor=c_fed, elinewidth=1.4, capsize=4, capthick=1.2, 
               markersize=6.5, linewidth=1.5, alpha=0.9, label='FedAvg (Undefended)')

ax_dp.set_title(r'(b) Demographic Parity Gap ($\Delta DP$)', fontweight='bold', pad=8)
ax_dp.set_ylabel(r'$\Delta DP$ (Lower is Better)')
ax_dp.set_ylim(0.04, 0.28)
ax_dp.set_xlim(3, 22)
ax_dp.set_xticks(x_indices)
ax_dp.set_xticklabels(x_labels)
ax_dp.grid(True, linestyle='--', alpha=0.5)
ax_dp.legend(loc='upper right', framealpha=0.9)

# Panel (c): Equalized Odds Gap
ax_eo = axes[1, 0]
eo_means = df_summary['EO_Gap_Mean'].values
eo_stds = df_summary['EO_Gap_Std'].values
fed_eo_means = df_summary['FedAvg_EO_Mean'].values
fed_eo_stds = df_summary['FedAvg_EO_Std'].values

ax_eo.errorbar(x_indices + jitter_pred, eo_means, yerr=eo_stds, fmt='s-', color=c_pred, 
               ecolor=c_pred, elinewidth=1.6, capsize=4.5, capthick=1.4, 
               markersize=6.5, linewidth=1.8, label='PRED')
ax_eo.errorbar(x_indices + jitter_fed, fed_eo_means, yerr=fed_eo_stds, fmt='^--', color=c_fed, 
               ecolor=c_fed, elinewidth=1.4, capsize=4, capthick=1.2, 
               markersize=6.5, linewidth=1.5, alpha=0.9, label='FedAvg (Undefended)')

ax_eo.set_title(r'(c) Equalized Odds Gap ($\Delta EO$)', fontweight='bold', pad=8)
ax_eo.set_ylabel(r'$\Delta EO$ (Lower is Better)')
ax_eo.set_xlabel('Federation Size ($N$)')
ax_eo.set_ylim(0.01, 0.16)
ax_eo.set_xlim(3, 22)
ax_eo.set_xticks(x_indices)
ax_eo.set_xticklabels(x_labels)
ax_eo.grid(True, linestyle='--', alpha=0.5)
ax_eo.legend(loc='upper right', framealpha=0.9)

# Panel (d): Test Accuracy
ax_acc = axes[1, 1]
acc_means = df_summary['Accuracy_Mean'].values
acc_stds = df_summary['Accuracy_Std'].values
fed_acc_means = df_summary['FedAvg_Acc_Mean'].values
fed_acc_stds = df_summary['FedAvg_Acc_Std'].values

ax_acc.errorbar(x_indices + jitter_pred, acc_means, yerr=acc_stds, fmt='o-', color=c_pred, 
                ecolor=c_pred, elinewidth=1.6, capsize=4.5, capthick=1.4, 
                markersize=6.5, linewidth=1.8, label='PRED')
ax_acc.errorbar(x_indices + jitter_fed, fed_acc_means, yerr=fed_acc_stds, fmt='^--', color=c_fed, 
                ecolor=c_fed, elinewidth=1.4, capsize=4, capthick=1.2, 
                markersize=6.5, linewidth=1.5, alpha=0.9, label='FedAvg (Undefended)')

ax_acc.set_title('(d) Global Predictive Accuracy', fontweight='bold', pad=8)
ax_acc.set_ylabel('Test Accuracy')
ax_acc.set_xlabel('Federation Size ($N$)')
ax_acc.set_ylim(0.80, 0.88)
ax_acc.set_xlim(3, 22)
ax_acc.set_xticks(x_indices)
ax_acc.set_xticklabels(x_labels)
ax_acc.grid(True, linestyle='--', alpha=0.5)
ax_acc.legend(loc='lower left', framealpha=0.9)

plt.suptitle(r'Federation Scaling on Adult Income ($20\%$ Attackers, Rank $r=4$, 30 Rounds, 5 Seeds)', 
             fontsize=12, fontweight='bold', y=0.99)
plt.tight_layout()

# Save outputs
fig_png_paper = os.path.join(out_paper_dir, "fig_federation_scaling.png")
fig_pdf_paper = os.path.join(out_paper_dir, "fig_federation_scaling.pdf")
fig_png_brain = os.path.join(out_brain_dir, "fig_federation_scaling.png")
fig_pdf_brain = os.path.join(out_brain_dir, "fig_federation_scaling.pdf")

fig.savefig(fig_png_paper, dpi=300, bbox_inches='tight')
fig.savefig(fig_pdf_paper, bbox_inches='tight')
fig.savefig(fig_png_brain, dpi=300, bbox_inches='tight')
fig.savefig(fig_pdf_brain, bbox_inches='tight')
plt.close()

# Also save standalone reproducible script to paper directory
with open("C:/Users/HP/.gemini/antigravity-ide/brain/18414c5f-eed7-43cb-9a90-f4a0cbab355b/scratch/build_scaling_assets.py", "r") as f_in:
    script_content = f_in.read()

with open(os.path.join(out_paper_dir, "plot_federation_scaling.py"), "w") as f_out:
    f_out.write(script_content)

print("Saved refined figures, tables, and scripts successfully!")
