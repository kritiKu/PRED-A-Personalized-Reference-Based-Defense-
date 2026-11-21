import os, ast, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = "C:/Users/HP/Desktop/pred/paper/fig_scaling_v2"
base_dir = "C:/Users/HP/Desktop/pred"
os.makedirs(out_dir, exist_ok=True)

scales = [5, 10, 20]
seeds = [43, 44, 45, 46, 47]
configs = {
    5: {'att': [3], 'honest': [0, 1, 2, 4]},
    10: {'att': [3, 8], 'honest': [0, 1, 2, 4, 5, 6, 7, 9]},
    20: {'att': [3, 8, 13, 18], 'honest': [i for i in range(20) if i not in [3, 8, 13, 18]]}
}

# 1. Gather all raw per-seed data
raw_rows = []
for nc in scales:
    att = configs[nc]['att']
    honest = configs[nc]['honest']
    folder = f"{base_dir}/logs/unified_n{nc}"
    
    for s in seeds:
        # PRED
        p_path = f"{folder}/pred-v2_seed{s}.csv"
        p_df = pd.read_csv(p_path)
        p_last = p_df.iloc[-1]
        
        tp_s, fp_s = 0, 0
        for excl_str in p_df['excluded']:
            excl = ast.literal_eval(excl_str)
            for c in excl:
                if c in att: tp_s += 1
                elif c in honest: fp_s += 1
        tpr_s = tp_s / (len(att) * 30)
        fpr_s = fp_s / (len(honest) * 30)
        
        raw_rows.append({
            'N': nc, 'Strategy': 'PRED', 'Seed': s,
            'Accuracy': float(p_last['accuracy']),
            'Delta_DP': float(p_last['dp_gap']),
            'Delta_EO': float(p_last['eo_gap']),
            'TPR': tpr_s * 100.0, # percentage
            'FPR': fpr_s * 100.0,
            'TP_count': tp_s,
            'FP_count': fp_s,
            'Source': p_path
        })
        
        # FedAvg
        f_path = f"{folder}/fedavg_seed{s}.csv"
        f_df = pd.read_csv(f_path)
        f_last = f_df.iloc[-1]
        raw_rows.append({
            'N': nc, 'Strategy': 'FedAvg', 'Seed': s,
            'Accuracy': float(f_last['accuracy']),
            'Delta_DP': float(f_last['dp_gap']),
            'Delta_EO': float(f_last['eo_gap']),
            'TPR': np.nan,
            'FPR': np.nan,
            'TP_count': np.nan,
            'FP_count': np.nan,
            'Source': f_path
        })

df_raw = pd.DataFrame(raw_rows)

# 2. Save fig_scaling_table.csv
summary_rows = []
for nc in scales:
    for strat in ['PRED', 'FedAvg']:
        sub = df_raw[(df_raw['N'] == nc) & (df_raw['Strategy'] == strat)]
        summary_rows.append({
            'N': nc,
            'Strategy': strat,
            'Acc_mean': sub['Accuracy'].mean(),
            'Acc_std_ddof1': sub['Accuracy'].std(ddof=1),
            'Acc_std_ddof0': sub['Accuracy'].std(ddof=0),
            'DP_mean': sub['Delta_DP'].mean(),
            'DP_std_ddof1': sub['Delta_DP'].std(ddof=1),
            'DP_std_ddof0': sub['Delta_DP'].std(ddof=0),
            'EO_mean': sub['Delta_EO'].mean(),
            'EO_std_ddof1': sub['Delta_EO'].std(ddof=1),
            'EO_std_ddof0': sub['Delta_EO'].std(ddof=0),
            'TPR_mean': sub['TPR'].mean() if strat == 'PRED' else np.nan,
            'TPR_std_ddof1': sub['TPR'].std(ddof=1) if strat == 'PRED' else np.nan,
            'TPR_std_ddof0': sub['TPR'].std(ddof=0) if strat == 'PRED' else np.nan,
            'TPR_pooled': (sub['TP_count'].sum() / (len(configs[nc]['att']) * 30 * 5)) * 100.0 if strat == 'PRED' else np.nan,
            'FP_total': sub['FP_count'].sum() if strat == 'PRED' else np.nan
        })

df_summary = pd.DataFrame(summary_rows)
table_path = f"{out_dir}/fig_scaling_table.csv"
df_summary.to_csv(table_path, index=False)
print(f"Saved {table_path}")

# 3. Plotting IEEE One Column Specification
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['figure.titlesize'] = 7

fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.2), layout='constrained')

c_pred = '#0055aa'     # Strong clear blue
c_fed = '#555555'      # Charcoal grey
m_pred = 'o'           # Filled circle
m_fed = '^'            # Open triangle

x_map = {5: 0, 10: 1, 20: 2}
x_ticks = [0, 1, 2]
x_labels = ['5', '10', '20']
x_lims = (-0.45, 2.45)

# Jitter offsets
jit_pred_mean = -0.06
jit_fed_mean = 0.06
jit_seeds_p = np.array([-0.13, -0.095, -0.06, -0.025, 0.01])
jit_seeds_f = np.array([0.01, 0.045, 0.08, 0.115, 0.15])

# Panel (a): TPR (%)
ax_a = axes[0, 0]
for nc in scales:
    xi = x_map[nc]
    sub = df_raw[(df_raw['N'] == nc) & (df_raw['Strategy'] == 'PRED')]
    vals = sub['TPR'].values
    mean_val = vals.mean()
    sd_val = vals.std(ddof=1)
    
    # Clip upper error bar at 100
    yerr_lower = sd_val
    yerr_upper = min(sd_val, max(0.0, 100.0 - mean_val))
    yerr = [[yerr_lower], [yerr_upper]]
    
    ax_a.scatter([xi + j for j in jit_seeds_p], vals, color=c_pred, alpha=0.35, s=7, edgecolors='none', zorder=2)
    ax_a.errorbar([xi + jit_pred_mean], [mean_val], yerr=yerr, fmt=m_pred, color=c_pred,
                 ecolor=c_pred, elinewidth=0.8, capsize=2.0, capthick=0.8, markersize=3.5, zorder=3)

ax_a.set_xlim(x_lims)
ax_a.set_ylim(75, 101)
ax_a.set_ylabel('TPR (%)', fontsize=7)
ax_a.text(0.04, 0.94, '(a)', transform=ax_a.transAxes, fontweight='bold', fontsize=7, va='top')
ax_a.set_xticks(x_ticks)
ax_a.set_xticklabels(x_labels)
ax_a.grid(True, linestyle=':', alpha=0.4)

# Panel (b): Delta DP
ax_b = axes[0, 1]
for nc in scales:
    xi = x_map[nc]
    # PRED
    sub_p = df_raw[(df_raw['N'] == nc) & (df_raw['Strategy'] == 'PRED')]
    vals_p = sub_p['Delta_DP'].values
    m_p, s_p = vals_p.mean(), vals_p.std(ddof=1)
    ax_b.scatter([xi + j for j in jit_seeds_p], vals_p, color=c_pred, alpha=0.35, s=7, edgecolors='none', zorder=2)
    ax_b.errorbar([xi + jit_pred_mean], [m_p], yerr=[[s_p], [s_p]], fmt=m_pred, color=c_pred,
                 ecolor=c_pred, elinewidth=0.8, capsize=2.0, capthick=0.8, markersize=3.5, label='PRED' if nc == 5 else "", zorder=3)
    
    # FedAvg
    sub_f = df_raw[(df_raw['N'] == nc) & (df_raw['Strategy'] == 'FedAvg')]
    vals_f = sub_f['Delta_DP'].values
    m_f, s_f = vals_f.mean(), vals_f.std(ddof=1)
    ax_b.scatter([xi + j for j in jit_seeds_f], vals_f, color=c_fed, alpha=0.35, s=7, edgecolors='none', zorder=2)
    ax_b.errorbar([xi + jit_fed_mean], [m_f], yerr=[[s_f], [s_f]], fmt=m_fed, color=c_fed, mfc='none', mec=c_fed,
                 ecolor=c_fed, elinewidth=0.8, capsize=2.0, capthick=0.8, markersize=3.5, label='FedAvg' if nc == 5 else "", zorder=3)

ax_b.set_xlim(x_lims)
ax_b.set_ylim(0.0, 0.32)
ax_b.set_ylabel(r'$\Delta$DP', fontsize=7)
ax_b.text(0.04, 0.94, '(b)', transform=ax_b.transAxes, fontweight='bold', fontsize=7, va='top')
ax_b.set_xticks(x_ticks)
ax_b.set_xticklabels(x_labels)
ax_b.grid(True, linestyle=':', alpha=0.4)
ax_b.legend(loc='upper right', frameon=False, fontsize=6, borderpad=0.2, handletextpad=0.2)

# Panel (c): Delta EO
ax_c = axes[1, 0]
for nc in scales:
    xi = x_map[nc]
    # PRED
    sub_p = df_raw[(df_raw['N'] == nc) & (df_raw['Strategy'] == 'PRED')]
    vals_p = sub_p['Delta_EO'].values
    m_p, s_p = vals_p.mean(), vals_p.std(ddof=1)
    ax_c.scatter([xi + j for j in jit_seeds_p], vals_p, color=c_pred, alpha=0.35, s=7, edgecolors='none', zorder=2)
    ax_c.errorbar([xi + jit_pred_mean], [m_p], yerr=[[s_p], [s_p]], fmt=m_pred, color=c_pred,
                 ecolor=c_pred, elinewidth=0.8, capsize=2.0, capthick=0.8, markersize=3.5, zorder=3)
    
    # FedAvg
    sub_f = df_raw[(df_raw['N'] == nc) & (df_raw['Strategy'] == 'FedAvg')]
    vals_f = sub_f['Delta_EO'].values
    m_f, s_f = vals_f.mean(), vals_f.std(ddof=1)
    ax_c.scatter([xi + j for j in jit_seeds_f], vals_f, color=c_fed, alpha=0.35, s=7, edgecolors='none', zorder=2)
    ax_c.errorbar([xi + jit_fed_mean], [m_f], yerr=[[s_f], [s_f]], fmt=m_fed, color=c_fed, mfc='none', mec=c_fed,
                 ecolor=c_fed, elinewidth=0.8, capsize=2.0, capthick=0.8, markersize=3.5, zorder=3)

ax_c.set_xlim(x_lims)
ax_c.set_ylim(0.0, 0.20)
ax_c.set_ylabel(r'$\Delta$EO', fontsize=7)
ax_c.set_xlabel('N', fontsize=7)
ax_c.text(0.04, 0.94, '(c)', transform=ax_c.transAxes, fontweight='bold', fontsize=7, va='top')
ax_c.set_xticks(x_ticks)
ax_c.set_xticklabels(x_labels)
ax_c.grid(True, linestyle=':', alpha=0.4)

# Panel (d): Accuracy
ax_d = axes[1, 1]
for nc in scales:
    xi = x_map[nc]
    # PRED
    sub_p = df_raw[(df_raw['N'] == nc) & (df_raw['Strategy'] == 'PRED')]
    vals_p = sub_p['Accuracy'].values
    m_p, s_p = vals_p.mean(), vals_p.std(ddof=1)
    ax_d.scatter([xi + j for j in jit_seeds_p], vals_p, color=c_pred, alpha=0.35, s=7, edgecolors='none', zorder=2)
    ax_d.errorbar([xi + jit_pred_mean], [m_p], yerr=[[s_p], [s_p]], fmt=m_pred, color=c_pred,
                 ecolor=c_pred, elinewidth=0.8, capsize=2.0, capthick=0.8, markersize=3.5, zorder=3)
    
    # FedAvg
    sub_f = df_raw[(df_raw['N'] == nc) & (df_raw['Strategy'] == 'FedAvg')]
    vals_f = sub_f['Accuracy'].values
    m_f, s_f = vals_f.mean(), vals_f.std(ddof=1)
    ax_d.scatter([xi + j for j in jit_seeds_f], vals_f, color=c_fed, alpha=0.35, s=7, edgecolors='none', zorder=2)
    ax_d.errorbar([xi + jit_fed_mean], [m_f], yerr=[[s_f], [s_f]], fmt=m_fed, color=c_fed, mfc='none', mec=c_fed,
                 ecolor=c_fed, elinewidth=0.8, capsize=2.0, capthick=0.8, markersize=3.5, zorder=3)

ax_d.set_xlim(x_lims)
ax_d.set_ylim(0.79, 0.88)
ax_d.set_ylabel('Accuracy', fontsize=7)
ax_d.set_xlabel('N', fontsize=7)
ax_d.text(0.04, 0.94, '(d)', transform=ax_d.transAxes, fontweight='bold', fontsize=7, va='top')
ax_d.set_xticks(x_ticks)
ax_d.set_xticklabels(x_labels)
ax_d.grid(True, linestyle=':', alpha=0.4)

pdf_path = f"{out_dir}/fig_scaling.pdf"
png_path = f"{out_dir}/fig_scaling.png"
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
plt.close()

print(f"Saved {pdf_path}")
print(f"Saved {png_path}")

# Save the reproducible script into fig_scaling_v2 as plot_fig_scaling.py
with open("C:/Users/HP/.gemini/antigravity-ide/brain/18414c5f-eed7-43cb-9a90-f4a0cbab355b/scratch/generate_scaling_v2.py", "r") as f_in:
    code_body = f_in.read()

with open(f"{out_dir}/plot_fig_scaling.py", "w") as f_out:
    f_out.write(code_body)
print(f"Saved {out_dir}/plot_fig_scaling.py")
