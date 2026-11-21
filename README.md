# PRED: A Personalized Reference-Based Defense Against Fairness Attacks in Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**PRED** (Personalized Reference-based Exclusion Defense) is a novel defense mechanism against **fairness attacks** in federated learning. Adversarial clients can subtly inject bias into the global model by exploiting skewed local data distributions via LoRA-style low-rank perturbations. PRED detects and excludes such clients using a **rank-residual signal** computed against a small shared reference dataset, without requiring the server to know which clients are malicious.

### Key Contributions
- 🔍 **PRED-v1**: Rank-residual detection using a shared reference update `g0` to identify malicious clients
- 🔍 **PRED-v2**: Importance-weighted, client-specific reference updates to avoid false exclusions of honest clients with skewed data
- 🛡️ **Fairness Repair**: Post-aggregation fairness correction via constrained SGD on the reference set
- 📊 **Multi-dataset evaluation**: Adult, ACS, Bank, COMPAS, Dutch Census across multiple seeds and client scales (n=5, 10, 20)

---

## Project Structure

```
PRED/
├── attacks/
│   └── lora_attack.py          # LoRA-based fairness attack
├── clients/
│   └── honest.py               # Honest client local training
├── defenses/
│   ├── aggregators.py          # FedAvg, Krum, FLTrust aggregation
│   ├── dpg_fairfl.py           # DPG-FairFL defense baseline
│   ├── equfl.py                # EquFL defense baseline
│   ├── fed_influence.py        # FedInfluence defense (v1)
│   ├── fed_influence_v2.py     # FedInfluence defense (v2)
│   ├── guardfed.py             # GuardFed defense baseline
│   ├── metrics.py              # Fairness & accuracy metrics (DP gap, EO gap)
│   └── signal.py               # PRED rank-residual signal computation
├── models/
│   └── mlp.py                  # SimpleMLP model
├── data/
│   ├── preprocess.py           # Adult dataset preprocessing & FL partitioning
│   ├── preprocess_acs.py       # ACS dataset preprocessing
│   ├── preprocess_bank.py      # Bank Marketing dataset preprocessing
│   ├── preprocess_compas.py    # COMPAS dataset preprocessing
│   └── preprocess_dutch.py     # Dutch Census dataset preprocessing
├── experiments/
│   ├── runner.py               # Core FL experiment runner
│   ├── phase1_baseline.py      # Baseline (no attack)
│   ├── phase3_attack.py        # Attack evaluation
│   ├── phase4_krum.py          # Krum defense
│   ├── phase5_fltrust.py       # FLTrust defense
│   ├── phase6_sweep.py         # Hyperparameter sweep
│   ├── phase7_gate.py          # PRED gating signal analysis
│   ├── phase8_9_defense.py     # PRED defense evaluation
│   ├── phase10_final.py        # Final PRED evaluation
│   ├── phase11_scaleup.py      # Scale-up experiments (n=10, 20)
│   ├── pred_detector.py        # PRED-v1 detector
│   ├── pred_v2_full.py         # PRED-v2 full pipeline
│   ├── unified_comparison.py   # All-dataset, all-method comparison
│   ├── run_ablation_predv2.py  # LoRA rank ablation study
│   └── run_*_all6.py           # Per-dataset 6-method comparisons
├── paper/
│   └── fig_scaling_v2/         # Federation scaling figures
├── results/
│   └── figures/                # Paper figures (Pareto, stability, ablation)
└── architecture_diagram_advanced.pdf
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Datasets

Data is downloaded and preprocessed automatically on first run via OpenML / `folktables`. Run the preprocessing scripts manually if needed:

```bash
# Adult (UCI) - downloads via sklearn
python data/preprocess.py

# ACS (folktables)
python data/preprocess_acs.py

# Bank, COMPAS, Dutch
python data/preprocess_bank.py
python data/preprocess_compas.py
python data/preprocess_dutch.py
```

> **Note:** Preprocessed `.pt` files and raw CSVs are **not** committed to this repo (they are large binary files). Run the preprocessing scripts to generate them locally.

---

## Running Experiments

### 1. Baseline (no attack)
```bash
python experiments/phase1_baseline.py
```

### 2. Attack Evaluation (LoRA fairness attack)
```bash
python experiments/phase3_attack.py
```

### 3. Standard Defenses (Krum, FLTrust)
```bash
python experiments/phase4_krum.py
python experiments/phase5_fltrust.py
```

### 4. PRED Defense
```bash
# PRED-v1
python experiments/pred_detector.py

# PRED-v2 (recommended)
python experiments/pred_v2_full.py
```

### 5. Full 6-Method Comparison (all datasets)
```bash
python experiments/run_adult_all6.py
python experiments/run_acs_all6.py
python experiments/run_bank_all6.py
python experiments/run_compas_all6.py
python experiments/run_dutch_all6.py
```

### 6. Scale-Up & Ablation
```bash
python experiments/phase11_scaleup.py         # n=10, n=20 clients
python experiments/run_ablation_predv2.py     # LoRA rank ablation (r=2,4,8,16)
python experiments/run_rank_sweep.py          # Rank sweep analysis
```

---

## Methods Compared

| Method | Type | Description |
|--------|------|-------------|
| FedAvg | Baseline | Standard federated averaging |
| Krum | Defense | Byzantine-robust aggregation |
| FLTrust | Defense | Reference-based trust scoring |
| GuardFed | Defense | Gradient guard defense |
| EquFL | Defense | Equalized federated learning |
| **PRED-v1** | **Ours** | Rank-residual detection |
| **PRED-v2** | **Ours** | Personalized importance-weighted detection + fairness repair |

---

## Datasets

| Dataset | Task | Sensitive Attribute | # Samples |
|---------|------|---------------------|-----------|
| Adult (UCI) | Income prediction | Sex | ~48K |
| ACS (folktables) | Income prediction | Race | ~300K |
| Bank Marketing | Subscription prediction | Age | ~45K |
| COMPAS | Recidivism | Race | ~7K |
| Dutch Census | Occupation | Sex | ~60K |

---

## Fairness Metrics

- **DP Gap** (Demographic Parity): `|P(ŷ=1|A=0) - P(ŷ=1|A=1)|`
- **EO Gap** (Equalized Opportunity): `|TPR(A=0) - TPR(A=1)|`

Lower is fairer. PRED-v2 simultaneously reduces both while maintaining accuracy.

---



---

## License

MIT License. See [LICENSE](LICENSE) for details.
