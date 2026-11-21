"""
COMPAS Recidivism Dataset Loader
Loads ProPublica COMPAS dataset, preprocesses for federated learning fairness experiments.
Sensitive attribute: race (African-American vs Caucasian)
Target: two_year_recid (binary)
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_compas(cache_file="data/compas_preprocessed.pt"):
    """
    Load COMPAS recidivism dataset, preprocess, and cache.
    Uses the ProPublica COMPAS dataset filtered to African-American and Caucasian defendants.
    """
    if os.path.exists(cache_file):
        cached = torch.load(cache_file, weights_only=False)
        return cached['X'], cached['y'], cached['sensitive']
    
    print("Fetching COMPAS dataset...")
    
    # Try loading from OpenML first
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name="compas-two-years", version=1, as_frame=True, parser="auto")
        df = data.frame.copy()
    except Exception:
        # Fallback: download from ProPublica GitHub
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        df = pd.read_csv(url)
    
    # Standard COMPAS preprocessing (following ProPublica methodology)
    # Filter to relevant columns
    cols_to_keep = [
        'age', 'c_charge_degree', 'race', 'sex', 'priors_count',
        'days_b_screening_arrest', 'two_year_recid', 'juv_fel_count',
        'juv_misd_count', 'juv_other_count'
    ]
    
    # Keep only columns that exist
    cols_available = [c for c in cols_to_keep if c in df.columns]
    df = df[cols_available].copy()
    
    # Drop missing values
    df = df.dropna().reset_index(drop=True)
    
    # Filter to African-American and Caucasian only (standard fairness convention)
    if 'race' in df.columns:
        race_col = 'race'
    else:
        # Try alternative column names
        for c in df.columns:
            if 'race' in c.lower():
                race_col = c
                break
        else:
            raise ValueError("No race column found in COMPAS dataset")
    
    # Standardize race values
    df[race_col] = df[race_col].astype(str).str.strip()
    
    # Filter to binary race groups
    aa_mask = df[race_col].str.contains('African', case=False, na=False)
    cauc_mask = df[race_col].str.contains('Caucasian', case=False, na=False)
    df = df[aa_mask | cauc_mask].reset_index(drop=True)
    
    if len(df) < 100:
        raise ValueError(f"Too few samples after race filtering: {len(df)}")
    
    # Target: two_year_recid
    if 'two_year_recid' in df.columns:
        y_col = 'two_year_recid'
    else:
        # Try alternative names
        for c in df.columns:
            if 'recid' in c.lower():
                y_col = c
                break
        else:
            raise ValueError("No recidivism column found")
    
    y = df[y_col].astype(int).values.astype(np.int64)
    
    # Sensitive attribute: race (African-American = 1, Caucasian = 0)
    sensitive_attr = aa_mask[df.index].astype(int).values.astype(np.int64)
    
    # Features: everything except target
    features_df = df.drop(columns=[y_col])
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Standard-scale numerics
    if numeric_cols:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(features_df[numeric_cols])
        scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)
    else:
        scaled_numeric_df = pd.DataFrame()
    
    # One-hot encode categoricals
    if categorical_cols:
        encoded_categorical_df = pd.get_dummies(features_df[categorical_cols], drop_first=True)
    else:
        encoded_categorical_df = pd.DataFrame()
    
    X_df = pd.concat([scaled_numeric_df, encoded_categorical_df], axis=1)
    X = X_df.values.astype(np.float32)
    
    os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
    torch.save({'X': X, 'y': y, 'sensitive': sensitive_attr}, cache_file)
    print(f"Cached COMPAS dataset to {cache_file}: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"  Class balance: {y.mean():.3f} positive rate")
    print(f"  Sensitive balance: {sensitive_attr.mean():.3f} African-American rate")
    
    return X, y, sensitive_attr


def prepare_compas_data(seed=42, n_clients=5, alpha=0.5, ref_size=250, save_dir="data"):
    """
    Prepare COMPAS data for federated learning. Identical interface to prepare_data().
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(save_dir, exist_ok=True)
    
    X, y, sensitive = load_and_preprocess_compas()
    n_samples = len(y)
    
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(n_samples)
    
    n_test = int(0.20 * n_samples)
    test_idx = shuffled_indices[:n_test]
    train_pool_idx = shuffled_indices[n_test:]
    
    ref_idx = train_pool_idx[:ref_size]
    client_pool_idx = train_pool_idx[ref_size:]
    
    test_data = {
        'x': torch.tensor(X[test_idx], dtype=torch.float32),
        'y': torch.tensor(y[test_idx], dtype=torch.long),
        'sensitive': torch.tensor(sensitive[test_idx], dtype=torch.long)
    }
    torch.save(test_data, os.path.join(save_dir, "test.pt"))
    
    ref_data = {
        'x': torch.tensor(X[ref_idx], dtype=torch.float32),
        'y': torch.tensor(y[ref_idx], dtype=torch.long),
        'sensitive': torch.tensor(sensitive[ref_idx], dtype=torch.long)
    }
    torch.save(ref_data, os.path.join(save_dir, "reference.pt"))
    
    y_pool = y[client_pool_idx]
    client_indices = [[] for _ in range(n_clients)]
    
    for label in [0, 1]:
        label_mask = (y_pool == label)
        label_idx = client_pool_idx[label_mask]
        rng.shuffle(label_idx)
        
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = (np.cumsum(proportions) * len(label_idx)).astype(int)[:-1]
        
        splits = np.split(label_idx, proportions)
        for i in range(n_clients):
            client_indices[i].extend(splits[i])
    
    for i in range(n_clients):
        c_idx = np.array(client_indices[i])
        rng.shuffle(c_idx)
        c_data = {
            'x': torch.tensor(X[c_idx], dtype=torch.float32),
            'y': torch.tensor(y[c_idx], dtype=torch.long),
            'sensitive': torch.tensor(sensitive[c_idx], dtype=torch.long)
        }
        torch.save(c_data, os.path.join(save_dir, f"client_{i}.pt"))


if __name__ == "__main__":
    X, y, s = load_and_preprocess_compas()
    print(f"COMPAS: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.3f}")
    print(f"  African-American rate: {s.mean():.3f}")
