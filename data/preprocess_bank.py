"""
Bank Marketing Dataset Loader
Loads UCI Bank Marketing dataset, preprocesses for federated learning fairness experiments.
Sensitive attribute: age (binarized per Zafar et al. 2017 convention)
Target: term deposit subscription (yes/no)
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_bank(cache_file="data/bank_preprocessed.pt"):
    """
    Load UCI Bank Marketing dataset, preprocess, and cache.
    Sensitive attribute: age binarized as young/old (<=25 or >60) vs middle-aged (25-60)
    following Zafar et al. 2017 convention.
    """
    if os.path.exists(cache_file):
        cached = torch.load(cache_file, weights_only=False)
        return cached['X'], cached['y'], cached['sensitive']
    
    print("Fetching Bank Marketing dataset...")
    
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="bank-marketing", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    
    # Drop missing values
    df = df.dropna().reset_index(drop=True)
    
    # Find target column
    # Bank Marketing target is typically 'y' or 'Class' (yes/no for term deposit)
    target_col = None
    for c in ['y', 'Y', 'Class', 'class', 'deposit']:
        if c in df.columns:
            target_col = c
            break
    
    if target_col is None:
        # Use the last column as target (common convention for OpenML)
        target_col = df.columns[-1]
    
    # Convert target to binary
    y_series = df[target_col].astype(str).str.strip().str.lower()
    if y_series.isin(['yes', 'no']).all() or y_series.isin(['1', '2']).any():
        # '1' is negative (no), '2' is positive (yes)
        y = y_series.apply(lambda v: 1 if v in ['yes', '2'] else 0).values.astype(np.int64)
    else:
        # Try numeric conversion
        y = pd.to_numeric(y_series, errors='coerce').fillna(0).astype(np.int64).values
    
    # Find age column
    age_col = None
    for c in ['age', 'Age', 'AGE', 'V1']:
        if c in df.columns:
            age_col = c
            break
    
    if age_col is None:
        raise ValueError(f"No age column found. Available columns: {list(df.columns)}")
    
    # Convert age to numeric
    age_values = pd.to_numeric(df[age_col], errors='coerce').fillna(df[age_col].median())
    
    # Binarize age per Zafar et al. 2017 convention:
    # Unprivileged (1): age <= 25 or age > 60 (young/old)
    # Privileged (0): 25 < age <= 60 (middle-aged)
    sensitive_attr = ((age_values <= 25) | (age_values > 60)).astype(int).values.astype(np.int64)
    
    # Features: everything except target
    features_df = df.drop(columns=[target_col])
    
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
    print(f"Cached Bank Marketing dataset to {cache_file}: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"  Class balance: {y.mean():.3f} positive rate")
    print(f"  Sensitive balance: {sensitive_attr.mean():.3f} young/old rate")
    
    return X, y, sensitive_attr


def prepare_bank_data(seed=42, n_clients=5, alpha=0.5, ref_size=250, save_dir="data"):
    """
    Prepare Bank Marketing data for federated learning. Identical interface to prepare_data().
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(save_dir, exist_ok=True)
    
    X, y, sensitive = load_and_preprocess_bank()
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
    X, y, s = load_and_preprocess_bank()
    print(f"Bank Marketing: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.3f}")
    print(f"  Young/old rate: {s.mean():.3f}")
