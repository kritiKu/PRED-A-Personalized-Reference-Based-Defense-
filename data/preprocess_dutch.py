"""
Dutch Census Dataset Loader
Loads the Dutch Census 2001 dataset, preprocesses for federated learning fairness experiments.
Sensitive attribute: sex (male/female)
Target: occupation (high-level vs low-level)
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_dutch(cache_file="data/dutch_preprocessed.pt"):
    """
    Load Dutch Census dataset, preprocess, and cache.
    Sensitive attribute: sex (male = 1, female = 0, per common convention or vice versa, we'll map male to 1)
    """
    if os.path.exists(cache_file):
        cached = torch.load(cache_file, weights_only=False)
        return cached['X'], cached['y'], cached['sensitive']
    
    print("Fetching Dutch Census dataset...")
    
    url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/dutch.csv"
    df = pd.read_csv(url)
    
    # Drop missing values
    df = df.dropna().reset_index(drop=True)
    
    target_col = 'occupation'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {df.columns}")
        
    y = df[target_col].astype(np.int64).values
    
    sensitive_col = 'sex'
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found. Columns: {df.columns}")
        
    # In this dataset, sex is 'male' / 'female'
    # Map male to 1 (privileged typically in this dataset), female to 0
    sensitive_attr = (df[sensitive_col].str.strip().str.lower() == 'male').astype(np.int64).values
    
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
    # Convert boolean columns from get_dummies to float32
    X = X_df.astype(np.float32).values
    
    os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
    torch.save({'X': X, 'y': y, 'sensitive': sensitive_attr}, cache_file)
    print(f"Cached Dutch Census dataset to {cache_file}: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"  Class balance: {y.mean():.3f} positive rate")
    print(f"  Sensitive balance: {sensitive_attr.mean():.3f} male rate")
    
    return X, y, sensitive_attr


def prepare_dutch_data(seed=42, n_clients=5, alpha=0.5, ref_size=250, save_dir="data"):
    """
    Prepare Dutch Census data for federated learning. Identical interface to prepare_data().
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(save_dir, exist_ok=True)
    
    X, y, sensitive = load_and_preprocess_dutch()
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
    X, y, s = load_and_preprocess_dutch()
    print(f"Dutch Census: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.3f}")
    print(f"  Male rate: {s.mean():.3f}")
