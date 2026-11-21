import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_adult(cache_file="data/adult_preprocessed.pt"):
    """
    Load UCI Adult dataset, preprocess, and cache to disk for fast multi-seed reuse.
    """
    if os.path.exists(cache_file):
        cached = torch.load(cache_file, weights_only=False)
        return cached['X'], cached['y'], cached['sensitive']
        
    print("Fetching UCI Adult dataset from OpenML...")
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = data.frame.dropna().reset_index(drop=True)
    
    label_col = "class"
    sensitive_col = "sex"
    
    y = df[label_col].apply(lambda v: 1 if ">50K" in str(v) else 0).values.astype(np.int64)
    sensitive_attr = df[sensitive_col].apply(lambda v: 1 if str(v).strip().lower() in ["male", "1", "m"] else 0).values.astype(np.int64)
    
    features_df = df.drop(columns=[label_col])
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(features_df[numeric_cols])
    scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)
    
    encoded_categorical_df = pd.get_dummies(features_df[categorical_cols], drop_first=True)
    
    X_df = pd.concat([scaled_numeric_df, encoded_categorical_df], axis=1)
    X = X_df.values.astype(np.float32)
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save({'X': X, 'y': y, 'sensitive': sensitive_attr}, cache_file)
    print(f"Cached dataset to {cache_file}: {X.shape[0]} samples, {X.shape[1]} features.")
    
    return X, y, sensitive_attr

def prepare_data(seed=42, n_clients=5, alpha=0.5, ref_size=250, save_dir="data"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(save_dir, exist_ok=True)
    
    X, y, sensitive = load_and_preprocess_adult()
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
    prepare_data(seed=42, n_clients=5, alpha=0.5, ref_size=250, save_dir="data")
