import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from folktables import ACSDataSource, ACSIncome

def load_and_preprocess_acs(cache_file="data/acs_preprocessed.pt"):
    """
    Load ACSIncome dataset (CA, 2018), preprocess, and cache.
    Sensitive attribute: SEX (Male=1, Female=0 or vice versa)
    Target: Income > $50,000 (PINCP > 50000)
    Filters: Age > 16, WKHP > 0, PINCP > 100
    """
    if os.path.exists(cache_file):
        cached = torch.load(cache_file, weights_only=False)
        return cached['X'], cached['y'], cached['sensitive']
    
    print("Fetching ACSIncome dataset (CA 2018)...")
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    
    # Filter as per standard ACSIncome task (these are built into ACSIncome task usually, but we will ensure)
    # The default ACSIncome task handles the filtering automatically via ACSIncome.df_to_pandas
    features, label, _ = ACSIncome.df_to_pandas(acs_data)
    
    y = label['PINCP'].astype(np.int64).values
    
    # Sex is in the features dataframe
    # Typically 1 = Male, 2 = Female in ACS. We will map Male -> 1, Female -> 0
    sensitive_col = 'SEX'
    if sensitive_col not in features.columns:
        raise ValueError("SEX not found in features.")
        
    sensitive_attr = (features[sensitive_col] == 1).astype(np.int64).values
    
    # Drop the sensitive column from features if desired, or keep it.
    # We usually keep the sensitive attribute in the features unless specified, but let's just 
    # encode it properly.
    features_df = features.copy()
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numeric_cols:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(features_df[numeric_cols])
        scaled_numeric_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)
    else:
        scaled_numeric_df = pd.DataFrame()
        
    if categorical_cols:
        encoded_categorical_df = pd.get_dummies(features_df[categorical_cols], drop_first=True)
    else:
        encoded_categorical_df = pd.DataFrame()
        
    X_df = pd.concat([scaled_numeric_df, encoded_categorical_df], axis=1)
    X = X_df.astype(np.float32).values
    
    os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
    torch.save({'X': X, 'y': y, 'sensitive': sensitive_attr}, cache_file)
    print(f"Cached ACSIncome dataset to {cache_file}: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"  Class balance: {y.mean():.3f} positive rate")
    print(f"  Sensitive balance: {sensitive_attr.mean():.3f} male rate")
    
    return X, y, sensitive_attr


def prepare_acs_data(seed=42, n_clients=5, alpha=0.5, ref_size=250, save_dir="data"):
    """
    Prepare ACSIncome data for federated learning.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.makedirs(save_dir, exist_ok=True)
    
    X, y, sensitive = load_and_preprocess_acs()
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
    X, y, s = load_and_preprocess_acs()
    print(f"Full ACS Dataset: {len(y)} samples | Pos={y.mean():.4f} | M={s.mean():.4f}")
    
    # Run diagnostic across seeds
    for seed in [43, 44, 45, 46, 47]:
        save_dir = f"data/seed_{seed}_acs_n5"
        prepare_acs_data(seed, 5, 0.5, 250, save_dir)
        
        ref = torch.load(f"{save_dir}/reference.pt", weights_only=False)
        ry, rs = ref['y'].numpy(), ref['sensitive'].numpy()
        print(f"\nSeed {seed} | Ref Set: Pos={ry.mean():.4f} (M_Pos={ry[rs==1].mean() if sum(rs==1)>0 else 0:.4f}, F_Pos={ry[rs==0].mean() if sum(rs==0)>0 else 0:.4f})")
        
        for i in range(5):
            cd = torch.load(f"{save_dir}/client_{i}.pt", weights_only=False)
            cy = cd['y'].numpy()
            pos = cy.mean()
            warning = "*** OUTLIER ***" if pos < 0.05 or pos > 0.95 else ""
            print(f"  Client {i} (N={len(cy)}): PosRate={pos:.4f} {warning}")
