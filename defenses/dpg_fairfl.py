import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Tabular Conditional GAN
# ─────────────────────────────────────────────────────────────────────────────

class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim, input_dim):
        super().__init__()
        # Input: z (noise) + y (label) + s (sensitive)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, z, y, s):
        y_enc = y.view(-1, 1).float()
        s_enc = s.view(-1, 1).float()
        x = torch.cat([z, y_enc, s_enc], dim=1)
        return self.net(x)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Input: x (features) + y (label) + s (sensitive)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, y, s):
        y_enc = y.view(-1, 1).float()
        s_enc = s.view(-1, 1).float()
        features = torch.cat([x, y_enc, s_enc], dim=1)
        return self.net(features)


def train_tabular_cgan(ref_data, input_dim, z_dim=16, epochs=500, lr=0.001, seed=42):
    """
    Trains a lightweight conditional GAN on the server's reference dataset.
    """
    torch.manual_seed(seed)
    generator = ConditionalGenerator(z_dim, input_dim)
    discriminator = ConditionalDiscriminator(input_dim)
    
    opt_G = optim.Adam(generator.parameters(), lr=lr)
    opt_D = optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    x_real = ref_data['x']
    y_real = ref_data['y']
    s_real = ref_data['sensitive']
    n_samples = len(x_real)
    batch_size = 64
    
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            b_x = x_real[idx]
            b_y = y_real[idx]
            b_s = s_real[idx]
            b_size = len(idx)
            
            # Train Discriminator
            opt_D.zero_grad()
            
            # Real loss
            real_preds = discriminator(b_x, b_y, b_s)
            d_loss_real = criterion(real_preds, torch.ones_like(real_preds))
            
            # Fake loss
            z = torch.randn(b_size, z_dim)
            fake_x = generator(z, b_y, b_s)
            fake_preds = discriminator(fake_x.detach(), b_y, b_s)
            d_loss_fake = criterion(fake_preds, torch.zeros_like(fake_preds))
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_D.step()
            
            # Train Generator
            opt_G.zero_grad()
            fake_preds_for_G = discriminator(fake_x, b_y, b_s)
            g_loss = criterion(fake_preds_for_G, torch.ones_like(fake_preds_for_G))
            g_loss.backward()
            opt_G.step()
            
    return generator


def generate_balanced_synthetic_data(generator, z_dim=16, n_per_group=100, seed=0):
    """
    Generates a perfectly balanced synthetic dataset covering all 4 intersectional groups.
    """
    torch.manual_seed(seed)
    total_samples = n_per_group * 4
    
    y_syn = torch.cat([
        torch.zeros(n_per_group * 2), 
        torch.ones(n_per_group * 2)
    ]).long()
    
    s_syn = torch.cat([
        torch.zeros(n_per_group), torch.ones(n_per_group),
        torch.zeros(n_per_group), torch.ones(n_per_group)
    ]).long()
    
    z = torch.randn(total_samples, z_dim)
    
    with torch.no_grad():
        generator.eval()
        x_syn = generator(z, y_syn, s_syn)
        
    return {'x': x_syn, 'y': y_syn, 'sensitive': s_syn}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Fairness-Gradient Down-weighting
# ─────────────────────────────────────────────────────────────────────────────

def dpg_fairfl_aggregate(model_class, input_dim, global_weights, updates, sizes, ref_data, 
                         generator=None, gamma=5.0, z_dim=16, seed=42):
    """
    DPG-FairFL Defense (2025).
    1. If generator is None, trains the tabular cGAN on ref_data.
    2. Generates balanced synthetic dataset.
    3. Evaluates each client update's Demographic Parity on the synthetic set.
    4. Down-weights updates that cause large DP gaps.
    """
    if generator is None:
        generator = train_tabular_cgan(ref_data, input_dim, z_dim=z_dim, epochs=500, lr=0.001, seed=seed)
        
    syn_data = generate_balanced_synthetic_data(generator, z_dim=z_dim, n_per_group=200, seed=seed+1)
    syn_x = syn_data['x']
    syn_s = syn_data['sensitive']
    
    n_clients = len(updates)
    dp_gaps = []
    
    for i in range(n_clients):
        # Create tentative global model from this update
        temp_gw = {}
        offset = 0
        for key, par in global_weights.items():
            numel = par.numel()
            temp_gw[key] = par + updates[i][offset:offset+numel].view_as(par)
            offset += numel
            
        m = model_class(input_dim)
        m.load_state_dict(temp_gw)
        m.eval()
        
        with torch.no_grad():
            out = m(syn_x)
            probs = torch.softmax(out, dim=1)[:, 1]
            
            p_male = probs[syn_s == 1].mean().item() if (syn_s == 1).sum() > 0 else probs.mean().item()
            p_female = probs[syn_s == 0].mean().item() if (syn_s == 0).sum() > 0 else probs.mean().item()
            
            dp_gap = abs(p_male - p_female)
            dp_gaps.append(dp_gap)
            
    # Softmax/Exponential down-weighting
    dp_tensor = torch.tensor(dp_gaps, dtype=torch.float32)
    weights = torch.exp(-gamma * dp_tensor)
    weights = weights / weights.sum()  # Normalize to 1
    
    # Aggregate updates
    agg_update = sum(w.item() * u for w, u in zip(weights, updates))
    
    new_gw = {}
    offset = 0
    for key, par in global_weights.items():
        numel = par.numel()
        new_gw[key] = par + agg_update[offset:offset+numel].view_as(par)
        offset += numel
        
    return new_gw, generator
