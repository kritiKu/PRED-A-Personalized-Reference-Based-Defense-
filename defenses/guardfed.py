"""
GuardFed Defense (arXiv 2511.09294, Li et al.)
Dual-perspective trust scoring: Fairness Index + Deviation Index.
"""
import torch
import torch.nn as nn
import numpy as np

def _train_fair_reference(model_class, input_dim, gw, ref_data, steps=200, lr=0.01, seed=0):
    """
    Train a fairness-aware reference model using reweighted loss.
    Reweighting: w_i = 1 / (P(a_i) * P(y_i)) to make a ⊥ y.
    (GuardFed paper Section V-B, Eq. for w_i)
    """
    torch.manual_seed(seed)
    rm = model_class(input_dim)
    rm.load_state_dict(gw)
    opt = torch.optim.SGD(rm.parameters(), lr=lr)
    
    rx, ry, rs = ref_data['x'], ref_data['y'], ref_data['sensitive']
    nr = len(rx)
    
    # Compute reweighting: w_i = 1 / (P(a_i) * P(y_i))
    # This makes sensitive attribute independent of label
    pa1 = (rs == 1).float().mean().item()
    pa0 = 1.0 - pa1
    py1 = (ry == 1).float().mean().item()
    py0 = 1.0 - py1
    
    weights = torch.ones(nr)
    for j in range(nr):
        pa = pa1 if rs[j].item() == 1 else pa0
        py = py1 if ry[j].item() == 1 else py0
        weights[j] = 1.0 / (max(pa, 0.01) * max(py, 0.01))
    weights = weights / weights.sum() * nr  # normalize to sum=nr
    
    crit = nn.CrossEntropyLoss(reduction='none')
    gen = torch.Generator().manual_seed(seed)
    
    step = 0
    while step < steps:
        perm = torch.randperm(nr, generator=gen)
        for s in range(0, nr, 32):
            if step >= steps:
                break
            idx = perm[s:s+32]
            bw = weights[idx]
            bw = bw / (bw.sum() + 1e-8)
            opt.zero_grad()
            loss = (crit(rm(rx[idx]), ry[idx]) * bw).sum()
            loss.backward()
            opt.step()
            step += 1
    
    return rm


def _compute_fairness_index(model, ref_data):
    """
    Fairness Index: |P(ŷ=1|a=0,y=1) - P(ŷ=1|a=1,y=1)|
    (GuardFed paper Section V-B, Eq. for Fair_n)
    Measures group disparity in true positive rates.
    """
    model.eval()
    with torch.no_grad():
        rx, ry, rs = ref_data['x'], ref_data['y'], ref_data['sensitive']
        preds = torch.argmax(model(rx), dim=1)
        
        # TPR for each sensitive group
        y1_a1 = (ry == 1) & (rs == 1)
        y1_a0 = (ry == 1) & (rs == 0)
        
        tpr_a1 = (preds[y1_a1] == 1).float().mean().item() if y1_a1.sum() > 0 else 0.0
        tpr_a0 = (preds[y1_a0] == 1).float().mean().item() if y1_a0.sum() > 0 else 0.0
        
        return abs(tpr_a1 - tpr_a0)


def guardfed_aggregate(model_class, input_dim, gw, updates, sizes, ref_data,
                       tau=5.0, gamma_threshold=0.1, ref_steps=200, seed=0):
    """
    GuardFed aggregation (paper Algorithm 1):
    1. Train fairness-aware reference model
    2. Compute trust score for each client
    3. Aggregate only clients with Trust > gamma_threshold
    
    Returns: (new_gw, excluded_list)
    """
    n = len(updates)
    
    # Step 1: Train fairness-aware reference model
    ref_model = _train_fair_reference(model_class, input_dim, gw, ref_data,
                                       steps=ref_steps, seed=seed)
    
    # Compute reference update vector
    gv = torch.cat([p.data.view(-1) for p in gw.values()])
    rv = torch.cat([p.data.view(-1) for p in ref_model.parameters()])
    g_ref = rv - gv  # reference update direction
    
    trust_scores = []
    
    for i in range(n):
        u = updates[i]
        
        # Step 2a: Deviation Index = ReLU(cos_sim(Δ_i, Δ_ref))
        cos_sim = torch.dot(u, g_ref) / (torch.norm(u) * torch.norm(g_ref) + 1e-9)
        dev_i = max(0.0, cos_sim.item())
        
        # Step 2b: Fairness Index — evaluate client's model on reference data
        client_weights = {}
        offset = 0
        for key, par in gw.items():
            numel = par.numel()
            client_weights[key] = par + u[offset:offset+numel].view_as(par)
            offset += numel
        
        client_model = model_class(input_dim)
        client_model.load_state_dict(client_weights)
        fair_i = _compute_fairness_index(client_model, ref_data)
        
        # Step 2c: Trust Score = Dev * exp(-τ * Fair)
        trust_i = dev_i * np.exp(-tau * fair_i)
        trust_scores.append(trust_i)
    
    # Step 3: Select clients with Trust > threshold
    excluded = [i for i in range(n) if trust_scores[i] <= gamma_threshold]
    included = [i for i in range(n) if trust_scores[i] > gamma_threshold]
    
    if not included:
        # Fallback: include all to avoid empty aggregation
        included = list(range(n))
        excluded = []
    
    # FedAvg over included clients
    inc_updates = [updates[i] for i in included]
    inc_sizes = [sizes[i] for i in included]
    total = sum(inc_sizes)
    avg = sum((s / total) * u for u, s in zip(inc_updates, inc_sizes))
    
    new_gw = {}
    offset = 0
    for key, par in gw.items():
        numel = par.numel()
        new_gw[key] = par + avg[offset:offset+numel].view_as(par)
        offset += numel
    
    return new_gw, excluded
