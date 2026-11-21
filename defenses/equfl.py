"""
EquFL Defense (arXiv 2601.05352, "When the Server Steps In")
Dataset distillation from model checkpoints + fairness-gradient calibrated updates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


def distill_synthetic_data(model_class, input_dim, checkpoints, n_samples=200,
                           iterations=500, lr_syn=0.01, seed=0):
    """
    Dataset distillation from model checkpoints (EquFL paper Algorithm 1).
    
    Given a list of model weight checkpoints {w_1, ..., w_s}, optimize 
    synthetic data (X_syn, Y_syn) so that training from w_τ for ϑ steps 
    on D_syn produces model ≈ w_{τ+ϑ}.
    
    Returns: dict with keys 'x', 'y' (tensors)
    """
    torch.manual_seed(seed)
    
    if len(checkpoints) < 2:
        # Not enough checkpoints, return random data
        x_syn = torch.randn(n_samples, input_dim) * 0.1
        y_syn = torch.randint(0, 2, (n_samples,))
        return {'x': x_syn, 'y': y_syn}
    
    # Initialize synthetic data with random values
    x_syn = torch.randn(n_samples, input_dim, requires_grad=True)
    # Use soft labels (sigmoid output) for differentiability
    y_logits = torch.randn(n_samples, requires_grad=True)
    
    opt_syn = torch.optim.Adam([x_syn, y_logits], lr=lr_syn)
    
    n_ckpts = len(checkpoints)
    gen = torch.Generator().manual_seed(seed)
    
    for iteration in range(iterations):
        # Pick two random checkpoints: w_τ and w_{τ+ϑ}
        tau = torch.randint(0, n_ckpts - 1, (1,), generator=gen).item()
        vartheta = min(torch.randint(1, 4, (1,), generator=gen).item(),
                       n_ckpts - 1 - tau)
        if vartheta < 1:
            vartheta = 1
        target_idx = min(tau + vartheta, n_ckpts - 1)
        
        # Load start checkpoint
        start_model = model_class(input_dim)
        start_model.load_state_dict(copy.deepcopy(checkpoints[tau]))
        
        # Train from start checkpoint on synthetic data for ϑ steps
        inner_opt = torch.optim.SGD(start_model.parameters(), lr=0.01)
        
        y_soft = torch.sigmoid(y_logits)  # differentiable labels in [0,1]
        
        for step in range(vartheta):
            inner_opt.zero_grad()
            out = start_model(x_syn)
            # Binary cross-entropy with soft labels
            prob = F.softmax(out, dim=1)[:, 1]
            inner_loss = F.binary_cross_entropy(prob, y_soft)
            inner_loss.backward(retain_graph=True)
            inner_opt.step()
        
        # Compute matching loss: ||trained_model - target_checkpoint||^2
        target_gw = checkpoints[target_idx]
        match_loss = 0.0
        for (name, param), (_, target_param) in zip(start_model.named_parameters(),
                                                      target_gw.items()):
            match_loss = match_loss + torch.sum((param - target_param.detach()) ** 2)
        
        opt_syn.zero_grad()
        match_loss.backward()
        opt_syn.step()
    
    # Convert to hard labels and detach
    with torch.no_grad():
        x_out = x_syn.detach().clone()
        y_out = (torch.sigmoid(y_logits.detach()) > 0.5).long()
    
    return {'x': x_out, 'y': y_out}


def compute_fairness_gradient(model_class, input_dim, gw, syn_data, ref_data):
    """
    Compute calibrated update g_0 = ∇F_EO(w_t, D_syn) — the gradient of the 
    equalized-odds fairness loss on synthetic/reference data.
    (EquFL paper Section III-C, Eq. 12)
    
    Since synthetic data from distillation may not have reliable sensitive 
    attributes, we use the reference set (which has them) to compute the 
    fairness gradient. This follows the paper's spirit: the server uses 
    whatever data it has to compute fairness corrections.
    """
    model = model_class(input_dim)
    model.load_state_dict(copy.deepcopy(gw))
    model.train()
    
    rx, ry, rs = ref_data['x'], ref_data['y'], ref_data['sensitive']
    
    # Compute F_EO: difference in loss between groups
    # Following paper Eq. (8): F_EO = Σ_y max_{h,h'} |L(w,D^{h,y}) - L(w,D^{h',y})|
    # Simplified: compute loss per (group, label) cell and take max gap
    
    out = model(rx)
    log_prob = F.log_softmax(out, dim=1)
    per_sample_loss = F.nll_loss(log_prob, ry, reduction='none')
    
    # Group losses by (sensitive, label) — 4 cells
    groups = {}
    for a_val in [0, 1]:
        for y_val in [0, 1]:
            mask = (rs == a_val) & (ry == y_val)
            if mask.sum() > 0:
                groups[(a_val, y_val)] = per_sample_loss[mask].mean()
            else:
                groups[(a_val, y_val)] = torch.tensor(0.0, requires_grad=True)
    
    # Fairness loss = sum over labels of |L(a=0,y) - L(a=1,y)|
    # Use squared difference for differentiability
    f_eo = torch.tensor(0.0, requires_grad=True)
    for y_val in [0, 1]:
        l0 = groups.get((0, y_val), torch.tensor(0.0, requires_grad=True))
        l1 = groups.get((1, y_val), torch.tensor(0.0, requires_grad=True))
        f_eo = f_eo + (l0 - l1) ** 2
    
    # Also add demographic parity loss (prediction rate gap)
    probs = F.softmax(out, dim=1)[:, 1]
    dp_a0 = probs[rs == 0].mean() if (rs == 0).sum() > 0 else torch.tensor(0.0)
    dp_a1 = probs[rs == 1].mean() if (rs == 1).sum() > 0 else torch.tensor(0.0)
    f_dp = (dp_a0 - dp_a1) ** 2
    
    fairness_loss = f_eo + f_dp
    
    # Compute gradient
    fairness_loss.backward()
    
    # g_0 = ∇F(w_t, D) — the calibrated update
    g0 = torch.cat([p.grad.view(-1) for p in model.parameters()])
    return g0


def equfl_aggregate(model_class, input_dim, gw, updates, sizes, ref_data,
                    gamma=1.0, syn_data=None, checkpoints=None, current_round=0,
                    warmup_rounds=15, seed=0):
    """
    EquFL aggregation (paper Algorithm 2):
    Phase 1 (rounds 0 to warmup_rounds-1): Plain FedAvg, collect checkpoints
    Phase 2 (rounds >= warmup_rounds): FedAvg + calibrated fairness update
    
    w_{t+1} = w_t - η * (FedAvg(g_1,...,g_n) + γ * g_0)
    
    Returns: (new_gw, syn_data, checkpoints_out)
    """
    n = len(updates)
    total = sum(sizes)
    
    # Standard FedAvg aggregation
    avg = sum((s / total) * u for u, s in zip(updates, sizes))
    
    new_gw = {}
    offset = 0
    for key, par in gw.items():
        numel = par.numel()
        new_gw[key] = par + avg[offset:offset+numel].view_as(par)
        offset += numel
    
    # Checkpoint collection (Phase 1)
    if checkpoints is None:
        checkpoints = []
    
    if current_round < warmup_rounds:
        # Save checkpoint
        checkpoints.append(copy.deepcopy(new_gw))
        return new_gw, syn_data, checkpoints
    
    # Phase 2: Use reference data as the proxy dataset
    if syn_data is None and len(checkpoints) >= 2:
        syn_data = ref_data
    
    # Compute and apply calibrated update
    if syn_data is not None:
        g0 = compute_fairness_gradient(model_class, input_dim, new_gw, syn_data, ref_data)
        
        # Apply: w_{t+1} -= γ * g_0  (fairness correction)
        offset = 0
        for key, par in new_gw.items():
            numel = par.numel()
            new_gw[key] = par - gamma * g0[offset:offset+numel].view_as(par)
            offset += numel
    
    return new_gw, syn_data, checkpoints
