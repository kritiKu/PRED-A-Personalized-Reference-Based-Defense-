import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def train_malicious_client(model_cls, input_dim, global_weights, client_data, batch_size=32, 
                           local_epochs=5, lr=0.01, seed=None, target_norm=None, r=4):
    """
    Malicious client (Phase 3) using fast tensor slicing and mini-batch perturbation optimization.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    model = model_cls(input_dim)
    model.load_state_dict(global_weights)
    model.train()
    
    x_train, y_train, sens_train = client_data['x'], client_data['y'], client_data['sensitive']
    num_samples = len(x_train)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    
    for epoch in range(local_epochs):
        if generator is not None:
            perm = torch.randperm(num_samples, generator=generator)
        else:
            perm = torch.randperm(num_samples)
            
        for start_idx in range(0, num_samples, batch_size):
            batch_idx = perm[start_idx:start_idx + batch_size]
            bx, by = x_train[batch_idx], y_train[batch_idx]
            
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    global_vec = torch.cat([p.data.view(-1) for p in global_weights.values()]).detach()
    local_vec = torch.cat([p.data.view(-1) for p in model.parameters()]).detach()
    local_update = local_vec - global_vec
    
    d = global_vec.numel()
    
    if target_norm is None or target_norm <= 0:
        target_norm = float(torch.norm(local_update).item())
        
    A = (torch.randn(d, r) * 0.01).detach().requires_grad_(True)
    B = (torch.randn(r, 1) * 0.01).detach().requires_grad_(True)
    
    pert_optimizer = torch.optim.Adam([A, B], lr=0.01)
    batch_size_pert = min(256, num_samples)
    
    for step in range(5):
        pert_optimizer.zero_grad()
        perturbation = (A @ B).view(-1)
        pert_norm = torch.norm(perturbation)
        
        cand_params = global_vec + local_update + perturbation
        
        temp_model = model_cls(input_dim)
        offset = 0
        for p in temp_model.parameters():
            numel = p.numel()
            p.data.copy_(cand_params[offset:offset + numel].view_as(p.data))
            offset += numel
            
        temp_model.train()
        
        perm_pert = torch.randperm(num_samples)[:batch_size_pert]
        bx_p, bsens_p = x_train[perm_pert], sens_train[perm_pert]
        
        logits = temp_model(bx_p)
        probs = F.softmax(logits, dim=1)[:, 1]
        
        mask_m = (bsens_p == 1)
        mask_f = (bsens_p == 0)
        
        p_m = probs[mask_m].mean() if mask_m.sum() > 0 else torch.tensor(0.0)
        p_f = probs[mask_f].mean() if mask_f.sum() > 0 else torch.tensor(0.0)
        
        fairness_violation = torch.abs(p_m - p_f)
        penalty = 0.5 * ((pert_norm - target_norm) ** 2)
        
        loss = -fairness_violation + 0.1 * penalty
        loss.backward()
        pert_optimizer.step()
        
    final_pert = (A @ B).view(-1).detach()
    malicious_update = local_update + final_pert
    
    return malicious_update, num_samples
