import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fedavg_aggregate(global_weights, client_updates, client_sizes):
    """
    Standard FedAvg aggregation: weighted sum of local updates.
    """
    total_samples = sum(client_sizes)
    d = client_updates[0].numel()
    avg_update = torch.zeros(d, dtype=torch.float32)
    
    for update, size in zip(client_updates, client_sizes):
        avg_update += (size / total_samples) * update
        
    new_weights = {}
    offset = 0
    for key, p in global_weights.items():
        numel = p.numel()
        layer_update = avg_update[offset:offset + numel].view_as(p)
        new_weights[key] = p + layer_update
        offset += numel
        
    return new_weights, avg_update

def krum_aggregate(global_weights, client_updates, client_sizes, f=1):
    """
    Multi-KRUM defense (Phase 4):
    Assumed adversary count f=1.
    For n clients, score each by sum of distances to its (n - f - 2) nearest neighbors.
    Average top (n - f) lowest scoring updates.
    """
    n = len(client_updates)
    if n <= 2 * f + 2:
        k = max(1, n - f - 2)
    else:
        k = n - f - 2
        
    m = n - f
    
    distances = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.norm(client_updates[i] - client_updates[j]).item()
            distances[i, j] = dist
            distances[j, i] = dist
            
    scores = []
    for i in range(n):
        sorted_dists, _ = torch.sort(distances[i])
        score = sorted_dists[1:1 + k].sum().item()
        scores.append((score, i))
        
    scores.sort(key=lambda x: x[0])
    selected_indices = [idx for _, idx in scores[:m]]
    
    d = client_updates[0].numel()
    avg_update = torch.zeros(d, dtype=torch.float32)
    for idx in selected_indices:
        avg_update += client_updates[idx] / m
        
    new_weights = {}
    offset = 0
    for key, p in global_weights.items():
        numel = p.numel()
        layer_update = avg_update[offset:offset + numel].view_as(p)
        new_weights[key] = p + layer_update
        offset += numel
        
    return new_weights, avg_update

def fltrust_aggregate(global_weights, client_updates, client_sizes, reference_data, model_cls, 
                      input_dim, batch_size=32, target_steps=200, seed=None):
    """
    FLTrust defense (Phase 5) avoiding silent truncation bug:
    Fast reference SGD training using tensor indexing, ensuring steps_completed reaches target_steps (200).
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    ref_x, ref_y = reference_data['x'], reference_data['y']
    num_ref = len(ref_x)
    
    ref_model = model_cls(input_dim)
    ref_model.load_state_dict(global_weights)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    
    steps_completed = 0
    while steps_completed < target_steps:
        if generator is not None:
            perm = torch.randperm(num_ref, generator=generator)
        else:
            perm = torch.randperm(num_ref)
            
        for start_idx in range(0, num_ref, batch_size):
            if steps_completed >= target_steps:
                break
            batch_idx = perm[start_idx:start_idx + batch_size]
            bx, by = ref_x[batch_idx], ref_y[batch_idx]
            
            ref_optimizer.zero_grad()
            out = ref_model(bx)
            loss = criterion(out, by)
            loss.backward()
            ref_optimizer.step()
            steps_completed += 1
            
    global_vec = torch.cat([p.data.view(-1) for p in global_weights.values()])
    ref_vec = torch.cat([p.data.view(-1) for p in ref_model.parameters()])
    g0 = ref_vec - global_vec
    g0_norm = torch.norm(g0)
    
    n = len(client_updates)
    trust_scores = []
    scaled_updates = []
    
    for i in range(n):
        gi = client_updates[i]
        gi_norm = torch.norm(gi)
        
        if gi_norm < 1e-8 or g0_norm < 1e-8:
            trust = 0.0
        else:
            cos_sim = F.cosine_similarity(gi.unsqueeze(0), g0.unsqueeze(0)).item()
            trust = max(0.0, cos_sim)
            
        trust_scores.append(trust)
        
        if trust > 0 and gi_norm > 1e-8:
            scaled_gi = gi * (g0_norm / gi_norm)
        else:
            scaled_gi = torch.zeros_like(gi)
            
        scaled_updates.append(scaled_gi)
        
    sum_trust = sum(trust_scores)
    d = g0.numel()
    avg_update = torch.zeros(d, dtype=torch.float32)
    
    if sum_trust > 1e-8:
        for trust, scaled_gi in zip(trust_scores, scaled_updates):
            avg_update += (trust / sum_trust) * scaled_gi
    else:
        avg_update = g0
        
    new_weights = {}
    offset = 0
    for key, p in global_weights.items():
        numel = p.numel()
        layer_update = avg_update[offset:offset + numel].view_as(p)
        new_weights[key] = p + layer_update
        offset += numel
        
    return new_weights, avg_update, g0, steps_completed
