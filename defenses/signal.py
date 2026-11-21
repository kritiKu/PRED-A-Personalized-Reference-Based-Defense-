import torch

def compute_layer_stable_rank(weight_matrix):
    """
    Fast stable rank approximation via power iteration instead of full SVD.
    Uses ||W||_F^2 / sigma_max^2 where sigma_max estimated via power iteration.
    """
    if weight_matrix.dim() != 2:
        return 1.0
    
    W = weight_matrix.float()
    f_norm_sq = torch.sum(W ** 2).item()
    
    if f_norm_sq < 1e-10:
        return 1.0
    
    # Fast power iteration: 3 steps to estimate largest singular value
    n, m = W.shape
    # Random init vector
    v = torch.randn(m, dtype=W.dtype)
    v = v / (torch.norm(v) + 1e-8)
    
    for _ in range(3):
        u = W @ v
        u_norm = torch.norm(u)
        if u_norm < 1e-8:
            return 1.0
        u = u / u_norm
        v = W.t() @ u
        v_norm = torch.norm(v)
        if v_norm < 1e-8:
            return 1.0
        v = v / v_norm
    
    # sigma_max ~ ||W @ v||
    sigma_max = torch.norm(W @ v).item()
    spec_norm_sq = sigma_max ** 2
    
    if spec_norm_sq < 1e-10:
        return 1.0
    
    return float(f_norm_sq / spec_norm_sq)


def compute_raw_rank_signal(client_update, g0, global_weights):
    """
    Compute raw rank-based signal for a client update (fast version):
    1. Project client update onto reference update g0.
    2. Residual = update - projection.
    3. Reshape residual to model weight matrices.
    4. Compute stable rank (fast power iteration) per 2D weight matrix.
    5. Normalize by reference update g0's average stable rank.
    """
    g0_norm_sq = torch.sum(g0 ** 2).item()
    if g0_norm_sq < 1e-8:
        proj = torch.zeros_like(client_update)
    else:
        dot = torch.sum(client_update * g0).item()
        proj = (dot / g0_norm_sq) * g0
        
    residual = client_update - proj
    
    offset = 0
    client_ranks = []
    ref_ranks = []
    
    for key, param in global_weights.items():
        numel = param.numel()
        if "weight" in key and param.dim() == 2:
            res_layer = residual[offset:offset + numel].view_as(param)
            g0_layer = g0[offset:offset + numel].view_as(param)
            
            c_sr = compute_layer_stable_rank(res_layer)
            r_sr = compute_layer_stable_rank(g0_layer)
            
            client_ranks.append(c_sr)
            ref_ranks.append(r_sr)
            
        offset += numel
        
    avg_client_sr = sum(client_ranks) / len(client_ranks) if len(client_ranks) > 0 else 1.0
    avg_ref_sr = sum(ref_ranks) / len(ref_ranks) if len(ref_ranks) > 0 else 1.0
    
    raw_signal = avg_client_sr / (avg_ref_sr + 1e-8)
    return raw_signal
