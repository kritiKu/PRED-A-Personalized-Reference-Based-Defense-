"""
Fed-Influence V2: Faithful implementation of Xue et al. (AAAI 2021)
"Toward Understanding the Influence of Individual Clients in Federated Learning"

Equations implemented precisely:
  Eq. 4:  Combinatorial influence — re-weighted counterfactual aggregate
  Eq. 6:  FIF = ΔDP(w_hyp) − ΔDP(w_t)  [literal, no gradient approximation]
  Eq. 8-9, 14-15: Sequential influence via Fisher-approximated HVP,
                   per local SGD step, using per-example gradients (Ns=32)
  Eq. 10: Recursive FIP update: Δ̂ᶜ_t = seq_t + comb_t  (carried forward each round)
  Eq. 13, Section 4.1 (LWET): Per-layer stability check; permanently truncate
           unstable layers to combinatorial-only estimator

Adapted for ΔDP influence (not loss/accuracy).
Full participation every round (all 5 clients, C_t = C).
m_k = actual local epoch count × ceil(n_k / batch_size) steps per client.
HVP applied once per epoch boundary (coarse trajectory) for tractability.
"""

import torch
import torch.nn as nn
import copy


# ─────────────────────────────────────────────────────────────────────────────
# State-dict arithmetic helpers
# ─────────────────────────────────────────────────────────────────────────────

def clone_sd(state_dict):
    return {k: v.detach().clone() for k, v in state_dict.items()}

def zeros_like_sd(state_dict):
    return {k: torch.zeros_like(v.detach()) for k, v in state_dict.items()}

def add_sd(a, b, scale_b=1.0):
    """a + scale_b * b, per layer."""
    return {k: a[k] + scale_b * b[k] for k in a}

def scale_sd(a, s):
    return {k: a[k] * s for k in a}

def norm_per_layer(sd):
    """Return dict {layer_name: float norm}."""
    return {k: v.norm().item() for k, v in sd.items()}


# ─────────────────────────────────────────────────────────────────────────────
# ΔDP evaluation on a weight dict
# ─────────────────────────────────────────────────────────────────────────────

def compute_dp(weights, eval_data, model_class, input_dim):
    """
    Compute hard ΔDP (Demographic Parity gap) on eval_data.
    eval_data: dict with keys 'x', 'y', 'sensitive'
    Returns: float ΔDP = |P(ŷ=1|A=1) − P(ŷ=1|A=0)|
    """
    model = model_class(input_dim)
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        x    = eval_data['x']
        sens = eval_data['sensitive']
        logits = model(x)
        preds  = logits.argmax(dim=1)

        m1 = (sens == 1)
        m0 = (sens == 0)
        p1 = preds[m1].float().mean().item() if m1.sum() > 0 else 0.0
        p0 = preds[m0].float().mean().item() if m0.sum() > 0 else 0.0
    return abs(p1 - p0)


# ─────────────────────────────────────────────────────────────────────────────
# Client training with epoch-boundary checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def train_honest_checkpointed(model_class, input_dim, global_weights, client_data,
                               batch_size, local_epochs, lr, seed):
    """
    Train honest client; save state_dict BEFORE each epoch (for HVP replay).
    Returns:
      update_sd : state_dict of (local_final − global), i.e. per-layer update
      n_samples : int
      pre_epoch_sd : list of state_dicts (len = local_epochs),
                     pre_epoch_sd[e] = model state BEFORE epoch e starts
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = model_class(input_dim)
    model.load_state_dict(clone_sd(global_weights))
    model.train()

    x_tr = client_data['x']
    y_tr = client_data['y']
    n    = len(y_tr)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    pre_epoch_sd = []
    for ep in range(local_epochs):
        pre_epoch_sd.append(clone_sd(model.state_dict()))   # snapshot BEFORE epoch

        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            bx, by = x_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    final_sd  = model.state_dict()
    update_sd = {k: final_sd[k].detach() - global_weights[k].detach()
                 for k in global_weights}
    return update_sd, n, pre_epoch_sd


def train_attacker_checkpointed(model_class, input_dim, global_weights, client_data,
                                 batch_size, local_epochs, lr, seed, target_norm):
    """
    Replicate lora_attack.py exactly (SGD phase + LoRA perturbation),
    also saves per-epoch intermediate states for HVP computation.

    Note: the LoRA optimisation step (5 Adam steps on A,B) happens AFTER local
    SGD, so the epoch checkpoints reflect the honest SGD trajectory only.
    The final update includes the LoRA perturbation (norm-matched).
    For HVP propagation we use the honest-SGD intermediate states (conservative
    approximation — the attacker hides inside them anyway).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = model_class(input_dim)
    model.load_state_dict(clone_sd(global_weights))
    model.train()

    x_tr   = client_data['x']
    y_tr   = client_data['y']
    s_tr   = client_data['sensitive']
    n      = len(y_tr)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ── Phase 1: honest-looking local SGD (saves pre-epoch states) ───────────
    pre_epoch_sd = []
    for ep in range(local_epochs):
        pre_epoch_sd.append(clone_sd(model.state_dict()))

        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            bx, by = x_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    global_vec = torch.cat([v.detach().view(-1) for v in global_weights.values()])
    local_vec  = torch.cat([p.data.view(-1) for p in model.parameters()]).detach()
    local_upd  = local_vec - global_vec
    d = global_vec.numel()

    # ── Phase 2: LoRA perturbation optimisation (exact replica of lora_attack) ─
    A = (torch.randn(d, 4) * 0.01).detach().requires_grad_(True)
    B = (torch.randn(4, 1) * 0.01).detach().requires_grad_(True)
    pert_opt = torch.optim.Adam([A, B], lr=0.01)
    bs_pert  = min(256, n)

    for _ in range(5):
        pert_opt.zero_grad()
        pert      = (A @ B).view(-1)
        pert_norm = torch.norm(pert)

        cand = global_vec + local_upd + pert
        tmp  = model_class(input_dim)
        off  = 0
        for p in tmp.parameters():
            sz = p.numel()
            p.data.copy_(cand[off:off + sz].view_as(p.data))
            off += sz
        tmp.train()

        perm_p = torch.randperm(n)[:bs_pert]
        logits = tmp(x_tr[perm_p])
        probs  = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        sm, sf = (s_tr[perm_p] == 1), (s_tr[perm_p] == 0)
        pm = probs[sm].mean() if sm.sum() > 0 else torch.tensor(0.0)
        pf = probs[sf].mean() if sf.sum() > 0 else torch.tensor(0.0)

        loss_att = -torch.abs(pm - pf) + 0.1 * (pert_norm - target_norm) ** 2
        loss_att.backward()
        pert_opt.step()

    final_pert = (A @ B).view(-1).detach()
    malicious_upd_flat = local_upd + final_pert

    # Reconstruct per-layer update_sd from flat vector
    update_sd = {}
    off = 0
    for k, v in global_weights.items():
        sz = v.numel()
        update_sd[k] = malicious_upd_flat[off:off + sz].view_as(v).detach()
        off += sz

    return update_sd, n, pre_epoch_sd


# ─────────────────────────────────────────────────────────────────────────────
# Fisher-approximated HVP step  (Eqs. 8, 9, 14, 15)
# ─────────────────────────────────────────────────────────────────────────────

def hvp_step(model_class, input_dim, pre_epoch_weights_sd,
             client_data, sigma_sd, lr, n_s, rng_seed):
    """
    One HVP propagation step (per epoch boundary, Eqs. 8-9, 14-15).

    For each of n_s examples z sampled from client data:
        g_j(w, z) = per-example gradient for layer j
        σ_j ← σ_j − (η / Ns) * g_j(w,z) * (g_j(w,z)ᵀ σ_j)

    Applied PER LAYER independently (block-diagonal Fisher, Eq. 14).

    Args:
        pre_epoch_weights_sd : model weights at the start of this epoch
        client_data          : {'x', 'y', ...}
        sigma_sd             : current FIP per layer {param_name: tensor}
        lr                   : local learning rate η
        n_s                  : mini-batch size (Ns=32 per spec)
        rng_seed             : for reproducible sampling

    Returns: updated sigma_sd (same structure)
    """
    torch.manual_seed(rng_seed)

    model = model_class(input_dim)
    model.load_state_dict(pre_epoch_weights_sd)
    model.train()
    criterion = nn.CrossEntropyLoss()

    x_tr = client_data['x']
    y_tr = client_data['y']
    n    = len(y_tr)

    # Sample Ns examples
    idx    = torch.randperm(n)[:n_s]
    x_samp = x_tr[idx]
    y_samp = y_tr[idx]

    # Per-layer accumulators: Σ_z g_j(z) * (g_j(z) · σ_j)
    accum = {k: torch.zeros_like(v) for k, v in sigma_sd.items()}

    for i in range(len(x_samp)):
        x_z = x_samp[i:i + 1]
        y_z = y_samp[i:i + 1]

        model.zero_grad()
        out    = model(x_z)
        loss_z = criterion(out, y_z)
        loss_z.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g_j   = param.grad.detach()           # same shape as layer j
            s_j   = sigma_sd[name]                # same shape
            dot_j = (g_j * s_j).sum()             # scalar dot product per layer
            accum[name] = accum[name] + g_j * dot_j

        model.zero_grad()

    # σ_j ← σ_j − (η/Ns) * accum_j
    new_sigma = {k: sigma_sd[k] - (lr / n_s) * accum[k] for k in sigma_sd}
    return new_sigma


# ─────────────────────────────────────────────────────────────────────────────
# FIP estimator class
# ─────────────────────────────────────────────────────────────────────────────

class FIPEstimator:
    """
    Maintains per-candidate running FIP vector Δ̂ᶜ (one per candidate client c).

    Usage:
        est = FIPEstimator(model_class, input_dim, n_clients, lr, ns=32)
        # Each round:
        fif_dict = est.update_round(
            round_idx, global_weights, update_sds, sizes, pre_epoch_sds,
            client_data_list, eval_data, attacker_id
        )
    """

    def __init__(self, model_class, input_dim, n_clients, lr, ns=32):
        self.model_class = model_class
        self.input_dim   = input_dim
        self.n           = n_clients
        self.lr          = lr
        self.ns          = ns

        # FIP vectors: {c: state_dict-shaped dict of zero tensors}
        self.fip = None   # initialised on first update with correct shapes
        # LWET flags: {c: {layer_name: bool}}
        self.lwet = None
        # Log of LWET activations
        self.lwet_log = []   # list of (round, candidate, layer)

    def _init_fip(self, global_weights):
        self.fip  = {c: zeros_like_sd(global_weights) for c in range(self.n)}
        self.lwet = {c: {k: False for k in global_weights} for c in range(self.n)}

    def update_round(self, round_idx, global_weights, update_sds, sizes,
                     pre_epoch_sds, client_data_list, eval_data):
        """
        Perform one round of FIP update for all candidate clients.

        Args:
            round_idx      : current round number (1-indexed)
            global_weights : current global model weights (BEFORE aggregation)
            update_sds     : list of n per-client update state_dicts
            sizes          : list of n dataset sizes (n_k)
            pre_epoch_sds  : list of n lists; pre_epoch_sds[k][ep] = state before epoch ep
            client_data_list: list of n client data dicts
            eval_data      : reference/test data for ΔDP evaluation

        Returns:
            fif_dict: {c: float FIF_c}  FIF = ΔDP(w_hyp) − ΔDP(w_t)
            with w_t = FedAvg aggregate, w_hyp = w_t + Δ̂ᶜ_t
        """
        n = self.n
        N = sum(sizes)

        if self.fip is None:
            self._init_fip(global_weights)

        # ── Compute FedAvg aggregate (w_t) ───────────────────────────────────
        w_t = clone_sd(global_weights)
        for k in w_t:
            w_t[k] = sum(
                (sizes[k2] / N) * (global_weights[k].detach() + update_sds[k2][k])
                for k2 in range(n)
            )

        dp_wt = compute_dp(w_t, eval_data, self.model_class, self.input_dim)

        fif_dict = {}

        for c in range(n):
            # ── Step 1: Combinatorial term (Eq. 4) ───────────────────────────
            # w_t^{-c} = weighted average of other clients' final local models
            other_ids = [k for k in range(n) if k != c]
            N_minus_c = sum(sizes[k] for k in other_ids)

            comb_sd = zeros_like_sd(global_weights)
            for k in other_ids:
                w_k_local = {key: global_weights[key].detach() + update_sds[k][key]
                             for key in global_weights}
                weight = sizes[k] / N_minus_c
                for key in comb_sd:
                    comb_sd[key] = comb_sd[key] + weight * w_k_local[key]

            # comb_term = w_t^{-c} − w_t  (Eq. 4)
            comb_term = {key: comb_sd[key] - w_t[key] for key in w_t}

            # ── Step 2: Sequential term (Eqs. 8, 9, 14, 15) ──────────────────
            # Propagate Δ̂ᶜ_{t-1} through each other client k's local trajectory
            prev_fip  = self.fip[c]
            prop_list = []   # (k, propagated_sigma_sd) for k ≠ c

            for k in other_ids:
                sigma_k = clone_sd(prev_fip)  # start from previous FIP

                epochs = len(pre_epoch_sds[k])
                for ep in range(epochs):
                    ep_weights = pre_epoch_sds[k][ep]
                    rng_seed   = round_idx * 10000 + k * 100 + ep
                    sigma_k = hvp_step(
                        self.model_class, self.input_dim,
                        ep_weights, client_data_list[k],
                        sigma_k, self.lr, self.ns, rng_seed
                    )

                prop_list.append((k, sigma_k))

            # Weighted combination of propagated sigmas (w/ -c re-normalised weights)
            seq_sd = zeros_like_sd(global_weights)
            for k, prop_k in prop_list:
                w = sizes[k] / N_minus_c
                for key in seq_sd:
                    seq_sd[key] = seq_sd[key] + w * prop_k[key]

            # ── Step 3: LWET check (Section 4.1) ─────────────────────────────
            prev_norms = norm_per_layer(prev_fip)
            seq_norms  = norm_per_layer(seq_sd)

            for key in seq_sd:
                if not self.lwet[c][key]:
                    # Check if sequential term is amplifying
                    if seq_norms[key] > prev_norms[key] + 1e-10:
                        self.lwet[c][key] = True   # permanently truncate this layer
                        self.lwet_log.append({
                            'round': round_idx, 'candidate': c, 'layer': key,
                            'seq_norm': seq_norms[key], 'prev_norm': prev_norms[key]
                        })

            # ── Step 4: Combine (Eq. 10) — apply LWET ────────────────────────
            new_fip = {}
            for key in global_weights:
                if self.lwet[c][key]:
                    # LWET truncated: use combinatorial only (Eq. 13)
                    new_fip[key] = comb_term[key].detach().clone()
                else:
                    # Full recursive update: seq + comb
                    new_fip[key] = seq_sd[key].detach() + comb_term[key].detach()

            self.fip[c] = new_fip

            # ── Step 5: FIF computation (Eq. 6, literal) ─────────────────────
            # w_hyp = w_t + Δ̂ᶜ_t
            w_hyp = {key: w_t[key].detach() + new_fip[key] for key in w_t}
            dp_hyp = compute_dp(w_hyp, eval_data, self.model_class, self.input_dim)

            # FIF_c = ΔDP(w_hyp) − ΔDP(w_t)
            fif_dict[c] = dp_hyp - dp_wt

        return fif_dict, dp_wt
