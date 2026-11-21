"""
Fed-Influence estimator adapted for Fairness (ΔDP) influence.

Implements Equations 10-15 from:
  Xue et al., "Toward Understanding the Influence of Individual Clients
  in Federated Learning", AAAI 2021.

Core idea (Eqs. 10-15):
  - Eq. 10: At each round t, client k contributes (lr_t / n) * g_k^t to global model
  - Eq. 11-12: Approximate Hessian via diagonal FIM: F̂_t = (1/n) Σ_k (g_k^t ⊙ g_k^t)
  - Eq. 13 (LWET): Apply FIM inversion independently per layer (layer-wise truncation)
  - Eq. 14-15: Recursive propagation: influence at round T from round t is
                propagated forward via: prod_{s=t+1}^{T} (I - lr_s * F̂_s / n)
                approximated as a scalar damping factor γ^(T-t) with γ = 1 - lr*mean(F̂)/n
  - ΔDP adaptation: Final FairIF_k = <∇_θ soft_ΔDP(θ^T), Σ_t propagated_IF_k^t>

Output:
  Signed influence score per (client, round):
    positive  → client pushes ΔDP upward (worsens fairness)
    negative  → client pushes ΔDP downward (improves fairness)
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fairness gradient (∇_θ soft-ΔDP)  — Eq. 14-15 terminal term
# ─────────────────────────────────────────────────────────────────────────────

def compute_fairness_gradient(model, test_data, device='cpu'):
    """
    Compute the gradient of soft-ΔDP w.r.t. model parameters.

    soft-ΔDP = (P_soft(ŷ=1 | A=1) - P_soft(ŷ=1 | A=0))^2
    where P_soft uses sigmoid on the positive-class logit (class 1).

    Returns: flat gradient vector (same shape as get_flat_params).
    """
    model.train()  # need grad tracking
    for p in model.parameters():
        p.requires_grad_(True)

    model.zero_grad()

    x    = test_data['x'].to(device)
    sens = test_data['sensitive'].to(device)

    logits = model(x)                           # (N, 2)
    pos_logit = logits[:, 1]                    # logit for class=1
    probs = torch.sigmoid(pos_logit)            # soft probability

    mask1 = (sens == 1).float()
    mask0 = (sens == 0).float()
    n1    = mask1.sum().clamp(min=1.0)
    n0    = mask0.sum().clamp(min=1.0)

    rate1  = (probs * mask1).sum() / n1
    rate0  = (probs * mask0).sum() / n0

    soft_dp = (rate1 - rate0) ** 2             # smooth, differentiable
    soft_dp.backward()

    grad_flat = torch.cat([
        p.grad.detach().flatten()
        for p in model.parameters()
        if p.grad is not None
    ])

    model.zero_grad()
    return grad_flat


# ─────────────────────────────────────────────────────────────────────────────
# Layer-wise Fed-Influence (LWET, Eq. 13)
# ─────────────────────────────────────────────────────────────────────────────

def get_layer_slices(model):
    """Return list of (start, end) index ranges for each parameter tensor."""
    slices = []
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        slices.append((offset, offset + numel))
        offset += numel
    return slices


def layer_wise_fim_inv_dot(vec_a, vec_b, fim_diag, lambda_damp=1e-3):
    """
    Compute <vec_a, F̂^{-1} ⊙ vec_b> with diagonal FIM,
    using damped inverse: (F̂_i + λ)^{-1}.

    vec_a, vec_b, fim_diag: flat tensors of same length.
    """
    fim_inv = 1.0 / (fim_diag + lambda_damp)
    return torch.dot(vec_a, fim_inv * vec_b).item()


# ─────────────────────────────────────────────────────────────────────────────
# Core Fed-Influence estimator  (Eqs. 10-15)
# ─────────────────────────────────────────────────────────────────────────────

class FedInfluenceEstimator:
    """
    Tracks per-round per-client influence on ΔDP using the
    recursive FIM-approximated Fed-Influence estimator.

    Usage:
        est = FedInfluenceEstimator(n_clients, lr)
        # Inside FL loop, each round t:
        est.record_round(t, client_updates, client_sizes, global_weights, model, test_data)
        # After all rounds:
        df = est.finalize(model, test_data)
    """

    def __init__(self, n_clients, lr, lambda_damp=1e-3, gamma_decay=True):
        """
        n_clients   : number of FL clients
        lr          : local learning rate (used as step size in Eq. 10)
        lambda_damp : damping for FIM inverse (numerical stability)
        gamma_decay : if True, propagate influence forward with discounting (Eq. 14-15)
        """
        self.n       = n_clients
        self.lr      = lr
        self.lam     = lambda_damp
        self.gamma_d = gamma_decay

        # Stored per-round data for recursive propagation
        self._rounds     = []   # list of dicts: {round, updates, fim_diag, gamma}

    def record_round(self, round_idx, client_updates, client_sizes, global_weights):
        """
        Record client gradient updates and compute diagonal FIM for this round.

        client_updates : list of flat tensors, one per client
        client_sizes   : list of ints
        global_weights : dict of state_dict (used only for shape reference here)
        """
        n = self.n
        updates = [u.detach().clone() for u in client_updates]

        # Eq. 11-12: Diagonal FIM ≈ (1/n) Σ_k g_k^2
        stacked  = torch.stack(updates)                # (n, d)
        fim_diag = (stacked ** 2).mean(dim=0)          # (d,)

        # Scalar damping factor for propagation (Eq. 14-15):
        # γ = 1 - lr * mean(F̂) / n  (scalar approximation of Jacobian product)
        mean_fim = fim_diag.mean().item()
        gamma    = max(0.0, 1.0 - self.lr * mean_fim / n)

        self._rounds.append({
            'round':    round_idx,
            'updates':  updates,    # list of (d,) tensors
            'fim_diag': fim_diag,   # (d,) tensor
            'gamma':    gamma,
        })

    def compute_all_influences(self, model, test_data, device='cpu'):
        """
        After all rounds, compute FairIF_k^t for each (client, round).

        Returns list of dicts with keys:
          round, client_id, influence_raw (single-round, non-propagated),
          influence_prop (recursively propagated to final round T).
        """
        T = len(self._rounds)
        if T == 0:
            return []

        # Final-round fairness gradient ∇_θ soft-ΔDP (Eq. 14-15 terminal)
        dp_grad = compute_fairness_gradient(model, test_data, device)  # (d,)

        records = []

        for t_idx, rdata in enumerate(self._rounds):
            round_idx = rdata['round']
            fim_diag  = rdata['fim_diag']
            updates   = rdata['updates']
            gamma_t   = rdata['gamma']

            # Propagation factor: γ^(T-1-t_idx) (Eq. 14-15)
            # Represents shrinkage of round-t contribution by subsequent rounds
            if self.gamma_d:
                prop_steps  = T - 1 - t_idx
                gammas_prod = 1.0
                for future_idx in range(t_idx + 1, T):
                    gammas_prod *= self._rounds[future_idx]['gamma']
            else:
                gammas_prod = 1.0

            for k, g_k in enumerate(updates):
                # Eq. 10: direct contribution of client k at round t
                #   IF_k^t = (lr/n) * g_k^t
                # Eq. 13 (LWET): preconditioning by layer-wise FIM inverse
                #   IF_k^t_precond = (lr/n) * F̂_t^{-1} ⊙ g_k^t
                # Final influence on ΔDP (Eq. 14-15):
                #   FairIF_k^t = <∇_θ ΔDP, γ_{t+1..T} * (lr/n) * F̂_t^{-1} ⊙ g_k^t>

                # Preconditioning (LWET diagonal)
                fim_inv   = 1.0 / (fim_diag + self.lam)
                precond_g = fim_inv * g_k                    # (d,)

                # Raw single-round influence (no propagation)
                influence_raw  = (self.lr / self.n) * torch.dot(dp_grad, precond_g).item()

                # Propagated influence (Eq. 14-15 recursive product)
                influence_prop = gammas_prod * influence_raw

                records.append({
                    'round':          round_idx,
                    'client_id':      k,
                    'influence_raw':  influence_raw,
                    'influence_prop': influence_prop,
                    'gamma_prod':     gammas_prod,
                })

        return records
