"""Pure functions for Atlas-MAG.

All functions trace to HADES atlas_equations:
- polynomial_features: Eq 5 (polynomial kernel) + impl-polynomial-features
- newton_schulz5: Eq 40 (Newton-Schulz5 parallel) + impl-newton-schulz5
- omega_loss: Eq 9 (Omega Rule) + Eq 8 (sliding window objective)
- lattice_update: Karami et al. 2504.05646 — orthogonal state recurrence
- lattice_readout: Karami et al. 2504.05646 — state readout
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def polynomial_features(x: torch.Tensor, degree: int, coeffs: torch.Tensor) -> torch.Tensor:
    """Construct explicit polynomial feature vector phi_p(x).

    Eq 5: phi_p(q)^T phi(k) ~ a_0 + a_1*(q^T k) + ... + a_p*(q^T k)^p

    CRITICAL (from impl-polynomial-features):
    - Uses upper-triangular outer product, NOT element-wise powers
    - Element-wise x^p misses cross-terms x_i*x_j, reducing capacity from O(d^2) to O(d)
    - For degree 2, d=64: produces 64 + 2080 = 2144 features

    Args:
        x: Input tensor [..., d]
        degree: Polynomial degree (typically 2)
        coeffs: Learnable coefficients [degree], init 1/i! (Eq 4)

    Returns:
        Feature tensor [..., d_poly] where d_poly = d + d*(d+1)/2 for degree 2
    """
    # Degree 1: scaled input
    features = [coeffs[0].abs().sqrt() * x]

    if degree >= 2:
        # Degree 2: upper triangular outer product (impl-polynomial-features)
        d = x.shape[-1]
        # x_outer: [..., d, d]
        x_outer = x.unsqueeze(-1) * x.unsqueeze(-2)
        # Take upper triangle (including diagonal) -> [..., d*(d+1)/2]
        indices = torch.triu_indices(d, d, device=x.device)
        poly2 = x_outer[..., indices[0], indices[1]]
        features.append(coeffs[1].abs().sqrt() * poly2)

    return torch.cat(features, dim=-1)


def newton_schulz5(
    S: torch.Tensor,
    num_iterations: int = 5,
    abc: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonalizing momentum matrix.

    Eq 40: S'_t <- Newton-Schulz5(S_t)
    From impl-newton-schulz5:
      X_{i+1} = X_i * (a*I + b*X_i^T*X_i + c*(X_i^T*X_i)^2)
    Converges to polar factor U*V^T (nearest semi-orthogonal matrix).

    Args:
        S: Momentum tensor [..., m, n] (can be non-square)
        num_iterations: Number of iterations (default 5, test-time compute knob)
        abc: Coefficients optimized for cubic convergence (from Muon)

    Returns:
        Orthogonalized momentum [..., m, n]
    """
    a, b, c = abc

    # Normalize for convergence: scale so spectral norm ~ 1
    # Use Frobenius norm as proxy (cheaper than spectral norm)
    norm = S.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    X = S / norm

    for _ in range(num_iterations):
        # X^T @ X: [..., n, n]
        XtX = X.transpose(-2, -1) @ X
        # a*I + b*X^T*X + c*(X^T*X)^2
        I = torch.eye(XtX.shape[-1], device=XtX.device, dtype=XtX.dtype)
        # Broadcast I to match batch dims
        I = I.expand_as(XtX)
        inner = a * I + b * XtX + c * (XtX @ XtX)
        X = X @ inner

    return X


def lattice_update(
    S: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mu: float,
    eta: float,
) -> torch.Tensor:
    """Orthogonal state recurrence update (Karami et al. 2504.05646).

    Updates state matrix S on the unit sphere given a single (k, v) pair.
    O(m*d) per token — replaces the O(d_mem^3) MLP+NS5 inner loop.
    Supports arbitrary leading batch dims.

    Steps:
        1. Normalize: Phi = S / ||S||_F
        2. Residual: h = Phi^T k - v
        3. Orthogonal gradient: Delta = k_hat (x) h - S * (h_hat . k_hat)
        4. Update: S_new = mu * S - eta * Delta
        5. Normalize back to unit sphere

    Args:
        S: State matrix [..., m, d]
        k: Key vector [..., m] (projected into slot space)
        v: Value vector [..., d]
        mu: State decay factor
        eta: Inner learning rate

    Returns:
        Updated state matrix [..., m, d], normalized to unit sphere
    """
    # 1. Frobenius norm over (m, d), keepdim for broadcast -> [..., 1, 1]
    S_norm = S.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
    Phi = S / S_norm

    # 2. Residual: h = Phi^T k - v -> [..., d]
    h = torch.einsum('...md,...m->...d', Phi, k) - v

    # 3. Orthogonal gradient on the sphere
    k_hat = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [..., m]
    h_hat = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [..., d]
    # Rank-1 term: [..., m, 1] * [..., 1, d] -> [..., m, d]
    # Tangent correction: [..., m, d] * [..., 1, d] * [..., m, 1] -> [..., m, d]
    Delta = (k_hat.unsqueeze(-1) * h.unsqueeze(-2)
             - S * (h_hat.unsqueeze(-2) * k_hat.unsqueeze(-1)))

    # 4. Decay + gradient step
    S_new = mu * S - eta * Delta

    # 5. Project back to unit sphere
    S_new_norm = S_new.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
    S_new = S_new / S_new_norm

    return S_new


def lattice_readout(S: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Readout from Lattice state matrix (Karami et al. 2504.05646).

    Simple linear readout: y = S^T q. Supports arbitrary leading batch dims.

    Args:
        S: State matrix [..., m, d]
        q: Query vector [..., m] (projected into slot space)

    Returns:
        Output vector [..., d]
    """
    return torch.einsum('...md,...m->...d', S, q)


def omega_loss(
    memory_fn,
    keys: torch.Tensor,
    values: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Omega Rule loss over a context window.

    Eq 9: L = sum_{i} gamma_i^(t) * ||M(k_i) - v_i||^2_2

    Computes L2 regression loss (attentional bias) weighted by per-token
    pruning gates gamma. The memory_fn is called on keys to produce
    predictions, compared against values.

    Args:
        memory_fn: Callable that applies memory MLP to keys
        keys: Key tensor [window_size, d_key] (already polynomial-featured)
        values: Value tensor [window_size, d_val]
        gamma: Per-token pruning gates [window_size] in [0, 1]

    Returns:
        Scalar loss
    """
    # M(k_i) for all tokens in window
    predictions = memory_fn(keys)
    # ||M(k_i) - v_i||^2 per token
    per_token_loss = (predictions - values).pow(2).sum(dim=-1)
    # Weighted sum with pruning gates
    loss = (gamma * per_token_loss).sum()
    return loss
