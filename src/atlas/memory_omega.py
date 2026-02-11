"""Atlas Memory Module — Pure Omega Rule (Sequential).

This is the naive, paper-faithful implementation of the Atlas memory system.
It implements ONLY the original Atlas paper equations with no optimizations
from later work (no TNT parallelization, no Lattice, no CMS).

This exists to demonstrate WHY you can't just read one paper and ship it.
Performance: ~7 tok/s on 2x A6000. Compare to memory.py (Lattice+CMS): ~6000 tok/s.

The bottleneck is clear: for EVERY token, we must:
  1. Compute omega_loss over the sliding window (Eq 9)
  2. Call autograd.grad with create_graph=True (Eq 58)
  3. Run Newton-Schulz5 orthogonalization — O(d_mem^3) (Eq 40)
  4. Update memory MLP weights (Eq 57)
  5. Read from memory via forward pass through the MLP

That's a full forward + backward + NS5 PER TOKEN. Sequentially. No escape.

Equations:
- Eq 9:  omega_loss = sum gamma_i ||M(k_i) - v_i||^2 — Omega Rule
- Eq 5:  Polynomial features for keys/queries
- Eq 40: Newton-Schulz5 orthogonalization of momentum
- Eq 41: M_t = alpha * M_{t-1} + NS5(S_t) — memory update
- Eq 42: M(x) = W_down(GELU(W_up(x)) * W_gate(x)) — gated MLP
- Eq 57: M_t = alpha * M_{t-1} + NS5(S_t)
- Eq 58: S_t = theta * S_{t-1} - eta * grad(omega_loss) — momentum

Code smell compliance:
- CS-18: forward() is the only public API
- CS-31: Memory layer is indivisible
- CS-10: No train/eval mode distinction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import polynomial_features, newton_schulz5, omega_loss
from .config import AtlasConfig

# Original Atlas hyperparameters (impl-ttl-hyperparameters, Option B)
# These were removed from AtlasConfig when Lattice replaced the MLP inner loop.
_ALPHA = 0.99   # Memory decay (Eq 57)
_THETA = 0.99   # Momentum decay (Eq 58)
_ETA = 0.01     # Inner learning rate (Eq 58)


class AtlasMemoryOmega(nn.Module):
    """Pure Omega Rule memory — sequential, per-token inner loop.

    The memory is a gated MLP whose weights update DURING the forward pass.
    Every single token triggers: omega_loss -> autograd.grad -> NS5 -> update.
    This is faithful to the paper but catastrophically slow.

    Architecture (Eq 43 — gated MLP):
        M(x) = W_down(GELU(W_up(x)) * W_gate(x))

    Inner loop per token t:
        1. loss = omega_loss over window [t-c+1, t]     (Eq 9)
        2. grads = d(loss)/d(W_up, W_gate, W_down)      (autograd)
        3. S_t = theta * S_{t-1} - eta * grads           (Eq 58)
        4. S'_t = NS5(S_t)                               (Eq 40)
        5. M_t = alpha * M_{t-1} + S'_t                  (Eq 57)
        6. y_t = M_t(q_t)                                (readout)
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        d = config.head_dim
        d_mem = d * config.memory_expansion

        # Polynomial feature coefficients (Eq 5, init a_i = 1/i!)
        poly_coeffs_init = torch.tensor(
            [1.0 / math.factorial(i + 1) for i in range(config.poly_degree)]
        )
        self.poly_coeffs = nn.Parameter(poly_coeffs_init)

        # Polynomial feature dimension
        d_poly = d
        if config.poly_degree >= 2:
            d_poly += d * (d + 1) // 2
        self.d_poly = d_poly

        # Polynomial compression (d_poly -> d_mem)
        self.poly_proj = nn.Linear(d_poly, d_mem, bias=False)

        # Memory MLP weights — these ARE the inner loop parameters
        # Eq 43: M(x) = W_down(GELU(W_up(x)) * W_gate(x))
        self.mem_w_up = nn.Linear(d_mem, d_mem, bias=False)
        self.mem_w_gate = nn.Linear(d_mem, d_mem, bias=False)
        self.mem_w_down = nn.Linear(d_mem, d, bias=False)

        # Pruning gate for gamma_i (learned, per impl-ttl-hyperparameters)
        gamma_hidden = max(d, 32)
        self.gamma_net = nn.Sequential(
            nn.Linear(d, gamma_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(gamma_hidden, 1, bias=False),
            nn.Sigmoid(),
        )

    def _apply_memory_mlp(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply gated MLP with given parameters.

        Eq 43: M(x) = W_down(GELU(W_up(x)) * W_gate(x))
        """
        up = F.linear(x, params['w_up'])
        gate = F.linear(x, params['w_gate'])
        return F.linear(F.gelu(up) * gate, params['w_down'])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run pure sequential Omega Rule inner loop.

        CS-18: forward() is the only public API.

        For every token: omega_loss -> autograd.grad -> NS5 -> update -> read.
        This is the reason this implementation gets 7 tok/s.

        Args:
            q: Query tensor [B, T, H, d]
            k: Key tensor [B, T, H, d]
            v: Value tensor [B, T, H, d]

        Returns:
            Output tensor [B, T, H, d]
        """
        return self._forward_impl(q.float(), k.float(), v.float())

    @torch.amp.autocast('cuda', enabled=False)
    def _forward_impl(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sequential inner loop in FP32. Every token is a full update cycle."""
        B, T, H, d = q.shape
        c = self.config.omega_context_window

        outputs = torch.zeros(B, T, H, d, device=q.device, dtype=q.dtype)

        # Initial memory parameters (the outer loop learns these)
        mem_params = {
            'w_up': self.mem_w_up.weight,
            'w_gate': self.mem_w_gate.weight,
            'w_down': self.mem_w_down.weight,
        }

        # Momentum buffers (zeros at start of each sequence)
        S = {name: torch.zeros_like(p) for name, p in mem_params.items()}

        # === Pure sequential: one token at a time, every token ===
        for t in range(T):
            # Omega window: [max(0, t-c+1), t+1)
            win_start = max(0, t - c + 1)
            win_k = k[:, win_start:t + 1]   # [B, win, H, d]
            win_v = v[:, win_start:t + 1]

            # Flatten for polynomial features
            win_k_flat = win_k.reshape(-1, d)
            win_v_flat = win_v.reshape(-1, d)

            # Eq 5: polynomial features + compression
            win_k_poly = polynomial_features(
                win_k_flat, self.config.poly_degree, self.poly_coeffs
            )
            win_k_compressed = self.poly_proj(win_k_poly)

            # Learned pruning gates
            gamma = self.gamma_net(win_k_flat).squeeze(-1)

            # Eq 9: omega_loss = sum gamma_i ||M(k_i) - v_i||^2
            def mem_fn(keys, _params=mem_params):
                return self._apply_memory_mlp(keys, _params)

            loss = omega_loss(mem_fn, win_k_compressed, win_v_flat, gamma)

            # Autograd: d(loss)/d(MLP weights)
            # create_graph=True is critical — without it, the outer loop
            # can't backprop through the inner loop to learn M_init.
            grads = torch.autograd.grad(
                loss, list(mem_params.values()),
                create_graph=True,
            )
            grad_dict = dict(zip(mem_params.keys(), grads))

            # Eq 58: momentum update S_t = theta * S_{t-1} - eta * grad
            for name in S:
                S[name] = _THETA * S[name] - _ETA * grad_dict[name]

            # Eq 40: Newton-Schulz5 orthogonalization
            # This is O(d_mem^3) PER TOKEN — the dominant cost
            S_ortho = {}
            for name, s in S.items():
                if s.dim() >= 2:
                    S_ortho[name] = newton_schulz5(
                        s.unsqueeze(0),
                        num_iterations=self.config.ns_iterations,
                        abc=self.config.ns_coefficients,
                    ).squeeze(0)
                else:
                    S_ortho[name] = s

            # Eq 57: memory update M_t = alpha * M_{t-1} + NS5(S_t)
            mem_params = {
                name: _ALPHA * mem_params[name] + S_ortho[name]
                for name in mem_params
            }

            # Readout: apply updated MLP to query
            q_t_flat = q[:, t:t + 1].reshape(-1, d)
            q_t_poly = polynomial_features(
                q_t_flat, self.config.poly_degree, self.poly_coeffs
            )
            q_t_compressed = self.poly_proj(q_t_poly)
            y_t = self._apply_memory_mlp(q_t_compressed, mem_params)
            outputs[:, t] = y_t.view(B, H, d)

        return outputs
