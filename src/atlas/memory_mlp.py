"""Atlas Memory Module — deep MLP with inner loop TTL updates (ORIGINAL).

This is the original MLP+NS5 memory implementation. Retained as a reference
artifact. The active memory module is in memory.py (Lattice + CMS).

Equations:
- Eq 42: M(x) = x + W1 * sigma(W2 * x) — deep memory MLP with residual
- Eq 43: M(x) = x + W1 * (sigma(W2 * x) ⊗ W3 * x) — gated MLP (SwiGLU-style)
- Eq 57: M_t = alpha_t * M_{t-1} + NS5(S_t) — memory update
- Eq 58: S_t = theta_t * S_{t-1} - eta * grad(omega_loss) — momentum
- Eq 9: omega_loss = sum gamma_i ||M(k_i) - v_i||^2 — Omega Rule
- Eq 5: polynomial features for keys/queries
- Eq 39: Fully parallelizable momentum (chunk-wise, Section 5.1)
- Eq 40: Batched Newton-Schulz5 (parallel across chunk positions)
- Eq 41: Sequential memory update M_t = alpha * M_{t-1} + NS5(S_t)

Architecture from:
- def-deep-mlp-memory: 2 layers, expansion 4, GELU, residual
- def-inner-outer-loop-partition: inner loop updates theta_M = {W1, W2, W3}
- impl-inner-loop-memory-management: chunkwise detachment pattern
- impl-ttl-hyperparameters: Option B (fixed alpha/theta/eta, learned gamma)
- impl-polynomial-features: proj_down compression for d_poly → d_mem
- atlas-parallel-training: momentum independent of memory state

Code smell compliance:
- CS-18: forward() is the only public API
- CS-31: Memory layer is indivisible (no separately-callable sub-modules)
- CS-10: No train/eval mode distinction

Performance: ~168 tok/s on 2x A6000 (bottleneck: NS5 O(d^3) per token).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import polynomial_features, newton_schulz5, omega_loss
from .config import AtlasConfig


class AtlasMemoryMLP(nn.Module):
    """Deep MLP memory with inner loop TTL updates in forward().

    The memory MLP parameters ARE the memory (def-associative-memory).
    The inner loop runs DURING forward() — this is test-time learning.
    The outer loop learns the initial memory parameters via backprop.

    Receives pre-projected, per-head (q, k, v) from the MAGBlock.
    QKV projections are shared with the attention branch at block level.

    Architecture (Eq 43 — gated MLP, recommended per alg-atlas-complete Option C):
        M(x) = W_down(GELU(W_up(x)) * W_gate(x))

    Parallelization (atlas-parallel-training, Section 5.1):
        Phase A (first bptt_len steps): Sequential with create_graph=True
            for outer loop gradient flow to M_init.
        Phase B (remaining steps): Parallel formulation (Eqs 39-41) where
            all gradients are computed w.r.t. frozen chunk boundary M_{t'},
            momentum accumulated via matmul, NS5 applied in batch.
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        d = config.head_dim
        d_mem = d * config.memory_expansion  # compressed memory dim

        # Polynomial feature coefficients (Eq 5, init a_i = 1/i! from Eq 4)
        poly_coeffs_init = torch.tensor(
            [1.0 / math.factorial(i + 1) for i in range(config.poly_degree)]
        )
        self.poly_coeffs = nn.Parameter(poly_coeffs_init)

        # Compute polynomial feature dimension
        d_poly = d  # degree 1
        if config.poly_degree >= 2:
            d_poly += d * (d + 1) // 2  # degree 2 outer product
        self.d_poly = d_poly

        # Polynomial feature compression (impl-polynomial-features: learned proj_down)
        # Outer loop parameter — reduces d_poly to d_mem for the memory MLP
        self.poly_proj = nn.Linear(d_poly, d_mem, bias=False)

        # Memory MLP weights — these are the inner loop parameters (theta_M)
        # Eq 43: M(x) = W_down(GELU(W_up(x)) * W_gate(x))
        self.mem_w_up = nn.Linear(d_mem, d_mem, bias=False)
        self.mem_w_gate = nn.Linear(d_mem, d_mem, bias=False)
        self.mem_w_down = nn.Linear(d_mem, d, bias=False)

        # Pruning gate network for gamma_i (always learned per impl-ttl-hyperparameters)
        # 2-layer MLP with SiLU + Sigmoid
        gamma_hidden = max(d, 32)
        self.gamma_net = nn.Sequential(
            nn.Linear(d, gamma_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(gamma_hidden, 1, bias=False),
            nn.Sigmoid(),
        )

    def _apply_memory_mlp(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply the memory MLP with given parameters.

        Eq 43: M(x) = W_down(GELU(W_up(x)) * W_gate(x))
        """
        up = F.linear(x, params['w_up'])
        gate = F.linear(x, params['w_gate'])
        down = F.linear(F.gelu(up) * gate, params['w_down'])
        return down

    def _manual_mlp_grads(self, x, v_target, gamma, params):
        """Compute per-key MLP gradients via manual backward pass.

        For omega_loss = sum_i gamma_i * ||MLP(x_i) - v_i||^2, compute
        d_loss/d_{W_up, W_gate, W_down} for EACH key independently.

        Returns dict of per-key gradients: {name: [N, *weight_shape]}.
        """
        # Forward through MLP (Eq 43)
        h_up = F.linear(x, params['w_up'])      # [N, d_mem]
        h_gate = F.linear(x, params['w_gate'])   # [N, d_mem]
        act = F.gelu(h_up)                       # [N, d_mem]
        hidden = act * h_gate                    # [N, d_mem]
        pred = F.linear(hidden, params['w_down'])  # [N, d]

        # Upstream gradient: d_loss/d_pred = 2 * gamma * (pred - v)
        d_pred = 2.0 * gamma.unsqueeze(-1) * (pred - v_target)  # [N, d]

        # Backward through W_down: pred = hidden @ W_down.T
        d_hidden = d_pred @ params['w_down']  # [N, d_mem]

        # Backward through gating: hidden = act * h_gate
        d_act = d_hidden * h_gate       # [N, d_mem]
        d_h_gate = d_hidden * act       # [N, d_mem]

        # Backward through GELU: act = gelu(h_up)
        # GELU'(x) = Phi(x) + x * phi(x)
        cdf = 0.5 * (1.0 + torch.erf(h_up * 0.7071067811865476))
        pdf = torch.exp(-0.5 * h_up * h_up) * 0.3989422804014327
        d_h_up = d_act * (cdf + h_up * pdf)  # [N, d_mem]

        # Per-key gradient tensors via batched outer products
        # [N, out, in] for each weight matrix
        grad_w_down = torch.einsum('nd,nm->ndm', d_pred, hidden)    # [N, d, d_mem]
        grad_w_up = torch.einsum('nm,nk->nmk', d_h_up, x)          # [N, d_mem, d_mem]
        grad_w_gate = torch.einsum('nm,nk->nmk', d_h_gate, x)      # [N, d_mem, d_mem]

        return {'w_up': grad_w_up, 'w_gate': grad_w_gate, 'w_down': grad_w_down}

    def _build_window_mask(self, chunk_len, c, device):
        """Build sliding window mask for omega rule aggregation.

        mask[t, i] = 1 if key i is in token t's omega window.
        Window for token t: [max(0, t-c+1), t].
        """
        rows = torch.arange(chunk_len, device=device)
        cols = torch.arange(chunk_len, device=device)
        diff = rows.unsqueeze(1) - cols.unsqueeze(0)  # [C, C]
        mask = (diff >= 0) & (diff < c)
        return mask.float()

    def _build_theta_lower(self, chunk_len, theta, device, dtype):
        """Build lower-triangular theta decay matrix for momentum accumulation (Eq 39).

        theta_lower[t, i] = theta^(t-i) for i <= t, else 0.
        S_t = theta^(t+1) * S_init - eta * sum_i theta^(t-i) * u_i
            = theta^(t+1) * S_init - eta * theta_lower[t, :] @ U
        """
        idx = torch.arange(chunk_len, device=device)
        exponents = idx.unsqueeze(0) - idx.unsqueeze(1)  # [C, C]
        theta_lower = torch.where(
            exponents >= 0,
            theta ** exponents.float().clamp(min=0),
            torch.zeros(1, device=device, dtype=dtype),
        )
        return theta_lower  # [C, C]

    def _parallel_inner_loop(self, q_chunk, k_chunk, v_chunk,
                             mem_params, S, outputs, chunk_start):
        """Process a chunk using the parallel formulation (Section 5.1, Eqs 39-41).

        atlas-parallel-training: momentum recurrence is independent of memory state.
        1. Compute all gradients w.r.t. frozen M_{t'} (batched forward + manual backward)
        2. Aggregate per-key grads into per-token grads via window mask
        3. Accumulate momentum via matmul with theta powers (Eq 39)
        4. Apply NS5 in batch (Eq 40)
        5. Sequential memory update + reads (Eq 41)
        """
        B, chunk_len, H, d = q_chunk.shape
        c = self.config.omega_context_window
        device = q_chunk.device
        dtype = q_chunk.dtype

        # === Phase 1: Batched forward + manual backward (all parallel) ===

        # Freeze mem_params at chunk boundary M_{t'}
        frozen_params = {name: p.detach() for name, p in mem_params.items()}

        # Compute polynomial features + compression for ALL keys in chunk at once
        k_flat = k_chunk.reshape(B * chunk_len * H, d)
        v_flat = v_chunk.reshape(B * chunk_len * H, d)

        k_poly = polynomial_features(k_flat, self.config.poly_degree, self.poly_coeffs)
        k_compressed = self.poly_proj(k_poly)  # [B*C*H, d_mem]

        # Compute gamma for all keys
        gamma_all = self.gamma_net(k_flat).squeeze(-1)  # [B*C*H]

        # Per-key MLP gradients via manual backward (no autograd.grad calls!)
        per_key_grads = self._manual_mlp_grads(
            k_compressed, v_flat, gamma_all, frozen_params
        )  # {name: [B*C*H, *weight_shape]}

        # === Phase 2: Aggregate per-key → per-token grads via window mask ===

        # Reshape per-key grads: [B*C*H, ...] → [B, C, H, ...]
        # Then sum over B and H (MLP weights are shared), apply window mask on C
        window_mask = self._build_window_mask(chunk_len, c, device)  # [C, C]

        per_token_grads = {}
        for name, pkg in per_key_grads.items():
            shape = pkg.shape[1:]  # weight shape, e.g., (d_mem, d_mem)
            # [B, C, H, *shape] → sum over B and H → [C, *shape]
            pkg_reshaped = pkg.reshape(B, chunk_len, H, *shape)
            pkg_summed = pkg_reshaped.sum(dim=(0, 2))  # [C, *shape]
            # Apply window mask: [C, C] @ [C, flat_size] → [C, flat_size]
            flat_size = pkg_summed[0].numel()
            pkg_flat = pkg_summed.reshape(chunk_len, flat_size)  # [C, flat]
            ptg_flat = window_mask @ pkg_flat  # [C, flat]
            per_token_grads[name] = ptg_flat.reshape(chunk_len, *shape)

        # === Phase 3: Momentum accumulation via matmul (Eq 39) ===

        theta_lower = self._build_theta_lower(chunk_len, self.config.theta, device, dtype)
        theta_powers = self.config.theta ** torch.arange(1, chunk_len + 1, device=device, dtype=dtype)

        S_all = {}
        for name in mem_params:
            shape = mem_params[name].shape
            flat_size = mem_params[name].numel()
            # U: per-token gradient matrix [C, flat_size]
            U = per_token_grads[name].reshape(chunk_len, flat_size)
            # S_init for this weight
            S_init_flat = S[name].reshape(flat_size)
            # Eq 39: S_t = theta^(t+1) * S_init - eta * theta_lower @ U
            weighted_grads = theta_lower @ U  # [C, flat_size]
            S_t = (theta_powers.unsqueeze(-1) * S_init_flat.unsqueeze(0)
                   - self.config.eta * weighted_grads)
            S_all[name] = S_t.reshape(chunk_len, *shape)

        # Update S to final state (last row)
        for name in S:
            S[name] = S_all[name][-1]

        # === Phase 4: Batched NS5 (Eq 40) ===

        S_ortho_all = {}
        for name, s_seq in S_all.items():
            if s_seq.dim() >= 3:  # [C, rows, cols] — NS5 expects [batch, rows, cols]
                S_ortho_all[name] = newton_schulz5(
                    s_seq,
                    num_iterations=self.config.ns_iterations,
                    abc=self.config.ns_coefficients,
                )
            else:
                S_ortho_all[name] = s_seq

        # === Phase 5: Sequential memory update + reads (Eq 41) ===

        for t_idx in range(chunk_len):
            t = chunk_start + t_idx
            # Eq 41: M_t = alpha * M_{t-1} + S'_t
            mem_params = {
                name: self.config.alpha * mem_params[name] + S_ortho_all[name][t_idx]
                for name in mem_params
            }
            # Read from memory using query
            q_t = q_chunk[:, t_idx:t_idx + 1]  # [B, 1, H, d]
            q_t_flat = q_t.reshape(-1, d)
            q_t_poly = polynomial_features(
                q_t_flat, self.config.poly_degree, self.poly_coeffs
            )
            q_t_compressed = self.poly_proj(q_t_poly)  # [B*H, d_mem]
            y_t = self._apply_memory_mlp(q_t_compressed, mem_params)  # [B*H, d]
            outputs[:, t] = y_t.view(B, H, d)

        return mem_params, S

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run memory module with inner loop TTL updates.

        CS-18: forward() is the only public API.

        Two-phase processing:
        - Phase A (first bptt_len steps): Sequential with create_graph=True
          for outer loop gradient flow to M_init.
        - Phase B (remaining steps): Parallel formulation (Eqs 39-41)
          where gradients are batched and momentum accumulated via matmul.

        Args:
            q: Query tensor [B, T, H, d] (pre-projected, normalized)
            k: Key tensor [B, T, H, d] (pre-projected, normalized)
            v: Value tensor [B, T, H, d] (pre-projected)

        Returns:
            Output tensor [B, T, H, d]
        """
        return self._forward_impl(q.float(), k.float(), v.float())

    @torch.amp.autocast('cuda', enabled=False)
    def _forward_impl(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Inner loop implementation in FP32. Called by forward()."""
        B, T, H, d = q.shape
        c = self.config.omega_context_window
        bptt = self.config.bptt_len

        # Collect outputs
        outputs = torch.zeros(B, T, H, d, device=q.device, dtype=q.dtype)

        # Get initial memory parameters (inner loop params = theta_M)
        mem_params = {
            'w_up': self.mem_w_up.weight,      # [d_mem, d_mem]
            'w_gate': self.mem_w_gate.weight,   # [d_mem, d_mem]
            'w_down': self.mem_w_down.weight,   # [d, d_mem]
        }

        # Initialize momentum buffers S for each weight (zeros)
        S = {name: torch.zeros_like(p) for name, p in mem_params.items()}

        # === Phase A: Sequential steps with create_graph=True ===
        phase_a_len = min(bptt, T)
        for t in range(phase_a_len):
            # Omega window: [max(0, t-c+1), t+1)
            win_start = max(0, t - c + 1)
            win_k = k[:, win_start:t + 1]
            win_v = v[:, win_start:t + 1]

            win_k_flat = win_k.reshape(-1, d)
            win_k_poly = polynomial_features(
                win_k_flat, self.config.poly_degree, self.poly_coeffs
            )
            win_k_compressed = self.poly_proj(win_k_poly)
            gamma = self.gamma_net(win_k_flat).squeeze(-1)
            win_v_flat = win_v.reshape(-1, d)

            def mem_fn(keys, _params=mem_params):
                return self._apply_memory_mlp(keys, _params)

            loss = omega_loss(mem_fn, win_k_compressed, win_v_flat, gamma)

            grads = torch.autograd.grad(
                loss, list(mem_params.values()),
                create_graph=True,  # Critical for outer loop gradient flow
            )

            # Eq 58: momentum update
            grad_dict = dict(zip(mem_params.keys(), grads))
            for name in S:
                S[name] = self.config.theta * S[name] - self.config.eta * grad_dict[name]

            # Eq 40: NS5
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

            # Eq 57: memory update
            mem_params = {
                name: self.config.alpha * mem_params[name] + S_ortho[name]
                for name in mem_params
            }

            # Read from memory
            q_t_flat = q[:, t:t + 1].reshape(-1, d)
            q_t_poly = polynomial_features(
                q_t_flat, self.config.poly_degree, self.poly_coeffs
            )
            q_t_compressed = self.poly_proj(q_t_poly)
            y_t = self._apply_memory_mlp(q_t_compressed, mem_params)
            outputs[:, t] = y_t.view(B, H, d)

        # === Phase B: Parallel processing for remaining tokens ===
        # Detach from Phase A (truncated BPTT boundary)
        if phase_a_len < T:
            mem_params = {name: p.detach().requires_grad_(True)
                          for name, p in mem_params.items()}
            S = {name: s.detach() for name, s in S.items()}

        # Process remaining tokens in chunks using parallel formulation
        for chunk_start in range(phase_a_len, T, c):
            chunk_end = min(chunk_start + c, T)

            if chunk_start > phase_a_len:
                # Detach at chunk boundaries (impl-inner-loop-memory-management)
                mem_params = {name: p.detach().requires_grad_(True)
                              for name, p in mem_params.items()}
                S = {name: s.detach() for name, s in S.items()}

            q_chunk = q[:, chunk_start:chunk_end]
            k_chunk = k[:, chunk_start:chunk_end]
            v_chunk = v[:, chunk_start:chunk_end]

            mem_params, S = self._parallel_inner_loop(
                q_chunk, k_chunk, v_chunk,
                mem_params, S, outputs, chunk_start,
            )

        return outputs
