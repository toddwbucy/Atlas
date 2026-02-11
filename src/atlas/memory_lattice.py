"""Atlas Memory Module — Lattice orthogonal state recurrence (NO CMS).

This is the plain batched Lattice implementation without CMS multi-rate updates.
All memory slots update every chunk at the same rate. Retained as a reference
artifact. The active memory module is in memory.py (Lattice + CMS).

Based on Karami, Pascanu, Mirrokni (Google Research, 2504.05646):
- Orthogonal state recurrence replaces MLP+NS5 inner loop
- O(m*d) per token instead of O(d_mem^3)
- Chunkwise parallel formulation for GPU efficiency

Two-phase processing:
- Phase A (first bptt_len tokens): Sequential lattice_update with full autograd
  for gradient flow to S_init (outer loop learns initial conditions).
- Phase B (remaining tokens): Chunkwise parallel — freeze Phi at chunk boundary,
  compute all residuals/deltas in parallel, mu-power lower-triangular accumulation,
  per-position normalization, batched readout.

All operations batched across B and H — no Python loops over batch/head.

Code smell compliance:
- CS-18: forward() is the only public API
- CS-31: Memory layer is indivisible (no separately-callable sub-modules)
- CS-10: No train/eval mode distinction

Performance: ~7,700 tok/s on 2x A6000 (42.8x speedup over MLP+NS5).
"""

import torch
import torch.nn as nn

from .primitives import lattice_update, lattice_readout
from .config import AtlasConfig


class AtlasMemoryLattice(nn.Module):
    """Lattice memory with batched orthogonal state recurrence.

    Single-rate: all m memory slots update at every chunk.
    Operations batched across B and H dimensions — no Python loops
    over batch or head. The only Python loop is over time steps in Phase A.
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        d = config.head_dim
        m = config.lattice_m if config.lattice_m > 0 else d

        self.m = m
        self.d = d

        # Initial state on unit sphere (outer loop parameter)
        raw = torch.randn(m, d)
        raw = raw / raw.norm().clamp(min=1e-8)
        self.S_init = nn.Parameter(raw)

        # Projections — shared across all heads
        self.k_proj = nn.Linear(d, m, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.q_proj = nn.Linear(d, m, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run memory module with Lattice inner loop.

        CS-18: forward() is the only public API.

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
        """Inner loop implementation in FP32."""
        B, T, H, d = q.shape
        m = self.m
        bptt = self.config.bptt_len
        mu = self.config.lattice_mu
        eta = self.config.lattice_eta
        chunk_size = self.config.lattice_chunk_size

        # Project all positions: [B*T*H, d] -> [B, T, H, m or d]
        flat_k = k.reshape(-1, d)
        flat_v = v.reshape(-1, d)
        flat_q = q.reshape(-1, d)

        k_proj = self.k_proj(flat_k).reshape(B, T, H, m)
        v_proj = self.v_proj(flat_v).reshape(B, T, H, d)
        q_proj = self.q_proj(flat_q).reshape(B, T, H, m)

        # Initialize state: [B, H, m, d] — broadcast S_init across batch and heads
        S = self.S_init.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).clone()

        readouts = torch.zeros(B, T, H, d, device=q.device, dtype=q.dtype)

        # === Phase A: Sequential over T with autograd (for gradient flow to S_init) ===
        phase_a_len = min(bptt, T)
        for t in range(phase_a_len):
            S = lattice_update(S, k_proj[:, t], v_proj[:, t], mu, eta)
            readouts[:, t] = lattice_readout(S, q_proj[:, t])

        # === Phase B: Chunkwise parallel for remaining tokens ===
        if phase_a_len < T:
            S = S.detach()

        for chunk_start in range(phase_a_len, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            C = chunk_end - chunk_start

            if chunk_start > phase_a_len:
                S = S.detach()

            # Freeze Phi at chunk boundary
            S_norm = S.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
            Phi = S / S_norm

            k_chunk = k_proj[:, chunk_start:chunk_end].transpose(1, 2)  # [B, H, C, m]
            v_chunk = v_proj[:, chunk_start:chunk_end].transpose(1, 2)  # [B, H, C, d]
            q_chunk = q_proj[:, chunk_start:chunk_end].transpose(1, 2)  # [B, H, C, m]

            # Residuals: Phi^T @ k - v for each position
            residuals = torch.matmul(k_chunk, Phi) - v_chunk  # [B, H, C, d]

            # Deltas: rank-1 update - tangent correction
            k_norms = k_chunk.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            k_hat = k_chunk / k_norms
            h_norms = residuals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h_hat = residuals / h_norms

            rank1 = k_hat.unsqueeze(-1) * residuals.unsqueeze(-2)       # [B, H, C, m, d]
            tangent = S.unsqueeze(2) * (h_hat.unsqueeze(-2) * k_hat.unsqueeze(-1))  # [B, H, C, m, d]
            deltas = rank1 - tangent  # [B, H, C, m, d]

            # mu-power accumulation via lower-triangular matmul
            idx = torch.arange(C, device=S.device, dtype=S.dtype)
            exponents = idx.unsqueeze(0) - idx.unsqueeze(1)
            mu_lower = torch.where(
                exponents >= 0,
                mu ** exponents.clamp(min=0),
                torch.zeros(1, device=S.device, dtype=S.dtype),
            )  # [C, C]

            BH = B * H
            deltas_flat = deltas.reshape(BH, C, m * d)
            accumulated = torch.matmul(mu_lower, deltas_flat)  # [BH, C, m*d]
            accumulated = accumulated.reshape(B, H, C, m, d)

            # S at each position in chunk
            mu_powers = (mu ** (idx + 1)).view(1, 1, C, 1, 1)
            S_seq = mu_powers * S.unsqueeze(2) - eta * accumulated  # [B, H, C, m, d]

            # Normalize back to unit sphere
            S_seq_norms = S_seq.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
            S_seq = S_seq / S_seq_norms

            # Batched readout: S_seq^T @ q for each position
            y_chunk = torch.matmul(
                S_seq.transpose(-1, -2), q_chunk.unsqueeze(-1)
            ).squeeze(-1)  # [B, H, C, d]

            # Store: [B, H, C, d] -> [B, C, H, d]
            readouts[:, chunk_start:chunk_end] = y_chunk.transpose(1, 2)

            # Advance state to last position in chunk
            S = S_seq[:, :, -1]

        # Apply output projection once
        outputs = self.o_proj(readouts.reshape(-1, d)).reshape(B, T, H, d)

        return outputs
