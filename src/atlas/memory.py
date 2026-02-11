"""Atlas Memory Module — Lattice with CMS multi-rate update.

Combines the Lattice orthogonal state recurrence (Karami et al. 2504.05646)
with Continuum Memory System multi-rate updates (Behrouz et al. 2512.24695).

Memory slots are partitioned into k independent frequency levels (Eq 74).
Each level has its own update cadence — high-frequency levels update every
chunk, low-frequency levels update rarely. Readout always uses all levels.

Code smell compliance:
- CS-18: forward() is the only public API
- CS-31: Memory layer is indivisible (no separately-callable sub-modules)
- CS-10: No train/eval mode distinction
"""

import math

import torch
import torch.nn as nn

from .primitives import lattice_update, lattice_readout
from .config import AtlasConfig


class AtlasMemory(nn.Module):
    """Lattice memory with CMS multi-rate orthogonal state recurrence.

    Memory slots are partitioned into k frequency levels (Eq 74, independent
    variant). Each level is an independent Lattice state matrix with its own
    update cadence. Readout aggregates all levels via learnable weights.

    All operations batched across B and H — no Python loops over batch/head.
    The only Python loop is over k frequency levels (typically 4).
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        d = config.head_dim
        m = config.lattice_m if config.lattice_m > 0 else d
        k = config.cms_levels

        assert m % k == 0, f"m={m} must be divisible by cms_levels={k}"
        self.m = m
        self.d = d
        self.k = k
        self.spl = m // k  # slots per level
        self.cadences = list(config.cms_cadences[:k])

        # Per-level initial states on unit sphere (outer loop parameters)
        S_inits = []
        for _ in range(k):
            raw = torch.randn(self.spl, d)
            raw = raw / raw.norm().clamp(min=1e-8)
            S_inits.append(nn.Parameter(raw))
        self.S_inits = nn.ParameterList(S_inits)

        # Shared projections — full m slots, partitioned per-level at readout/update
        self.k_proj = nn.Linear(d, m, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.q_proj = nn.Linear(d, m, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        # Learned mu/eta gates (impl-dgd-stability-constraints)
        # Per-token, per-head data-dependent gates with affine clamping.
        # Zero-weight init + calibrated bias so initial outputs match scalar defaults.
        self.mu_gate = nn.Linear(d, 1, bias=True)
        self.eta_gate = nn.Linear(d, 1, bias=True)

        mu_default = config.lattice_mu
        mu_min = config.lattice_mu_min
        mu_max = config.lattice_mu_max
        eta_default = config.lattice_eta
        eta_max = config.lattice_eta_max

        nn.init.zeros_(self.mu_gate.weight)
        # bias = logit((mu - mu_min) / (mu_max - mu_min))
        mu_frac = (mu_default - mu_min) / (mu_max - mu_min)
        self.mu_gate.bias.data.fill_(math.log(mu_frac / (1.0 - mu_frac)))

        nn.init.zeros_(self.eta_gate.weight)
        # bias = logit(eta / eta_max)
        eta_frac = eta_default / eta_max
        self.eta_gate.bias.data.fill_(math.log(eta_frac / (1.0 - eta_frac)))

        # Mark gates to skip generic model init (preserves calibrated bias)
        self.mu_gate._skip_model_init = True
        self.eta_gate._skip_model_init = True

        # Learnable aggregation weights across levels (Eq 74: Agg function)
        self.agg_weights = nn.Parameter(torch.ones(k) / k)

    def _slot_slice(self, level: int) -> slice:
        """Return slice for level's slots in the m dimension."""
        return slice(level * self.spl, (level + 1) * self.spl)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run memory module with CMS multi-rate Lattice inner loop.

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
        spl = self.spl
        bptt = self.config.bptt_len
        mu_min = self.config.lattice_mu_min
        mu_max = self.config.lattice_mu_max
        eta_max = self.config.lattice_eta_max
        chunk_size = self.config.lattice_chunk_size

        # Project all positions: [B*T*H, d] -> [B, T, H, m or d]
        flat_k = k.reshape(-1, d)
        flat_v = v.reshape(-1, d)
        flat_q = q.reshape(-1, d)

        k_proj = self.k_proj(flat_k).reshape(B, T, H, m)
        v_proj = self.v_proj(flat_v).reshape(B, T, H, d)
        q_proj = self.q_proj(flat_q).reshape(B, T, H, m)

        # Per-token, per-head gates from value embeddings (impl-dgd-stability-constraints)
        # gate_input: [B, T, H, d] — use v_proj as data-dependent signal
        gate_input = v_proj  # [B, T, H, d]
        mu_raw = self.mu_gate(gate_input).squeeze(-1)    # [B, T, H]
        eta_raw = self.eta_gate(gate_input).squeeze(-1)   # [B, T, H]

        # Affine clamping: sigmoid -> scale to [min, max]
        mu_t = mu_min + (mu_max - mu_min) * torch.sigmoid(mu_raw)   # [B, T, H]
        eta_t = eta_max * torch.sigmoid(eta_raw)                     # [B, T, H]

        # Initialize per-level states: each [B, H, spl, d]
        S_levels = []
        for ell in range(self.k):
            S = self.S_inits[ell].unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).clone()
            S_levels.append(S)

        # Normalized aggregation weights
        w = torch.softmax(self.agg_weights, dim=0)  # [k]

        readouts = torch.zeros(B, T, H, d, device=q.device, dtype=q.dtype)

        # === Phase A: Sequential over T, all levels update (for gradient flow) ===
        phase_a_len = min(bptt, T)
        for t in range(phase_a_len):
            # Per-token gate values: [B, H] -> [B, H, 1, 1] for broadcast
            mu_step = mu_t[:, t].unsqueeze(-1).unsqueeze(-1)    # [B, H, 1, 1]
            eta_step = eta_t[:, t].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            y_t = torch.zeros(B, H, d, device=q.device, dtype=q.dtype)
            for ell in range(self.k):
                sl = self._slot_slice(ell)
                kt = k_proj[:, t, :, sl]  # [B, H, spl]
                vt = v_proj[:, t]          # [B, H, d]
                qt = q_proj[:, t, :, sl]   # [B, H, spl]

                S_levels[ell] = lattice_update(S_levels[ell], kt, vt, mu_step, eta_step)
                y_t = y_t + w[ell] * lattice_readout(S_levels[ell], qt)

            readouts[:, t] = y_t

        # === Phase B: Chunkwise parallel, multi-rate per level ===
        if phase_a_len < T:
            S_levels = [S.detach() for S in S_levels]

        chunk_idx = 0
        for chunk_start in range(phase_a_len, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            C = chunk_end - chunk_start

            if chunk_start > phase_a_len:
                S_levels = [S.detach() for S in S_levels]
                chunk_idx += 1

            y_chunk = torch.zeros(B, H, C, d, device=q.device, dtype=q.dtype)

            # Per-token mu/eta for this chunk: [B, H, C]
            mu_chunk = mu_t[:, chunk_start:chunk_end].transpose(1, 2)  # [B, H, C]
            eta_chunk = eta_t[:, chunk_start:chunk_end].transpose(1, 2)  # [B, H, C]

            # Log-cumsum-exp for variable-rate decay matrix (impl-dgd-stability-constraints)
            log_mu = torch.log(mu_chunk.clamp(min=1e-8))        # [B, H, C]
            cum_log_mu = torch.cumsum(log_mu, dim=-1)            # [B, H, C]

            # mu_lower[i, j] = prod(mu[j+1:i+1]) = exp(cum_log_mu[i] - cum_log_mu[j])
            # Causal: only i >= j
            diff = cum_log_mu.unsqueeze(-1) - cum_log_mu.unsqueeze(-2)  # [B, H, C, C]
            causal = torch.arange(C, device=q.device).unsqueeze(0) >= torch.arange(C, device=q.device).unsqueeze(1)
            mu_lower = torch.where(causal, torch.exp(diff), torch.zeros(1, device=q.device, dtype=q.dtype))
            # [B, H, C, C]

            # S decay: mu_powers[i] = exp(cum_log_mu[i]) = prod(mu[0:i+1])
            mu_powers = torch.exp(cum_log_mu).unsqueeze(-1).unsqueeze(-1)  # [B, H, C, 1, 1]

            for ell in range(self.k):
                sl = self._slot_slice(ell)

                # Always readout, but only update if cadence hits
                if chunk_idx % self.cadences[ell] == 0:
                    # Full update for this level
                    S = S_levels[ell]  # [B, H, spl, d]

                    S_norm = S.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
                    Phi = S / S_norm

                    k_chunk_l = k_proj[:, chunk_start:chunk_end, :, sl].transpose(1, 2)  # [B, H, C, spl]
                    v_chunk_l = v_proj[:, chunk_start:chunk_end].transpose(1, 2)          # [B, H, C, d]
                    q_chunk_l = q_proj[:, chunk_start:chunk_end, :, sl].transpose(1, 2)   # [B, H, C, spl]

                    # Residuals
                    residuals = torch.matmul(k_chunk_l, Phi) - v_chunk_l  # [B, H, C, d]

                    # Deltas
                    k_norms = k_chunk_l.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    k_hat = k_chunk_l / k_norms
                    h_norms = residuals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    h_hat = residuals / h_norms

                    rank1 = k_hat.unsqueeze(-1) * residuals.unsqueeze(-2)
                    tangent = S.unsqueeze(2) * (h_hat.unsqueeze(-2) * k_hat.unsqueeze(-1))
                    deltas = rank1 - tangent  # [B, H, C, spl, d]

                    # Accumulate deltas with variable-rate mu decay
                    BH = B * H
                    mu_lower_flat = mu_lower.reshape(BH, C, C)
                    deltas_flat = deltas.reshape(BH, C, spl * d)
                    accumulated = torch.bmm(mu_lower_flat, deltas_flat)  # [BH, C, spl*d]
                    accumulated = accumulated.reshape(B, H, C, spl, d)

                    # S decay + eta-scaled accumulated deltas
                    S_seq = mu_powers * S.unsqueeze(2) - eta_chunk.unsqueeze(-1).unsqueeze(-1) * accumulated

                    # Normalize
                    S_seq_norms = S_seq.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-8)
                    S_seq = S_seq / S_seq_norms

                    # Readout
                    y_level = torch.matmul(
                        S_seq.transpose(-1, -2), q_chunk_l.unsqueeze(-1)
                    ).squeeze(-1)  # [B, H, C, d]

                    y_chunk = y_chunk + w[ell] * y_level

                    # Advance state
                    S_levels[ell] = S_seq[:, :, -1]

                else:
                    # Frozen level: skip update, just readout from current S
                    S = S_levels[ell]  # [B, H, spl, d]
                    q_chunk_l = q_proj[:, chunk_start:chunk_end, :, sl].transpose(1, 2)  # [B, H, C, spl]

                    # Batched readout: S^T @ q for each position
                    # [B, H, d, spl] @ [B, H, spl, C] -> [B, H, d, C] -> transpose -> [B, H, C, d]
                    y_level = torch.matmul(
                        S.transpose(-1, -2), q_chunk_l.transpose(-1, -2)
                    ).transpose(-1, -2)

                    y_chunk = y_chunk + w[ell] * y_level

            # Store: [B, H, C, d] -> [B, C, H, d]
            readouts[:, chunk_start:chunk_end] = y_chunk.transpose(1, 2)

        # Apply output projection once
        outputs = self.o_proj(readouts.reshape(-1, d)).reshape(B, T, H, d)

        return outputs
