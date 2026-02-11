"""Sliding Window Attention (SWA) with Rotary Position Embeddings (RoPE).

From HADES:
- impl-positional-encoding: RoPE on Q, K in SWA path only (not memory path)
- Memory path does NOT use positional encoding

Receives pre-projected per-head (q, k, v) from the MAGBlock.
QKV projections are shared with the memory branch at block level.

CS-18: forward() is the only public API.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AtlasConfig


class SlidingWindowAttention(nn.Module):
    """Sliding Window Attention with RoPE positional encoding.

    Receives pre-projected per-head tensors. Applies RoPE internally
    (impl-positional-encoding: RoPE only in SWA path, not memory path).
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config
        d = config.head_dim

        # Pre-compute RoPE frequencies (impl-positional-encoding)
        # theta_i = 10000^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer('inv_freq', inv_freq)

        self.scale = d ** -0.5

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply Rotary Position Embeddings.

        impl-positional-encoding: Standard sinusoidal rotation of pairs of
        dimensions with position-dependent frequencies.

        Args:
            x: [B, H, T, d]
            seq_len: sequence length T

        Returns:
            Tensor with RoPE applied [B, H, T, d]
        """
        d = x.shape[-1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos_f = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, T, d/2]
        sin_f = freqs.sin().unsqueeze(0).unsqueeze(0)

        x1 = x[..., :d // 2]
        x2 = x[..., d // 2:]
        rotated = torch.cat([
            x1 * cos_f - x2 * sin_f,
            x1 * sin_f + x2 * cos_f,
        ], dim=-1)

        return rotated.to(x.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sliding window attention with RoPE.

        Args:
            q: Query tensor [B, T, H, d] (pre-projected, normalized)
            k: Key tensor [B, T, H, d] (pre-projected, normalized)
            v: Value tensor [B, T, H, d] (pre-projected)

        Returns:
            Output tensor [B, T, H, d]
        """
        B, T, H, d = q.shape
        W = self.config.swa_window_size

        # Transpose for attention: [B, H, T, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K (impl-positional-encoding: SWA path only)
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)

        # Scaled dot-product attention with sliding window causal mask
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create causal sliding window mask
        row_idx = torch.arange(T, device=q.device).unsqueeze(1)
        col_idx = torch.arange(T, device=q.device).unsqueeze(0)
        causal_mask = col_idx <= row_idx
        window_mask = (row_idx - col_idx) < W
        mask = causal_mask & window_mask

        attn_weights = attn_weights.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attend
        out = torch.matmul(attn_weights, v)  # [B, H, T, d]

        # Back to [B, T, H, d]
        return out.transpose(1, 2).contiguous()
