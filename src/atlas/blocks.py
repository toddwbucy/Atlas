"""MAGBlock — Memory As Gate composition with shared QKV projections.

From HADES:
- titans architecture-variants-comparison: MAG flow = Memory || SWA -> Gate
- model-atlas-mag: Atlas memory gated with SWA, parallel execution
- alg-atlas-complete Step 1: Q = xW_Q, K = xW_K, V = xW_V (shared projection)
- architectural-backbone: RMSNorm -> Atlas Layer -> RMSNorm -> SwiGLU
- titans architectural-details: depthwise conv, RMSNorm on Q/K

MAG pattern (Titans Eqs 26-28, adapted for Atlas):
  output = x + attn_out * sigmoid(mem_out)

Memory and attention share QKV projections (one set per block).
QKV projections + convs are at the block level, not duplicated.

CS-18: forward() is the only public API.
CS-10: No train/eval mode distinction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AtlasConfig
from .memory import AtlasMemory
from .attention import SlidingWindowAttention


class SwiGLU(nn.Module):
    """SwiGLU feedforward block (architectural-backbone: SwiGLU MLP).

    FFN(x) = W_down(SiLU(W_gate(x)) * W_up(x))
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MAGBlock(nn.Module):
    """Memory As Gate block with shared QKV projections.

    Architecture (architectural-backbone + alg-atlas-complete):
        Pre-norm -> Shared QKV proj + conv + norm
        -> [Memory(q,k,v) || Attention(q,k,v)]
        -> MAG gating: attn * sigmoid(mem) -> out_proj -> residual
        -> Pre-norm -> SwiGLU -> residual
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        d = config.head_dim
        H = config.n_heads

        # Pre-norm for attention+memory branch
        self.norm1 = nn.RMSNorm(config.d_model)

        # Shared QKV projections (Step 1 from alg-atlas-complete)
        self.w_q = nn.Linear(config.d_model, H * d, bias=False)
        self.w_k = nn.Linear(config.d_model, H * d, bias=False)
        self.w_v = nn.Linear(config.d_model, H * d, bias=False)

        # Shared short convs (Step 3 from alg-atlas-complete, titans architectural-details)
        self.conv_q = nn.Conv1d(
            H * d, H * d, kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1, groups=H * d,
        )
        self.conv_k = nn.Conv1d(
            H * d, H * d, kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1, groups=H * d,
        )
        self.conv_v = nn.Conv1d(
            H * d, H * d, kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size - 1, groups=H * d,
        )

        # Shared Q, K normalization (Step 2 from alg-atlas-complete)
        self.q_norm = nn.RMSNorm(d)
        self.k_norm = nn.RMSNorm(d)

        # Parallel branches (receive per-head tensors)
        self.memory = AtlasMemory(config)
        self.attention = SlidingWindowAttention(config)

        # Single output projection for MAG-gated output
        self.out_proj = nn.Linear(H * d, config.d_model, bias=False)

        # FFN branch (architectural-backbone: SwiGLU)
        self.norm2 = nn.RMSNorm(config.d_model)
        # Hidden dim chosen to hit ~54M total parameter target
        ffn_hidden = config.d_model
        self.ffn = SwiGLU(config.d_model, ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MAG block forward pass.

        CS-18: forward() is the only public API method.

        Args:
            x: Input tensor [B, T, d_model]

        Returns:
            Output tensor [B, T, d_model]
        """
        B, T, D = x.shape
        H = self.attention.config.n_heads
        d = self.attention.config.head_dim

        # Pre-norm
        normed = self.norm1(x)

        # Step 1: Shared QKV projection
        q = self.w_q(normed)  # [B, T, H*d]
        k = self.w_k(normed)
        v = self.w_v(normed)

        # Step 3: Shared short conv (causal)
        q = self.conv_q(q.transpose(1, 2))[..., :T].transpose(1, 2)
        k = self.conv_k(k.transpose(1, 2))[..., :T].transpose(1, 2)
        v = self.conv_v(v.transpose(1, 2))[..., :T].transpose(1, 2)

        # Reshape to per-head: [B, T, H, d]
        q = q.reshape(B, T, H, d)
        k = k.reshape(B, T, H, d)
        v = v.reshape(B, T, H, d)

        # Step 2: Normalize Q, K (per-head RMSNorm)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Run memory and attention in parallel on shared QKV
        # MAG pattern: Memory || SWA -> Gate
        mem_out = self.memory(q, k, v)     # [B, T, H, d]
        attn_out = self.attention(q, k, v)  # [B, T, H, d]

        # MAG gating: attn_out * sigmoid(mem_out)
        # Titans Eq 28 adapted for Atlas
        gated = attn_out * torch.sigmoid(mem_out)

        # Reshape and project back to d_model
        gated = gated.reshape(B, T, H * d)
        x = x + self.out_proj(gated)

        # FFN branch with pre-norm + residual
        x = x + self.ffn(self.norm2(x))

        return x
