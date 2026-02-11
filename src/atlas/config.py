"""Atlas-MAG configuration dataclass.

All parameters trace to HADES graph:
- Model dims: atlas_abstractions/model-atlas-mag + AGENT_BUILD_PROMPT target spec
- Memory params: atlas_abstractions/impl-ttl-hyperparameters (Option B simplified)
- NS coefficients: Muon optimizer (Jordan et al. 2024) via impl-ttl-hyperparameters
- Polynomial degree: atlas_definitions/def-polynomial-feature-mapping + Eq 5
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AtlasConfig:
    # Model dimensions
    d_model: int = 768
    n_heads: int = 12
    head_dim: int = 64  # d_model / n_heads
    n_layers: int = 6
    vocab_size: int = 32000

    # Legacy memory parameters (retained for primitives compatibility)
    memory_expansion: int = 4  # hidden_dim = head_dim * expansion (Eq 42/43)
    poly_degree: int = 2  # Polynomial feature degree (Eq 5)
    omega_context_window: int = 256  # Sliding window c for Omega Rule (Eq 9)
    ns_iterations: int = 5  # Newton-Schulz5 iterations (Eq 40)
    ns_coefficients: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)  # a, b, c

    # Lattice hyperparameters (Karami et al. 2504.05646)
    lattice_m: int = -1           # Memory slots (-1 = head_dim)
    lattice_mu: float = 0.99      # State decay (default / bias calibration target)
    lattice_eta: float = 0.01     # Inner learning rate (default / bias calibration target)
    lattice_chunk_size: int = 64  # Chunkwise parallel chunk size

    # Learned gate bounds (impl-dgd-stability-constraints)
    lattice_mu_min: float = 0.95   # Alpha clamp lower bound for mu gate
    lattice_mu_max: float = 1.0    # Alpha clamp upper bound for mu gate
    lattice_eta_max: float = 0.1   # Max inner learning rate for eta gate

    # CMS multi-rate update (Behrouz et al. 2512.24695, Eq 71/74)
    # Independent frequency levels — each level updates at its own cadence.
    # Slots are evenly partitioned across levels. m must be divisible by cms_levels.
    cms_levels: int = 4                        # Number of frequency levels
    cms_cadences: Tuple[int, ...] = (1, 4, 16, 64)  # Chunks between updates per level

    # Truncated BPTT length for inner loop (impl-inner-loop-memory-management)
    # Limits autograd graph depth within each chunk to O(bptt_len).
    # Smaller = less GPU memory, but coarser gradient to S_init.
    bptt_len: int = 8  # Detach every N inner loop steps

    # Attention
    swa_window_size: int = 512  # Sliding Window Attention window

    # Architecture (from architectural-backbone)
    conv_kernel_size: int = 4  # Short conv on K, Q, V projections

    # Outer loop (from impl-training-pipeline, Appendix E)
    # These are used by scripts/build.py, not by the model itself
    outer_lr: float = 4e-4
    outer_weight_decay: float = 0.1
    batch_size_tokens: int = 500_000
    context_length: int = 4096


def tiny_config() -> AtlasConfig:
    """Tiny config for CPU testing. Must run in < 1 second."""
    return AtlasConfig(
        d_model=64,
        n_heads=4,
        head_dim=16,
        n_layers=2,
        vocab_size=256,
        memory_expansion=4,
        poly_degree=2,
        omega_context_window=8,
        ns_iterations=5,
        swa_window_size=16,
        conv_kernel_size=4,
        context_length=32,
        lattice_chunk_size=8,
        cms_levels=2,
        cms_cadences=(1, 4),
    )


def full_config() -> AtlasConfig:
    """Full ~54M parameter config."""
    return AtlasConfig()
