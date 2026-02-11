"""Tests for AtlasMemory module (Lattice orthogonal state recurrence)."""

import torch
import pytest
from atlas.memory import AtlasMemory
from atlas.config import tiny_config


class TestAtlasMemory:
    """Tests for the Lattice memory module with inner loop."""

    def _make_qkv(self, config, B=2, T=8):
        """Helper to create per-head QKV tensors."""
        return (
            torch.randn(B, T, config.n_heads, config.head_dim),
            torch.randn(B, T, config.n_heads, config.head_dim),
            torch.randn(B, T, config.n_heads, config.head_dim),
        )

    def test_output_shape(self, config):
        """Memory output should be [B, T, H, d]."""
        mem = AtlasMemory(config)
        q, k, v = self._make_qkv(config)
        out = mem(q, k, v)
        assert out.shape == (2, 8, config.n_heads, config.head_dim)

    def test_gradients_flow_to_memory_init(self, config):
        """Bilevel: gradients must flow to S_init and projection weights.

        This is the critical test for bilevel optimization.
        The inner loop modifies state during forward; the outer loop
        must learn the initial conditions via backprop.
        """
        mem = AtlasMemory(config)
        q, k, v = self._make_qkv(config, B=1)
        out = mem(q, k, v)
        loss = out.sum()
        loss.backward()

        # S_inits (per-level initial states) should have gradients
        for ell, S_init in enumerate(mem.S_inits):
            assert S_init.grad is not None, f"Level {ell}: S_init has no gradient"
        # Projection weights should have gradients
        assert mem.k_proj.weight.grad is not None
        assert mem.v_proj.weight.grad is not None

    def test_gradients_flow_to_projections(self, config):
        """Outer loop params (q_proj, o_proj) should have gradients."""
        mem = AtlasMemory(config)
        q, k, v = self._make_qkv(config, B=1)
        out = mem(q, k, v)
        loss = out.sum()
        loss.backward()

        assert mem.q_proj.weight.grad is not None
        assert mem.o_proj.weight.grad is not None

    def test_no_nan_output(self, config):
        """Output should not contain NaN values."""
        mem = AtlasMemory(config)
        q, k, v = self._make_qkv(config, B=2, T=16)
        out = mem(q, k, v)
        assert not torch.isnan(out).any()

    def test_different_sequence_lengths(self, config):
        """Should work with various sequence lengths."""
        mem = AtlasMemory(config)
        for T in [1, 4, 8, 16]:
            q, k, v = self._make_qkv(config, B=1, T=T)
            out = mem(q, k, v)
            assert out.shape == (1, T, config.n_heads, config.head_dim)

    def test_no_mode_distinction(self, config):
        """CS-10: No train/eval mode distinction."""
        mem = AtlasMemory(config)
        q, k, v = self._make_qkv(config, B=1)
        out1 = mem(q, k, v)
        out2 = mem(q, k, v)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_state_stays_bounded(self, config):
        """Verify state norms don't explode over T=64 tokens."""
        mem = AtlasMemory(config)
        q, k, v = self._make_qkv(config, B=1, T=64)
        out = mem(q, k, v)
        # Output should be bounded — no explosion
        assert out.abs().max() < 1e4
        assert not torch.isinf(out).any()

    def test_gate_init_matches_defaults(self, config):
        """Zero-weight gates should output scalar defaults at init.

        mu_gate(0) should produce mu=0.99, eta_gate(0) should produce eta=0.01.
        This ensures the model starts from the known-good fixed-parameter baseline.
        """
        mem = AtlasMemory(config)
        d = config.head_dim
        mu_min = config.lattice_mu_min
        mu_max = config.lattice_mu_max
        eta_max = config.lattice_eta_max

        with torch.no_grad():
            zero_in = torch.zeros(1, 1, d)
            mu_out = mu_min + (mu_max - mu_min) * torch.sigmoid(mem.mu_gate(zero_in))
            eta_out = eta_max * torch.sigmoid(mem.eta_gate(zero_in))

        assert abs(mu_out.item() - config.lattice_mu) < 1e-5, \
            f"mu_gate(0) = {mu_out.item()}, expected {config.lattice_mu}"
        assert abs(eta_out.item() - config.lattice_eta) < 1e-5, \
            f"eta_gate(0) = {eta_out.item()}, expected {config.lattice_eta}"

    def test_mu_clamping_bounds(self, config):
        """mu gate output must always be in [mu_min, mu_max] for any input."""
        mem = AtlasMemory(config)
        d = config.head_dim
        mu_min = config.lattice_mu_min
        mu_max = config.lattice_mu_max

        with torch.no_grad():
            # Extreme positive and negative inputs
            extreme = torch.cat([
                torch.full((1, 10, d), 1000.0),
                torch.full((1, 10, d), -1000.0),
                torch.randn(1, 100, d) * 10,
            ], dim=1)
            raw = mem.mu_gate(extreme).squeeze(-1)
            mu_out = mu_min + (mu_max - mu_min) * torch.sigmoid(raw)

        assert mu_out.min().item() >= mu_min - 1e-7, \
            f"mu below lower bound: {mu_out.min().item()}"
        assert mu_out.max().item() <= mu_max + 1e-7, \
            f"mu above upper bound: {mu_out.max().item()}"

    def test_eta_clamping_bounds(self, config):
        """eta gate output must always be in [0, eta_max] for any input."""
        mem = AtlasMemory(config)
        d = config.head_dim
        eta_max = config.lattice_eta_max

        with torch.no_grad():
            extreme = torch.cat([
                torch.full((1, 10, d), 1000.0),
                torch.full((1, 10, d), -1000.0),
                torch.randn(1, 100, d) * 10,
            ], dim=1)
            raw = mem.eta_gate(extreme).squeeze(-1)
            eta_out = eta_max * torch.sigmoid(raw)

        assert eta_out.min().item() >= -1e-7, \
            f"eta below 0: {eta_out.min().item()}"
        assert eta_out.max().item() <= eta_max + 1e-7, \
            f"eta above upper bound: {eta_out.max().item()}"

    def test_gate_gradients_flow(self, config):
        """Backward through memory must produce gradients on gate weights."""
        mem = AtlasMemory(config)
        q, k, v = self._make_qkv(config, B=1)
        out = mem(q, k, v)
        loss = out.sum()
        loss.backward()

        assert mem.mu_gate.weight.grad is not None, "mu_gate.weight has no gradient"
        assert mem.eta_gate.weight.grad is not None, "eta_gate.weight has no gradient"
        assert mem.mu_gate.bias.grad is not None, "mu_gate.bias has no gradient"
        assert mem.eta_gate.bias.grad is not None, "eta_gate.bias has no gradient"
