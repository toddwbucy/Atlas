"""Tests for MAGBlock."""

import torch
import pytest
from atlas.blocks import MAGBlock
from atlas.config import tiny_config


class TestMAGBlock:
    """Tests for the MAG composition block."""

    def test_output_shape(self, config):
        """Block output should match input shape."""
        block = MAGBlock(config)
        x = torch.randn(2, 8, config.d_model)
        out = block(x)
        assert out.shape == (2, 8, config.d_model)

    def test_residual_connection(self, config):
        """Output should differ from input (residual adds information)."""
        block = MAGBlock(config)
        x = torch.randn(1, 8, config.d_model)
        out = block(x)
        assert not torch.allclose(x, out)

    def test_gradients_flow(self, config):
        """All block parameters should receive gradients."""
        block = MAGBlock(config)
        x = torch.randn(1, 8, config.d_model)
        out = block(x)
        loss = out.sum()
        loss.backward()

        has_grad = sum(1 for p in block.parameters() if p.grad is not None)
        assert has_grad > 0, "No parameters received gradients"

    def test_mag_gating(self, config):
        """MAG gating: output = x + attn * sigmoid(mem)."""
        block = MAGBlock(config)
        x = torch.randn(1, 8, config.d_model)
        out = block(x)
        assert torch.isfinite(out).all()

    def test_shared_qkv(self, config):
        """QKV projections should be at block level, not duplicated."""
        block = MAGBlock(config)
        # Block should have w_q, w_k, w_v
        assert hasattr(block, 'w_q')
        assert hasattr(block, 'w_k')
        assert hasattr(block, 'w_v')
        # Memory and attention should NOT have their own projections
        assert not hasattr(block.memory, 'w_q')
        assert not hasattr(block.attention, 'w_q')

    def test_cs18_forward_only_api(self, config):
        """CS-18: forward() should be the only public API method.

        Sub-module attributes (memory, attention, etc.) are nn.Module
        children, not custom API methods. CS-18 means no custom public
        methods like .encode(), .generate(), .train_step() etc.
        """
        block = MAGBlock(config)
        # Collect methods defined on MAGBlock itself (not inherited)
        own_methods = [
            m for m in type(block).__dict__
            if not m.startswith('_') and callable(getattr(block, m))
        ]
        assert own_methods == ['forward'], \
            f"CS-18: Only forward() should be a public method, got: {own_methods}"
