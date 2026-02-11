"""Tests for AtlasMAGModel."""

import torch
import pytest
from atlas.model import AtlasMAGModel
from atlas.config import tiny_config, full_config


class TestAtlasMAGModel:
    """Tests for the full Atlas-MAG language model."""

    def test_smoke_forward(self, config, batch):
        """Smoke test: forward pass produces logits."""
        model = AtlasMAGModel(config)
        out = model(batch)
        assert 'logits' in out
        assert out['logits'].shape == (2, 32, config.vocab_size)

    def test_smoke_forward_with_loss(self, config, batch):
        """Smoke test: forward with labels computes loss."""
        model = AtlasMAGModel(config)
        out = model(batch, labels=batch)
        assert 'loss' in out
        assert out['loss'].dim() == 0  # scalar
        assert out['loss'].item() > 0  # loss should be positive

    def test_backward(self, config, batch):
        """Backward pass should work without error."""
        model = AtlasMAGModel(config)
        out = model(batch, labels=batch)
        out['loss'].backward()

    def test_bilevel_gradient_flow(self, config):
        """Bilevel: memory init params must have non-None gradients after backward.

        The inner loop (TTL) modifies memory during forward.
        The outer loop must learn initial conditions via backprop.
        """
        model = AtlasMAGModel(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        out = model(input_ids, labels=input_ids)
        out['loss'].backward()

        # Check memory initialization parameters have gradients
        for i, block in enumerate(model.blocks):
            mem = block.memory
            for ell, S_init in enumerate(mem.S_inits):
                assert S_init.grad is not None, \
                    f"Block {i}, level {ell}: S_init has no gradient (bilevel broken)"

    def test_parameter_count_full(self):
        """Full config should produce ~54M parameters."""
        cfg = full_config()
        model = AtlasMAGModel(cfg)
        total = sum(p.numel() for p in model.parameters())
        assert 40_000_000 < total < 70_000_000, \
            f"Expected ~54M params, got {total/1e6:.1f}M"

    def test_parameter_count_tiny(self, config):
        """Tiny config should be small enough for CPU testing."""
        model = AtlasMAGModel(config)
        total = sum(p.numel() for p in model.parameters())
        assert total < 5_000_000, f"Tiny model too large: {total/1e6:.1f}M"

    def test_no_position_embedding(self, config):
        """impl-positional-encoding: No absolute positional embedding."""
        model = AtlasMAGModel(config)
        # Should not have a position embedding module
        assert not hasattr(model, 'pos_emb')
        assert not hasattr(model, 'position_embedding')

    def test_weight_tying(self, config):
        """LM head should share weights with token embedding."""
        model = AtlasMAGModel(config)
        assert model.lm_head.weight is model.token_emb.weight

    def test_cs38_no_training_names(self, config):
        """CS-38: No 'training' in class/method names (except nn.Module.training attr)."""
        model = AtlasMAGModel(config)
        for name, module in model.named_modules():
            class_name = type(module).__name__
            # CS-13: 'training' should not appear in class names
            assert 'training' not in class_name.lower() or class_name == 'Module', \
                f"CS-13 violation: class '{class_name}' contains 'training'"

    def test_different_batch_sizes(self, config):
        """Should work with various batch sizes."""
        model = AtlasMAGModel(config)
        for B in [1, 2, 4]:
            ids = torch.randint(0, config.vocab_size, (B, 8))
            out = model(ids)
            assert out['logits'].shape[0] == B
