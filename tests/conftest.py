"""Shared fixtures for Atlas tests."""

import pytest
import torch
from atlas.config import tiny_config, AtlasConfig


@pytest.fixture
def config():
    """Tiny config for CPU testing."""
    return tiny_config()


@pytest.fixture
def batch():
    """Tiny batch of random input_ids."""
    cfg = tiny_config()
    return torch.randint(0, cfg.vocab_size, (2, 32))
