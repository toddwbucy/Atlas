"""Atlas-MAG: Memory As Gate architecture with the Omega Rule."""

from .config import AtlasConfig, tiny_config, full_config
from .model import AtlasMAGModel
from .memory import AtlasMemory
from .attention import SlidingWindowAttention
from .blocks import MAGBlock
from .primitives import polynomial_features, newton_schulz5, omega_loss, lattice_update, lattice_readout

__all__ = [
    'AtlasConfig', 'tiny_config', 'full_config',
    'AtlasMAGModel',
    'AtlasMemory',
    'SlidingWindowAttention',
    'MAGBlock',
    'polynomial_features', 'newton_schulz5', 'omega_loss',
    'lattice_update', 'lattice_readout',
]
