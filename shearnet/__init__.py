"""ShearNet: Neural network-based galaxy shear estimation.

A Python library for estimating galaxy shears using neural networks,
with support for comparison against traditional methods like
metacalibration and NGmix.
"""

import logging

# Silence noisy absl logging before importing the JAX-backed submodules below;
# the re-export imports therefore intentionally follow this call (hence noqa).
logging.getLogger("absl").setLevel(logging.ERROR)

from .core.dataset import generate_dataset  # noqa: E402
from .core.models import (  # noqa: E402
    EnhancedGalaxyNN,
    GalaxyResNet,
    SimpleGalaxyNN,
)
from .core.train import train_model  # noqa: E402

__version__ = "0.1.0"
__author__ = "Sayan Saha"
__email__ = "sayan.iiserp@gmail.com"

__all__ = [
    "generate_dataset",
    "SimpleGalaxyNN",
    "EnhancedGalaxyNN",
    "GalaxyResNet",
    "train_model",
    "__version__",
]
