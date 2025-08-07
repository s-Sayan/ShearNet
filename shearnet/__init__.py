"""ShearNet: Neural network-based galaxy shear estimation.

A Python library for estimating galaxy shears using neural networks,
with support for comparison against traditional methods like
metacalibration and NGmix.
"""

__version__ = "0.1.0"
__author__ = "Sayan Saha"
__email__ = "sayan.iiserp@gmail.com"
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
# Import main functionality for easy access
from .core.dataset import generate_dataset
from .core.models import SimpleGalaxyNN, EnhancedGalaxyNN, GalaxyResNet
from .core.train import train_model

__all__ = [
    "generate_dataset",
    "SimpleGalaxyNN",
    "EnhancedGalaxyNN", 
    "GalaxyResNet",
    "train_model",
    "__version__",
]