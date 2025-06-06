"""Command-line interfaces for ShearNet.

Provides command-line tools for training and evaluating galaxy
shear estimation models.
"""

from .train import main as train_main
from .evaluate import main as eval_main

__all__ = [
    "train_main",
    "eval_main",
]