"""Core functionality for galaxy simulation, modeling, and training."""

from .dataset import generate_dataset
from .models import SimpleGalaxyNN, EnhancedGalaxyNN, GalaxyResNet
from .train import train_model, loss_fn, train_step, eval_step

__all__ = [
    # Dataset
    "generate_dataset",
    # Models
    "SimpleGalaxyNN",
    "EnhancedGalaxyNN",
    "GalaxyResNet",
    # Training
    "train_model",
    "loss_fn",
    "train_step", 
    "eval_step",
]