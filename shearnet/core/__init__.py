"""Core functionality for galaxy simulation, modeling, and training."""

from .dataset import generate_dataset
from .models import OriginalGalaxyNN, EnhancedGalaxyNN, OriginalGalaxyResNet, GalaxyResNet, ResearchBackedGalaxyResNet
from .train import train_model, loss_fn, train_step, eval_step
from .attention import SpatialAttention

__all__ = [
    # Dataset
    "generate_dataset",
    # Models
    "OriginalGalaxyNN",
    "EnhancedGalaxyNN", 
    "OriginalGalaxyResNet",
    "GalaxyResNet",
    "ResearchBackedGalaxyResNet",
    # Training
    "train_model",
    "loss_fn",
    "train_step", 
    "eval_step",
]