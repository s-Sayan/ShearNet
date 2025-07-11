"""Core functionality for galaxy simulation, modeling, and training."""

from .dataset import generate_dataset, split_combined_images
from .models import OriginalGalaxyNN, EnhancedGalaxyNN, OriginalGalaxyResNet, GalaxyResNet, ResearchBackedGalaxyResNet, ForkLike
from .train import train_model, loss_fn, train_step, eval_step
from .attention import SpatialAttention

__all__ = [
    # Dataset
    "generate_dataset",
    "split_combined_images",
    # Models
    "OriginalGalaxyNN",
    "EnhancedGalaxyNN", 
    "OriginalGalaxyResNet",
    "GalaxyResNet",
    "ResearchBackedGalaxyResNet",
    "ForkLike",
    # Training
    "train_model",
    "loss_fn",
    "train_step", 
    "eval_step",
]