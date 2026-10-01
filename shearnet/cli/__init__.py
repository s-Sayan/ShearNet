"""Command-line interfaces: ``shearnet-train`` and ``shearnet-eval``."""

from .evaluate import main as eval_main
from .train import main as train_main

__all__ = ["train_main", "eval_main"]
