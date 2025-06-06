"""Utility functions for evaluation, plotting, and metrics."""

from .plot_helpers import (
    plot_residuals,
    visualize_samples,
    plot_true_vs_predicted,
    animate_model_epochs,
)
from .metrics import (
    eval_model,
    eval_ngmix,
    eval_mcal,
    loss_fn_eval,
    loss_fn_ngmix,
    loss_fn_mcal,
)


from .device import get_device
__all__ = [
    # Plotting
    "plot_residuals",
    "visualize_samples", 
    "plot_true_vs_predicted",
    "animate_model_epochs",
    # Metrics and evaluation
    "eval_model",
    "eval_ngmix",
    "eval_mcal",
    "loss_fn_eval",
    "loss_fn_ngmix",
    "loss_fn_mcal",
    # device detection
    "get_device"
]