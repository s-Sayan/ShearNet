"""Utility functions for evaluation, plotting, metrics, and devices."""

from .device import get_device
from .metrics import (
    eval_model,
    eval_ngmix,
    loss_fn_mcal,
    loss_fn_ngmix,
)
from ..plotting import (
    animate_model_epochs,
    plot_residuals,
    plot_true_vs_predicted,
    visualize_galaxy_samples,
    visualize_psf_samples,
)
from .simutils import (
    create_wcs_from_params,
)

__all__ = [
    # Plotting
    "plot_residuals",
    "visualize_galaxy_samples",
    "visualize_psf_samples",
    "plot_true_vs_predicted",
    "animate_model_epochs",
    # Metrics and evaluation
    "eval_model",
    "eval_ngmix",
    "loss_fn_ngmix",
    "loss_fn_mcal",
    # Simulation utilities
    "create_wcs_from_params",
    # Device utilities
    "get_device",
]
