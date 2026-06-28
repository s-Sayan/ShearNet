"""Utility functions for evaluation, plotting, metrics, and devices."""

from .plot_helpers import (
    plot_residuals,
    visualize_galaxy_samples,
    visualize_psf_samples,
    plot_true_vs_predicted,
    animate_model_epochs,
)

from .metrics import (
    eval_model,
    eval_ngmix,
    loss_fn_ngmix,
    loss_fn_mcal,
)

from .simutils import (
    create_wcs_from_params,
)

from .device import get_device

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
