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

from .deconv_metrics import (
    eval_ngmix_deconv,
    compare_deconv_methods,
    calculate_psnr,
    calculate_ssim
)
from .deconv_plots import (
    plot_deconv_samples,
    plot_deconv_metrics,
    plot_deconv_comparison,
    plot_deconv_residuals
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
    # Deconv evaluation
    "eval_deconv_model",
    "eval_fft_deconv",
    "compare_deconv_methods", 
    "calculate_psnr",
    "calculate_ssim",
    # Deconv plotting
    "plot_deconv_samples",
    "plot_deconv_metrics", 
    "plot_deconv_comparison",
    "plot_deconv_residuals"
]

from .notebook_output_system import (
    log_print,
    save_plot,
    log_array_stats,
    experiment_section,
    get_output_manager
)