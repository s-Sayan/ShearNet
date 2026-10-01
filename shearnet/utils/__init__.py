"""Utility functions for evaluation, plotting, metrics, and devices.

The re-exports below resolve on first use, so importing a submodule such as
``shearnet.utils.normalization`` does not drag in ngmix and matplotlib.
"""

import importlib

_EXPORTS = {
    # Plotting
    "plot_residuals": "..plotting",
    "visualize_galaxy_samples": "..plotting",
    "visualize_psf_samples": "..plotting",
    "plot_true_vs_predicted": "..plotting",
    "animate_model_epochs": "..plotting",
    # Metrics and evaluation
    "eval_model": "..metrics",
    "eval_ngmix": "..metrics",
    "loss_fn_ngmix": "..metrics",
    "loss_fn_mcal": "..metrics",
    # Simulation utilities
    "create_wcs_from_params": ".simutils",
    # Device utilities
    "get_device": ".device",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
