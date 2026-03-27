"""Label normalization utilities for ShearNet.

Z-score normalizes each output parameter independently so the loss
function weights all parameters equally regardless of magnitude
differences (e.g., g1/g2 ~ O(0.1) vs flux ~ O(1e4)).

Usage
-----
    from shearnet.utils.normalization import (
        fit_normalizer,
        transform_labels,
        inverse_transform_labels,
        save_normalizer,
        load_normalizer,
    )

    # Training
    norm_params = fit_normalizer(train_labels)
    train_labels_norm = transform_labels(train_labels, norm_params)
    save_normalizer(norm_params, path)

    # Evaluation
    norm_params = load_normalizer(path)
    preds_physical = inverse_transform_labels(preds_norm, norm_params)
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Core parameter names (used for printing only; sliced to n_params)
# ---------------------------------------------------------------------------
_PARAM_NAMES = ["g1", "g2", "hlr", "flux"]


def fit_normalizer(labels: np.ndarray) -> dict:
    """Compute per-parameter mean and std from labels.

    Fits on the full array passed in, so pass only training labels
    (not validation) to avoid distributional leakage.

    Parameters
    ----------
    labels : np.ndarray, shape (N, n_params)
        Raw (physical-unit) labels from generate_dataset.

    Returns
    -------
    norm_params : dict
        {"mean": np.ndarray (n_params,), "std": np.ndarray (n_params,)}
    """
    mean = labels.mean(axis=0)
    std  = labels.std(axis=0)

    # Guard against zero std (e.g., a constant parameter like fixed flux)
    std = np.where(std < 1e-8, 1.0, std)

    norm_params = {"mean": mean, "std": std}
    _print_normalizer_stats(norm_params)
    return norm_params


def transform_labels(labels: np.ndarray, norm_params: dict) -> np.ndarray:
    """Z-score normalize labels to zero mean and unit variance.

    Parameters
    ----------
    labels : np.ndarray, shape (N, n_params)
    norm_params : dict
        Output of fit_normalizer.

    Returns
    -------
    np.ndarray, shape (N, n_params)
        Normalized labels.
    """
    return (labels - norm_params["mean"]) / norm_params["std"]


def inverse_transform_labels(labels_norm: np.ndarray, norm_params: dict) -> np.ndarray:
    """Denormalize predictions back to physical units.

    Parameters
    ----------
    labels_norm : np.ndarray, shape (N, n_params)
        Normalized predictions from the model.
    norm_params : dict
        Output of fit_normalizer (or load_normalizer).

    Returns
    -------
    np.ndarray, shape (N, n_params)
        Predictions in original physical units.
    """
    return labels_norm * norm_params["std"] + norm_params["mean"]


def save_normalizer(norm_params: dict, path: str) -> None:
    """Save normalization statistics to a .npz file.

    Parameters
    ----------
    norm_params : dict
        Output of fit_normalizer.
    path : str
        Destination path, e.g. "<plot_path>/<model_name>/label_normalizer.npz".
        Parent directories are created automatically.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, mean=norm_params["mean"], std=norm_params["std"])
    print(f"Label normalizer saved to: {path}")


def load_normalizer(path: str) -> dict:
    """Load normalization statistics from a .npz file.

    Parameters
    ----------
    path : str
        Path to a file previously saved by save_normalizer.

    Returns
    -------
    norm_params : dict
        {"mean": np.ndarray, "std": np.ndarray}
    """
    data = np.load(path)
    norm_params = {"mean": data["mean"], "std": data["std"]}
    print(f"Label normalizer loaded from: {path}")
    _print_normalizer_stats(norm_params)
    return norm_params


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _print_normalizer_stats(norm_params: dict) -> None:
    """Pretty-print per-parameter mean and std."""
    n = len(norm_params["mean"])
    names = _PARAM_NAMES[:n]
    print("  Label normalization statistics:")
    for name, mu, sigma in zip(names, norm_params["mean"], norm_params["std"]):
        print(f"    {name:>6}: mean = {mu:+.6e},  std = {sigma:.6e}")