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

from ..logging_utils import get_logger

logger = get_logger(__name__)

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
    std = labels.std(axis=0)

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
    logger.info(f"Label normalizer saved to: {path}")


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
    logger.info(f"Label normalizer loaded from: {path}")
    _print_normalizer_stats(norm_params)
    return norm_params


# ===========================================================================
# Image (input) normalization -- INDEPENDENT of the label normalization above.
#
# The label normalizer (fit_normalizer/transform_labels) standardizes the
# network *outputs* (g1, g2, ...). The image normalizer below standardizes the
# network *inputs* (the galaxy/PSF stamps). They act on different tensors and do
# not interact: at inference you apply the image normalizer to the inputs, run
# the model, and apply inverse_transform_labels to the outputs.
#
# By design this is always DATASET-LEVEL (a single scalar mean/std per channel,
# computed once over the training set), never per-stamp -- so it is a fixed
# affine rescale that does not depend on any individual galaxy's flux, and the
# exact same constants are reused at train, eval and benchmarking time. The
# galaxy and PSF channels are standardized separately because their pixel scales
# differ by orders of magnitude (galaxy flux ~1e4 vs a unit-flux PSF).
# ---------------------------------------------------------------------------


def fit_image_normalizer(galaxy_images: np.ndarray, psf_images: np.ndarray = None) -> dict:
    """Compute dataset-level (scalar) mean/std for the galaxy (and PSF) stamps.

    Fit on the training portion only. Returns a single scalar mean and std per
    channel, so normalization is a fixed affine rescale identical for every
    stamp (never per-stamp).

    Parameters
    ----------
    galaxy_images : np.ndarray, shape (N, H, W)
    psf_images : np.ndarray, shape (N, H, W), optional
        PSF stamps for the two-branch models; ``None`` for single-branch.

    Returns
    -------
    img_params : dict
        ``{"gal_mean", "gal_std"}`` and, when ``psf_images`` is given,
        ``{"psf_mean", "psf_std"}`` (all Python floats).
    """
    gal_mean = float(np.mean(galaxy_images))
    gal_std = float(np.std(galaxy_images))
    gal_std = gal_std if gal_std > 1e-12 else 1.0
    params = {"gal_mean": gal_mean, "gal_std": gal_std}

    if psf_images is not None:
        psf_mean = float(np.mean(psf_images))
        psf_std = float(np.std(psf_images))
        psf_std = psf_std if psf_std > 1e-12 else 1.0
        params["psf_mean"] = psf_mean
        params["psf_std"] = psf_std

    _print_image_normalizer_stats(params)
    return params


def transform_images(images, img_params: dict, channel: str = "gal"):
    """Standardize a stamp array with the dataset-level constants for ``channel``.

    Parameters
    ----------
    images : array, shape (..., H, W)
    img_params : dict
        Output of :func:`fit_image_normalizer` (or :func:`load_image_normalizer`).
    channel : str
        ``"gal"`` or ``"psf"`` -- selects which stored mean/std to use.

    Returns
    -------
    Standardized array of the same shape. If ``channel`` stats are absent
    (e.g. ``"psf"`` for a single-branch run), the input is returned unchanged.
    """
    mkey, skey = f"{channel}_mean", f"{channel}_std"
    if mkey not in img_params or skey not in img_params:
        return images
    return (images - img_params[mkey]) / img_params[skey]


def save_image_normalizer(img_params: dict, path: str) -> None:
    """Save image normalization statistics to a ``.npz`` file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in img_params.items()})
    logger.info(f"Image normalizer saved to: {path}")


def load_image_normalizer(path: str) -> dict:
    """Load image normalization statistics from a ``.npz`` file."""
    data = np.load(path)
    img_params = {k: float(data[k]) for k in data.files}
    logger.info(f"Image normalizer loaded from: {path}")
    _print_image_normalizer_stats(img_params)
    return img_params


def maybe_normalize_images(galaxy_images, psf_images, model_dir):
    """Apply the saved image normalizer to ``(galaxy_images, psf_images)`` if present.

    Convenience for every call site (eval, benchmarks) that must reproduce the
    exact input scaling used at training. Looks for ``image_normalizer.npz`` in
    ``model_dir``; if absent (the default, image normalization off), the inputs
    are returned unchanged so existing runs are untouched.

    Returns
    -------
    (galaxy_images, psf_images) : possibly-normalized arrays. ``psf_images`` may
    be ``None`` and is passed through as such.
    """
    path = os.path.join(model_dir, "image_normalizer.npz")
    if not os.path.exists(path):
        return galaxy_images, psf_images
    img_params = load_image_normalizer(path)
    galaxy_images = transform_images(galaxy_images, img_params, channel="gal")
    if psf_images is not None:
        psf_images = transform_images(psf_images, img_params, channel="psf")
    return galaxy_images, psf_images


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _print_normalizer_stats(norm_params: dict) -> None:
    """Pretty-print per-parameter mean and std."""
    n = len(norm_params["mean"])
    names = _PARAM_NAMES[:n]
    logger.info("  Label normalization statistics:")
    for name, mu, sigma in zip(names, norm_params["mean"], norm_params["std"]):
        logger.info(f"    {name:>6}: mean = {mu:+.6e},  std = {sigma:.6e}")


def _print_image_normalizer_stats(img_params: dict) -> None:
    """Pretty-print the dataset-level image mean/std per channel."""
    logger.info("  Image normalization statistics (dataset-level):")
    for ch in ("gal", "psf"):
        mkey, skey = f"{ch}_mean", f"{ch}_std"
        if mkey in img_params:
            logger.info(
                f"    {ch:>6}: mean = {img_params[mkey]:+.6e},  std = {img_params[skey]:.6e}"
            )
