"""Command-line interface for training ShearNet models."""

import argparse
import logging
import os
import shutil

import jax.numpy as jnp
import jax.random as random

import shearnet.core.models

from ..config.config_handler import Config
from ..core.dataset import generate_dataset, split_combined_images
from ..core.specs import DatasetSpec, TrainConfig
from ..core.train import train_model
from ..utils.device import get_device
from ..utils.normalization import (
    fit_normalizer,
    save_normalizer,
    transform_labels,
)
from ..plotting import plot_learning_curve

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Suppress noisy absl logging emitted by JAX (importing the package also does
# this, before JAX is imported, so import-time messages are already silenced).
logging.getLogger("absl").setLevel(logging.ERROR)


def create_parser():
    """Create argument parser for training."""
    # Get the SHEARNET_DATA_PATH environment variable
    data_path = os.getenv("SHEARNET_DATA_PATH", os.path.abspath("."))

    # Set default save_path and plot_path
    default_save_path = os.path.join(data_path, "model_checkpoint")
    default_plot_path = os.path.join(data_path, "plots")

    parser = argparse.ArgumentParser(
        description="Train a galaxy shear estimation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Your original command (still works)
  shearnet-train --epochs 10 --batch_size 64 --samples 10000 --psf_fwhm 0.25 \
    --save --model_name cnn6 --plot --nn cnn --patience 20

  # Use config file
  shearnet-train --config configs/cnn6_experiment.yaml

  # Use config file but override specific values
  shearnet-train --config configs/cnn6_experiment.yaml --samples 20000 --model_name cnn6_big

  # Override multiple values
  shearnet-train --config configs/base.yaml --epochs 100 --nn resnet --save --plot
        """,
    )

    # Config file argument
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration file (optional)"
    )

    # For config overrides, use default=None so we can detect what user actually specified
    # When not using config, we'll use the defaults from the code
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--samples", type=int, default=None, help="Number of training samples.")
    parser.add_argument("--patience", type=int, default=None, help="Patience for early stopping.")
    parser.add_argument("--psf_fwhm", type=float, default=None, help="PSF sigma for simulation.")
    parser.add_argument("--nse_sd", type=float, default=None, help="noise sd for simulation.")
    parser.add_argument("--exp", type=str, default=None, help="Which experiment to run")
    parser.add_argument("--nn", type=str, default=None, help="Which model to use")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate for the learning rate scheduler",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=None, help="Weight decay for adamw optimizer"
    )
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model.")
    parser.add_argument(
        "--val_split", type=float, default=None, help="Validation split fraction (default: 0.2)"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=None, help="Evaluate every N epochs (default: 1)"
    )
    parser.add_argument("--psfex_model_file", type=str, default=None, help="psfex_model_file path")
    # Keep defaults for paths since they're computed
    parser.add_argument(
        "--save_path",
        type=str,
        default=default_save_path,
        help="Path to save the model parameters.",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default=default_plot_path,
        help="Path to save the learning curve plot.",
    )
    parser.add_argument(
        "--hlr_type",
        type=str,
        default="constant",
        help="hlr type can be constant or catalog. constant will be 0.5.",
    )
    parser.add_argument(
        "--flux_type",
        type=str,
        default="constant",
        help="hlr type can be constant or catalog. constant will be 12258.97.",
    )

    parser.add_argument(
        "--plot",
        action="store_const",
        const=True,
        default=None,
        help="Enable plotting (overrides config)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_const",
        const=False,
        dest="plot",
        help="Disable plotting (overrides config)",
    )
    parser.add_argument(
        "--stamp_size", type=int, default=None, help="Stamp size of the training data."
    )
    parser.add_argument(
        "--pixel_size", type=float, default=None, help="Pixel size of the training data."
    )

    parser.add_argument(
        "--process_psf",
        action="store_const",
        const=True,
        default=None,
        help="Process psf images on separate CNN branch.",
    )
    parser.add_argument(
        "--output_keys",
        type=tuple,
        default=("g1", "g2"),
        help="Please input a tuple of strings of either g1, g2, hlr, flux, psf_e1, psf_e2, psf_T",
    )
    parser.add_argument(
        "--gap",
        action="store_const",
        const=True,
        default=None,
        help="Global average pooling? Boolean.",
    )

    parser.add_argument(
        "--galaxy_type", type=str, default=None, help="Galaxy model type for fork-like models"
    )
    parser.add_argument(
        "--psf_type", type=str, default=None, help="PSF model type for fork-like models"
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default=None,
        help='Fusion strategy for fork-like model: "concat" (default) or "transformer"',
    )

    parser.add_argument(
        "--apply_psf_shear",
        action="store_const",
        const=True,
        default=None,
        help="Apply random shear to PSF images",
    )
    parser.add_argument(
        "--psf_shear_range",
        type=float,
        default=None,
        help="Maximum absolute shear value for PSF (default: 0.05)",
    )
    parser.add_argument(
        "--loss_weights",
        type=float,
        nargs="+",
        default=None,
        help="Per-output loss weights, one per output_key in order",
    )

    return parser


def _unit_tests_config(config):
    """Map a unit-tests-style config onto the legacy keys read below.

    Handles the meta/paths/image/psf/galaxy/train blocks; a no-op for
    legacy/default-schema configs.
    """
    if config.get("meta") is None and config.get("train") is None:
        return config
    mapping = {
        "train.samples": "dataset.samples",
        "train.seed": "dataset.seed",
        "image.noise_sd": "dataset.nse_sd",
        "image.stamp_size": "dataset.stamp_size",
        "image.pixel_scale": "dataset.pixel_size",
        "psf.gaussian_fwhm": "dataset.psf_fwhm",
        "psf.mode": "dataset.exp",
        "galaxy.hlr_type": "dataset.hlr_type",
        "galaxy.flux_type": "dataset.flux_type",
        "paths.psfex_model_file": "dataset.psfex_model_file",
        "model.architecture": "model.type",
        "model.galaxy_branch": "model.galaxy.type",
        "model.psf_branch": "model.psf.type",
        "train.epochs": "training.epochs",
        "train.batch_size": "training.batch_size",
        "train.learning_rate": "training.learning_rate",
        "train.weight_decay": "training.weight_decay",
        "train.patience": "training.patience",
        "train.val_split": "training.val_split",
        "train.eval_interval": "training.eval_interval",
        "train.loss_weights": "training.loss_weights",
        "meta.model_name": "output.model_name",
        "train.plot": "plotting.plot",
        "paths.train_catalog": "catalog.cosmos_cat_fname",
    }
    for src, dst in mapping.items():
        val = config.get(src)
        if val is not None:
            config._set_nested(dst, val)
    return config


# Argparse fallback defaults, used only when no ``--config`` file is given.
# These mirror ``config/default_config.yaml`` for every shared key (the
# ``test_cli_defaults_match_yaml`` test guards against drift); ``plot`` is the
# one intentional exception — the bare CLI does not plot unless asked.
_CLI_DEFAULTS = {
    "epochs": 10,
    "seed": 42,
    "batch_size": 32,
    "samples": 10000,
    "patience": 10,
    "psf_fwhm": 0.25,
    "nse_sd": 1e-5,
    "exp": "ideal",
    "nn": "cnn",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "model_name": "my_model",
    "plot": False,
    "val_split": 0.2,
    "eval_interval": 1,
    "stamp_size": 53,
    "pixel_size": 0.141,
    "process_psf": False,
    "galaxy_type": "research_backed",
    "psf_type": "forklens_psf",
    "fusion": "concat",
    "apply_psf_shear": False,
    "psf_shear_range": 0.05,
    "gap": False,
    "output_keys": ("g1", "g2"),
}


def _config_from_args(args):
    """Build a :class:`Config` from argparse values when no ``--config`` is given.

    Resolves every value as ``args.<x> if not None else _CLI_DEFAULTS[<x>]`` and
    writes it into a default-seeded ``Config`` (the original no-config behavior).
    """
    d = _CLI_DEFAULTS
    config = Config()  # Start with package defaults
    config._set_nested(
        "dataset.samples", args.samples if args.samples is not None else d["samples"]
    )
    config._set_nested(
        "dataset.psf_fwhm", args.psf_fwhm if args.psf_fwhm is not None else d["psf_fwhm"]
    )
    config._set_nested("dataset.nse_sd", args.nse_sd if args.nse_sd is not None else d["nse_sd"])
    config._set_nested("dataset.exp", args.exp if args.exp is not None else d["exp"])
    config._set_nested("dataset.seed", args.seed if args.seed is not None else d["seed"])
    config._set_nested(
        "dataset.stamp_size", args.stamp_size if args.stamp_size is not None else d["stamp_size"]
    )
    config._set_nested(
        "dataset.pixel_size", args.pixel_size if args.pixel_size is not None else d["pixel_size"]
    )
    config._set_nested("model.type", args.nn if args.nn is not None else d["nn"])
    config._set_nested("training.epochs", args.epochs if args.epochs is not None else d["epochs"])
    config._set_nested(
        "training.batch_size", args.batch_size if args.batch_size is not None else d["batch_size"]
    )
    config._set_nested(
        "training.learning_rate",
        args.learning_rate if args.learning_rate is not None else d["learning_rate"],
    )
    config._set_nested(
        "training.weight_decay",
        args.weight_decay if args.weight_decay is not None else d["weight_decay"],
    )
    config._set_nested(
        "training.patience", args.patience if args.patience is not None else d["patience"]
    )
    config._set_nested(
        "training.val_split", args.val_split if args.val_split is not None else d["val_split"]
    )
    config._set_nested(
        "training.eval_interval",
        args.eval_interval if args.eval_interval is not None else d["eval_interval"],
    )
    config._set_nested(
        "output.model_name", args.model_name if args.model_name is not None else d["model_name"]
    )
    config._set_nested("output.save_path", args.save_path)
    config._set_nested("output.plot_path", args.plot_path)
    config._set_nested("plotting.plot", args.plot)
    config._set_nested(
        "model.process_psf",
        args.process_psf if args.process_psf is not None else d["process_psf"],
    )
    config._set_nested(
        "model.galaxy.type", args.galaxy_type if args.galaxy_type is not None else d["galaxy_type"]
    )
    config._set_nested(
        "model.psf.type", args.psf_type if args.psf_type is not None else d["psf_type"]
    )
    config._set_nested("model.fusion", args.fusion if args.fusion is not None else d["fusion"])
    config._set_nested(
        "dataset.apply_psf_shear",
        args.apply_psf_shear if args.apply_psf_shear is not None else d["apply_psf_shear"],
    )
    config._set_nested(
        "dataset.psf_shear_range",
        args.psf_shear_range if args.psf_shear_range is not None else d["psf_shear_range"],
    )
    config._set_nested(
        "dataset.psfex_model_file",
        args.psfex_model_file if args.psfex_model_file is not None else d.get("psfex_model_file"),
    )
    config._set_nested(
        "model.output_keys", args.output_keys if args.output_keys is not None else d["output_keys"]
    )
    config._set_nested(
        "dataset.hlr_type", args.hlr_type if args.hlr_type is not None else d["hlr_type"]
    )
    config._set_nested(
        "dataset.flux_type", args.flux_type if args.flux_type is not None else d["flux_type"]
    )
    config._set_nested("model.gap", args.gap if args.gap is not None else d["gap"])
    config._set_nested(
        "training.loss_weights", args.loss_weights if args.loss_weights is not None else None
    )
    config._set_nested("catalog.cosmos_cat_fname", None)
    return config


def _apply_psf_model_fixup(config):
    """Reconcile ``process_psf`` with the chosen architecture, in place.

    ``process_psf`` requires the two-branch ``fork-like`` model; without it the
    ``fork-like`` model is unsupported. Mismatches are corrected (with a warning)
    exactly as the CLI did previously.
    """
    process_psf = config.get("model.process_psf")
    nn = config.get("model.type")
    if process_psf:
        if nn != "fork-like":
            logger.info(
                "\nWARNING: When --process-psf is enabled, it requires the fork-like model."
            )
            logger.info("Setting default fork-like model...")
            nn = "fork-like"
            galaxy_type = _CLI_DEFAULTS["galaxy_type"]
            psf_type = _CLI_DEFAULTS["psf_type"]
            config._set_nested("model.type", nn)
            config._set_nested("model.galaxy.type", galaxy_type)
            config._set_nested("model.psf.type", psf_type)
            logger.info(
                f"Model type changed to: '{nn}' with galaxy: '{galaxy_type}', psf: '{psf_type}'\n"
            )
    else:
        if nn == "fork-like":
            logger.info(
                "\nWARNING: When --process-psf is disabled, fork-like model is not supported."
            )
            logger.info("Setting default model...")
            nn = _CLI_DEFAULTS["nn"]
            config._set_nested("model.type", nn)
            logger.info(f"Model type changed to: '{nn}'\n")


def build_train_config(args):
    """Resolve a fully-populated :class:`Config` from parsed CLI ``args``.

    Handles both modes — loading ``--config`` (with optional CLI overrides and
    the unit-tests schema adapter) or, when no config file is given, falling back
    to ``_config_from_args`` — and applies the ``process_psf`` / ``fork-like``
    compatibility fixup. Performs no simulation or training, so it is unit-testable
    without the heavy GalSim/ngmix dependencies.
    """
    if args.config:
        config = Config(args.config)
        config = _unit_tests_config(config)
        config.update_from_args(args)
        logger.info(f"\nUsing config file: {args.config}")
        if any(getattr(args, k) is not None for k in _CLI_DEFAULTS.keys()):
            logger.info("With command-line overrides")
    else:
        config = _config_from_args(args)

    _apply_psf_model_fixup(config)
    return config


def _prepare_training_data(config):
    """Simulate the training set and z-score normalize its labels.

    Reads the dataset/model settings from ``config``, generates the stamps,
    splits off the PSF channel for the two-branch model, and fits the label
    normalizer on the training portion only.

    Returns:
        ``(galaxy_images, psf_images, labels, norm_params)`` where ``psf_images``
        is ``None`` for single-branch models and ``labels`` is already normalized.
    """
    spec = DatasetSpec.from_config(config)
    val_split = config.get("training.val_split")

    galaxy_images, labels = generate_dataset(**spec.as_kwargs())
    # Split off the PSF channel only when the two-branch model needs it;
    # single-branch models leave psf_images as None.
    psf_images = None
    if spec.return_psf:
        galaxy_images, psf_images = split_combined_images(
            galaxy_images, has_psf=True, has_clean=False
        )
        logger.info(f"Shape of train PSF images: {psf_images.shape}")
    logger.info(f"Shape of train images: {galaxy_images.shape}")
    logger.info(f"Shape of train labels: {labels.shape}")

    # Fit the normalizer on the training portion only, then transform all labels.
    split_idx = int(len(labels) * (1 - val_split))
    norm_params = fit_normalizer(labels[:split_idx])
    labels = transform_labels(labels, norm_params)
    return galaxy_images, psf_images, labels, norm_params


def _save_run_artifacts(config, model_dir, norm_params):
    """Persist the resolved config, label normalizer, and a model-source snapshot."""
    os.makedirs(model_dir, exist_ok=True)

    config_path = os.path.join(model_dir, "training_config.yaml")
    config.save(config_path)
    logger.info(f"\nTraining configuration saved to: {config_path}")

    normalizer_path = os.path.join(model_dir, "label_normalizer.npz")
    save_normalizer(norm_params, normalizer_path)

    try:
        models_source = shearnet.core.models.__file__
        architecture_dest = os.path.join(model_dir, "architecture.py")
        shutil.copy2(models_source, architecture_dest)
        logger.info(f"Model architecture saved to: {architecture_dest}")
    except Exception as e:
        logger.info(f"WARNING: Could not copy model architecture file: {e}")


def _save_losses(loss_path, train_loss, val_loss, val_loss_per_key, output_keys):
    """Save the per-epoch loss histories to ``loss_path`` (no-op if ``None``)."""
    logger.info("Saving training and validation loss...")
    if loss_path is None:
        return
    val_loss_per_key_arr = (
        jnp.stack(val_loss_per_key) if val_loss_per_key else jnp.zeros((0, len(output_keys)))
    )
    jnp.savez(
        loss_path,
        train_loss=train_loss,
        val_loss=val_loss,
        val_loss_per_key=val_loss_per_key_arr,
        output_keys=output_keys,
    )


def main():
    """Run the model-training command-line interface."""
    parser = create_parser()
    args = parser.parse_args()

    config = build_train_config(args)
    config.print_config()

    # Training/model settings used directly by main(); dataset settings are read
    # inside _prepare_training_data.
    output_keys = tuple(config.get("model.output_keys"))
    model_name = config.get("output.model_name")
    plot_flag = config.get("plotting.plot")

    save_path = os.path.abspath(args.save_path) if args.save_path else None
    plot_path = os.path.abspath(args.plot_path) if args.plot_path else None

    os.makedirs(save_path, exist_ok=True) if save_path else None
    os.makedirs(plot_path, exist_ok=True) if plot_path else None

    get_device()

    train_galaxy_images, train_psf_images, train_labels, norm_params = _prepare_training_data(
        config
    )

    rng_key = random.PRNGKey(config.get("dataset.seed"))

    model_dir = os.path.join(plot_path, model_name)
    _save_run_artifacts(config, model_dir, norm_params)

    train_cfg = TrainConfig.from_config(config, save_path=save_path)
    state, train_loss, val_loss, val_loss_per_key = train_model(
        train_galaxy_images,
        train_labels,
        rng_key,
        psf_images=train_psf_images,
        **train_cfg.as_kwargs(),
    )

    if plot_flag:
        logger.info("Plotting learning curve...")
        plot_save_path = (
            os.path.join(plot_path, model_name, "learning_curve.png") if plot_path else None
        )
        plot_learning_curve(val_loss, train_loss, plot_save_path)

    loss_path = os.path.join(plot_path, model_name, f"{model_name}_loss.npz") if plot_path else None
    _save_losses(loss_path, train_loss, val_loss, val_loss_per_key, output_keys)


if __name__ == "__main__":
    main()
