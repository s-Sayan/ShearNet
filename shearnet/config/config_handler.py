"""Layered YAML + command-line configuration handling for ShearNet."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def load_default_config() -> Dict[str, Any]:
    """Return the package default configuration as a plain dict.

    Single source of truth for defaults: both :class:`Config` and the CLI
    fallback defaults read from this file.
    """
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


class Config:
    """Layered configuration for the ShearNet CLIs.

    Loads ``config/default_config.yaml`` first, then deep-merges an optional
    user YAML on top, and finally lets command-line arguments override
    individual values. Output paths default to ``$SHEARNET_DATA_PATH`` (or the
    current directory) and are created on init.

    Values are read and written with dot-notation paths, e.g.
    ``config.get('training.epochs')`` or ``config._set_nested('dataset.seed', 0)``.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the config, optionally merging the YAML at ``config_path``."""
        self.default_config_path = DEFAULT_CONFIG_PATH
        self.config = self._load_config(config_path)
        self._normalize_schema()
        self._setup_paths()

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        # Load default config first
        with open(self.default_config_path, "r") as f:
            config = yaml.safe_load(f)

        # If custom config provided, update defaults
        if config_path is not None:
            with open(config_path, "r") as f:
                custom_config = yaml.safe_load(f)
            if custom_config:
                # Normalize deprecated key names before merging, so the rest of
                # the code only ever sees the canonical keys.
                self._migrate_legacy_keys(custom_config)
                # Deep merge custom config into default
                self._deep_merge(config, custom_config)

        return config

    def _migrate_legacy_keys(self, custom: Dict[str, Any]) -> None:
        """Rename deprecated keys in a user config dict to their canonical names.

        Maps the legacy ``dataset.psf_sigma`` onto the canonical
        ``dataset.psf_fwhm`` (unless the user also set ``psf_fwhm`` explicitly,
        in which case the canonical value wins). This is done on the user dict
        before merging, so a config written with ``psf_sigma`` behaves exactly
        as if it had used ``psf_fwhm`` -- previously ``psf_sigma`` was honored by
        evaluation but silently ignored by training.
        """
        dataset = custom.get("dataset")
        if isinstance(dataset, dict) and "psf_sigma" in dataset:
            sigma = dataset.pop("psf_sigma")
            dataset.setdefault("psf_fwhm", sigma)

    # Maps the alternate "unit-tests" schema (meta/paths/image/psf/galaxy/train
    # blocks) onto the canonical dataset/model/training/output keys read elsewhere.
    _SCHEMA_MAP = {
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
        "train.ema_decay": "training.ema_decay",
        "meta.model_name": "output.model_name",
        "train.plot": "plotting.plot",
        "paths.train_catalog": "catalog.cosmos_cat_fname",
        "image.normalize_images": "dataset.normalize_images",
        "train.normalize_labels": "dataset.normalize_labels",
        "train.nproc": "dataset.nproc",
        "train.compute_metacal": "dataset.compute_metacal",
    }

    def _normalize_schema(self) -> None:
        """Translate the unit-tests-style schema onto the canonical keys.

        A no-op for the default/legacy schema (detected by the absence of the
        ``meta`` and ``train`` top-level blocks). Centralizing this in ``Config``
        means both the train and eval entry points understand either schema.
        """
        if self.get("meta") is None and self.get("train") is None:
            return
        for src, dst in self._SCHEMA_MAP.items():
            val = self.get(src)
            if val is not None:
                self._set_nested(dst, val)

    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _setup_paths(self) -> None:
        """Set up default paths based on environment variables."""
        data_path = os.getenv("SHEARNET_DATA_PATH", os.path.abspath("."))

        if self.config["output"]["save_path"] is None:
            self.config["output"]["save_path"] = os.path.join(data_path, "model_checkpoint")

        if self.config["output"]["plot_path"] is None:
            self.config["output"]["plot_path"] = os.path.join(data_path, "plots")

        # Ensure paths exist
        os.makedirs(self.config["output"]["save_path"], exist_ok=True)
        os.makedirs(self.config["output"]["plot_path"], exist_ok=True)

    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update config with command-line arguments."""
        args_dict = vars(args)

        # Get the mapping for training mode
        mapping = self._get_train_mapping()

        # Update config with non-None arguments
        for arg_name, config_path in mapping.items():
            if arg_name in args_dict and args_dict[arg_name] is not None:
                self._set_nested(config_path, args_dict[arg_name])

    def _get_train_mapping(self) -> Dict[str, str]:
        """Get argument mapping for training mode."""
        return {
            # Dataset args
            "samples": "dataset.samples",
            "psf_fwhm": "dataset.psf_fwhm",
            "exp": "dataset.exp",
            "seed": "dataset.seed",
            "nse_sd": "dataset.nse_sd",
            "normalized": "dataset.normalized",
            "normalize_images": "dataset.normalize_images",
            "normalize_labels": "dataset.normalize_labels",
            "d4_augment": "dataset.d4_augment",
            "stamp_size": "dataset.stamp_size",
            "pixel_size": "dataset.pixel_size",
            "apply_psf_shear": "dataset.apply_psf_shear",
            "psf_shear_range": "dataset.psf_shear_range",
            # Model args
            "process_psf": "model.process_psf",
            "nn": "model.type",
            "galaxy_type": "model.galaxy.type",
            "psf_type": "model.psf.type",
            "fusion": "model.fusion",
            # Training args
            "epochs": "training.epochs",
            "batch_size": "training.batch_size",
            "learning_rate": "training.learning_rate",
            "weight_decay": "training.weight_decay",
            "patience": "training.patience",
            "loss": "training.loss",
            "ema_decay": "training.ema_decay",
            # Output args
            "save_path": "output.save_path",
            "plot_path": "output.plot_path",
            "model_name": "output.model_name",
            # Plotting args
            "plot": "plotting.plot",
        }

    def _set_nested(self, path: str, value: Any) -> None:
        """Set nested config value using dot notation."""
        keys = path.split(".")
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        keys = path.split(".")
        current = self.config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            # KeyError: a key is missing. TypeError: an intermediate value is
            # None or a scalar (not subscriptable), e.g. descending past a leaf.
            return default

    def save(self, path: str) -> None:
        """Save current config to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def print_config(self) -> None:
        """Print current configuration."""
        logger.info("\n" + "=" * 50)
        logger.info("Training Configuration")
        logger.info("=" * 50)

        for section in ["dataset", "model", "training", "output", "plotting"]:
            if section in self.config:
                logger.info(f"\n{section}:")
                for key, value in self.config[section].items():
                    logger.info(f"  {key}: {value}")
        logger.info("=" * 50 + "\n")

    def print_eval_config(self) -> None:
        """Print only evaluation-relevant configuration."""
        logger.info("\n" + "=" * 50)
        logger.info("Evaluation Configuration")
        logger.info("=" * 50)

        # Only print relevant sections for evaluation
        sections_to_print = ["evaluation", "model", "plotting", "comparison"]

        for section in sections_to_print:
            if section in self.config:
                logger.info(f"\n{section}:")
                for key, value in self.config[section].items():
                    logger.info(f"  {key}: {value}")
        logger.info("=" * 50 + "\n")
