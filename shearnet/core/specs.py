"""Typed parameter groups for the sprawling dataset/training call signatures.

``generate_dataset`` and ``train_model`` take long lists of keyword arguments.
:class:`DatasetSpec` and :class:`TrainConfig` group the ones the CLI passes into
cohesive, documented objects built straight from a :class:`~shearnet.config.config_handler.Config`,
so call sites read as one object instead of a dozen positional/keyword arguments.

The underlying functions keep their existing signatures; ``as_kwargs()`` simply
expands a spec back into the keyword arguments they already accept.
"""

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

from .dataset import generate_dataset
from .train import train_model


@dataclass
class DatasetSpec:
    """Settings for :func:`shearnet.core.dataset.generate_dataset`.

    Field names match ``generate_dataset``'s keyword arguments, so
    ``generate_dataset(**spec.as_kwargs())`` is equivalent to passing them
    individually.
    """

    samples: int
    psf_fwhm: float
    exp: str = "ideal"
    seed: int = 42
    npix: int = 53
    scale: float = 0.141
    return_psf: bool = False
    nse_sd: float = 1e-5
    psf_file_or_dir: Optional[str] = None
    output_keys: Tuple[str, ...] = ("g1", "g2")
    hlr_type: str = "constant"
    flux_type: str = "constant"
    cosmos_cat_fname: Optional[str] = None
    compute_metacal: bool = False
    nproc: Optional[int] = None

    @classmethod
    def from_config(cls, config) -> "DatasetSpec":
        """Build a spec from a :class:`Config` (training-side key layout)."""
        return cls(
            samples=config.get("dataset.samples"),
            psf_fwhm=config.get("dataset.psf_fwhm"),
            exp=config.get("dataset.exp"),
            seed=config.get("dataset.seed"),
            npix=config.get("dataset.stamp_size"),
            scale=config.get("dataset.pixel_size"),
            return_psf=config.get("model.process_psf"),
            nse_sd=config.get("dataset.nse_sd"),
            psf_file_or_dir=config.get("dataset.psfex_model_file"),
            output_keys=tuple(config.get("model.output_keys")),
            hlr_type=config.get("dataset.hlr_type"),
            flux_type=config.get("dataset.flux_type"),
            cosmos_cat_fname=config.get("catalog.cosmos_cat_fname"),
            compute_metacal=config.get("dataset.compute_metacal", False),
            nproc=config.get("dataset.nproc", None),
        )

    def as_kwargs(self) -> dict:
        """Return the spec as keyword arguments for ``generate_dataset``."""
        return asdict(self)

    def build(self):
        """Generate the dataset described by this spec.

        Equivalent to ``generate_dataset(**spec.as_kwargs())``; lets callers drive
        simulation from one object instead of a dozen keyword arguments.
        """
        return generate_dataset(**self.as_kwargs())


@dataclass
class TrainConfig:
    """Hyperparameters for :func:`shearnet.core.train.train_model`.

    Excludes the data arrays (galaxy/psf images, labels, rng key), which are
    passed positionally; everything else is grouped here. Field names match
    ``train_model``'s keyword arguments.
    """

    epochs: int = 10
    batch_size: int = 32
    nn: str = "cnn"
    galaxy_type: str = "research_backed"
    psf_type: str = "forklens_psf"
    fusion: str = "concat"
    save_path: Optional[str] = None
    model_name: str = "my_model"
    val_split: float = 0.2
    eval_interval: int = 1
    patience: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    output_keys: Tuple[str, ...] = ("g1", "g2")
    gap: bool = False
    weights: Optional[list] = field(default=None)
    loss: str = "mse"
    ema_decay: Optional[float] = None

    @classmethod
    def from_config(cls, config, save_path=None) -> "TrainConfig":
        """Build a training config from a :class:`Config` (plus the save path)."""
        return cls(
            epochs=config.get("training.epochs"),
            batch_size=config.get("training.batch_size"),
            nn=config.get("model.type"),
            galaxy_type=config.get("model.galaxy.type"),
            psf_type=config.get("model.psf.type"),
            fusion=config.get("model.fusion", "concat"),
            save_path=save_path,
            model_name=config.get("output.model_name"),
            val_split=config.get("training.val_split"),
            eval_interval=config.get("training.eval_interval"),
            patience=config.get("training.patience"),
            lr=config.get("training.learning_rate"),
            weight_decay=config.get("training.weight_decay"),
            output_keys=tuple(config.get("model.output_keys")),
            gap=config.get("model.gap"),
            weights=config.get("training.loss_weights"),
            loss=config.get("training.loss", "mse"),
            ema_decay=config.get("training.ema_decay", None),
        )

    def as_kwargs(self) -> dict:
        """Return the config as keyword arguments for ``train_model``."""
        return asdict(self)

    def run(self, galaxy_images, labels, rng_key, psf_images=None):
        """Train a model with this configuration.

        Equivalent to ``train_model(galaxy_images, labels, rng_key,
        psf_images=psf_images, **cfg.as_kwargs())``; groups the ~16
        hyperparameters into one object at the call site.
        """
        return train_model(
            galaxy_images, labels, rng_key, psf_images=psf_images, **self.as_kwargs()
        )
