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

from .dataset import PSF_DATA_DIR, generate_dataset
from .models import is_fork_model
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
    # Uniform per-object applied shear used by the differentiable response
    # losses.  Zero preserves the historic zero-shear training population.
    base_shear_range: float = 0.0
    # Random per-object PSF shear. ``cli.evaluate`` has always honoured
    # ``dataset.apply_psf_shear``; the training path dropped it on the floor
    # because the spec had no field for it, so a config asking for a sheared-PSF
    # population trained on round PSFs and evaluated on sheared ones.
    apply_psf_shear: bool = False
    psf_shear_range: float = 0.05
    psf_file_or_dir: Optional[str] = None
    output_keys: Tuple[str, ...] = ("g1", "g2")
    hlr_type: str = "constant"
    flux_type: str = "constant"
    cosmos_cat_fname: Optional[str] = None
    #: Galaxy light profile, ``'exp'`` or ``'gauss'``; ``type`` to the renderers.
    gal_type: str = "exp"
    compute_metacal: bool = False
    add_noise: bool = True
    nproc: Optional[int] = None
    # -- renderer selection -------------------------------------------------
    #: ``'galsim'`` (default, unchanged) or ``'jax-galsim'``. The jax-galsim
    #: backend renders the same stamps through a differentiable, batched
    #: ``jit(vmap(...))`` path; every per-object random draw is replicated, so
    #: switching backends changes the renderer and nothing else.
    backend: str = "galsim"
    #: Pinned FFT grid for the jax-galsim path. ``jit``/``vmap`` need static
    #: shapes, so there is no per-object sizing. Validated at Delta-m = -2e-05.
    jax_fft_size: int = 256
    #: Objects per batched render. Device memory scales as
    #: ``batch_size * fft_size**2`` -- raise one only by lowering the other.
    jax_batch_size: int = 256
    #: ``'upfront'`` (default) or ``'inloop'``. In-loop rendering happens inside
    #: the jitted training step; it needs ``backend='jax-galsim'`` and cannot
    #: produce the ``psf_*`` labels (they require an ngmix fit).
    generation: str = "upfront"

    #: Keys consumed only by the jax-galsim renderer.
    _JAX_ONLY = ("jax_fft_size", "jax_batch_size", "base_shear_range")
    #: Keys the jax-galsim renderer has no use for (no worker pool, and metacal
    #: reconvolutions are pointless for a backend built for analytic responses).
    _GALSIM_ONLY = ("nproc", "compute_metacal")

    @classmethod
    def from_config(cls, config, population: str = "training") -> "DatasetSpec":
        """Build a spec from a :class:`~shearnet.config.config_handler.Config`.

        ``population`` picks the catalog, seed and size: ``"training"`` reads
        ``simulation.catalogs.train_file`` / ``training.seed`` /
        ``training.nobj``. The evaluation renderer starts from the training
        spec and replaces those three itself.
        """
        if population != "training":
            raise ValueError("DatasetSpec.from_config builds the training population")
        return cls(
            samples=config.get("training.nobj"),
            psf_fwhm=config.get("simulation.psf.gaussian_fwhm"),
            exp=config.get("simulation.psf.mode"),
            seed=config.get("training.seed"),
            npix=config.get("simulation.stamp_size"),
            scale=config.get("simulation.pixel_scale"),
            # A fork model takes a (galaxy, PSF) pair, so it needs the PSF
            # stamps rendered; a single-branch model has nowhere to put them.
            return_psf=is_fork_model(config.get("model.type")),
            nse_sd=config.get("simulation.noise_sigma"),
            base_shear_range=config.get("training.base_shear_range"),
            apply_psf_shear=config.get("simulation.apply_psf_shear"),
            psf_shear_range=config.get("simulation.psf_shear_range"),
            # null means the PSFEx library bundled with the repository
            psf_file_or_dir=config.get("simulation.psf.psfex_file") or PSF_DATA_DIR,
            output_keys=tuple(config.get("model.output_keys")),
            hlr_type=config.get("simulation.hlr_type"),
            flux_type=config.get("simulation.flux_type"),
            cosmos_cat_fname=config.get("simulation.catalogs.train_file"),
            gal_type=config.get("simulation.gal_model"),
            # Fresh-noise training generates noise-free stamps here and re-draws
            # noise every epoch in train_model, so bake no noise in at gen time.
            add_noise=not config.get("training.resample_noise"),
            nproc=config.get("run_options.ncores"),
            backend=config.get("simulation.backend"),
            jax_fft_size=config.get("simulation.jax_fft_size"),
            jax_batch_size=config.get("simulation.jax_batch_size"),
            generation=config.get("training.generation"),
        )

    def __post_init__(self):
        if self.backend not in ("galsim", "jax-galsim"):
            raise ValueError(
                f"dataset.backend must be 'galsim' or 'jax-galsim', " f"got {self.backend!r}"
            )
        if self.generation not in ("upfront", "inloop"):
            raise ValueError(
                f"dataset.generation must be 'upfront' or 'inloop', " f"got {self.generation!r}"
            )
        if self.generation == "inloop" and self.backend != "jax-galsim":
            raise ValueError(
                "dataset.generation: inloop renders inside the jitted training "
                "step and therefore requires dataset.backend: jax-galsim "
                f"(got {self.backend!r})."
            )

    def as_kwargs(self) -> dict:
        """Return the spec as keyword arguments for the selected backend.

        Backend-specific keys are dropped so each renderer sees only arguments
        it accepts, and ``backend`` itself never reaches the callee.
        """
        kwargs = asdict(self)
        kwargs.pop("backend", None)
        kwargs.pop("generation", None)
        kwargs["type"] = kwargs.pop("gal_type")
        drop = self._GALSIM_ONLY if self.backend == "jax-galsim" else self._JAX_ONLY
        for key in drop:
            kwargs.pop(key, None)
        return kwargs

    def build(self):
        """Generate the dataset described by this spec, on the chosen backend.

        ``backend='galsim'`` calls :func:`~shearnet.core.dataset.generate_dataset`;
        ``backend='jax-galsim'`` calls
        :func:`~shearnet.core.dataset_jax.generate_dataset_jax`, which returns
        the same shapes. The jax-galsim import is deferred to here so the
        dependency is only required when the backend is actually selected.
        """
        if self.generation == "inloop":
            raise ValueError(
                "build() materialises a dataset, which dataset.generation: "
                "inloop deliberately never does -- use build_inloop_generator()."
            )
        if self.backend == "jax-galsim":
            from .dataset_jax import generate_dataset_jax

            return generate_dataset_jax(**self.as_kwargs())
        return generate_dataset(**self.as_kwargs())

    def build_inloop_generator(self, batch_size: int):
        """Sample the truth table and wrap it for in-loop rendering.

        No stamps are produced here: only the per-object truth (a few tens of MB
        for 500k objects) plus the device-resident PSFEx bank.
        """
        if self.generation != "inloop":
            raise ValueError("build_inloop_generator() requires dataset.generation: inloop")
        from .dataset_jax import JaxRenderConfig
        from .inloop import InLoopGenerator, sample_truth

        cfg = JaxRenderConfig(
            npix=self.npix,
            scale=self.scale,
            psf_fwhm=self.psf_fwhm,
            gal_type=self.gal_type,
            exp=self.exp,
            fft_size=self.jax_fft_size,
            batch_size=self.jax_batch_size,
        )
        truth = sample_truth(
            self.samples,
            cfg,
            seed=self.seed,
            nse_sd=self.nse_sd,
            base_shear_range=self.base_shear_range,
            apply_psf_shear=self.apply_psf_shear,
            psf_shear_range=self.psf_shear_range,
            psf_file_or_dir=self.psf_file_or_dir,
            hlr_type=self.hlr_type,
            flux_type=self.flux_type,
            cosmos_cat_fname=self.cosmos_cat_fname,
            add_noise=False,  # noise is drawn inside the step
        )
        return InLoopGenerator(truth, cfg, batch_size)


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
    head: str = "gap"
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
    dropout: float = 0.0
    #: Channel widths of the ``d4cnn`` backbone inside ``d4-fork-like``. None
    #: keeps the default (16, 32); Lin et al. (2026) use five layers at base 32.
    branch_features: Optional[list] = field(default=None)
    #: The 'shearnet-d4' schedule and the fusion/head sizes. Carried here so an
    #: up-front run can describe the SAME network as an in-loop one -- a
    #: d4-fork-like arm with generation: upfront otherwise fell back to the
    #: defaults silently, training a model its config did not name.
    d4_features: Optional[list] = field(default=None)
    d4_depths_galaxy: Optional[list] = field(default=None)
    d4_depths_psf: Optional[list] = field(default=None)
    d4_multiscale: Optional[bool] = None
    orbit_scan: bool = True
    fusion_pos: str = "learned"
    design: Optional[str] = None
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    num_pool_heads: Optional[int] = None
    num_self_attn_layers: Optional[int] = None
    ffn_dim: Optional[int] = None
    resample_noise: bool = False
    # Per-step noise std in *model-input* units (physical nse_sd / image gal_std).
    # Not read from config -- computed in cli.train and injected before run().
    resample_noise_sd: float = 0.0

    @classmethod
    def from_config(cls, config) -> "TrainConfig":
        """Build a training config from a :class:`Config`."""
        return cls(
            epochs=config.get("training.epochs"),
            batch_size=config.get("training.batch_size"),
            nn=config.get("model.type"),
            galaxy_type=config.get("model.galaxy_branch"),
            psf_type=config.get("model.psf_branch"),
            fusion=config.get("model.fusion"),
            head=config.get("model.head"),
            val_split=config.get("training.val_split"),
            eval_interval=config.get("training.eval_interval"),
            patience=config.get("training.patience"),
            lr=config.get("training.learning_rate"),
            weight_decay=config.get("training.weight_decay"),
            output_keys=tuple(config.get("model.output_keys")),
            gap=config.get("model.gap"),
            weights=config.get("training.loss_weights"),
            loss=config.get("training.loss"),
            ema_decay=config.get("training.ema_decay"),
            dropout=config.get("model.dropout"),
            branch_features=config.get("model.branch_features"),
            d4_features=config.get("model.d4_features"),
            d4_depths_galaxy=config.get("model.d4_depths_galaxy"),
            d4_depths_psf=config.get("model.d4_depths_psf"),
            d4_multiscale=config.get("model.d4_multiscale"),
            orbit_scan=config.get("model.orbit_scan"),
            fusion_pos=config.get("model.fusion_pos"),
            design=config.get("model.design"),
            d_model=config.get("model.d_model"),
            num_heads=config.get("model.num_heads"),
            num_pool_heads=config.get("model.num_pool_heads"),
            num_self_attn_layers=config.get("model.num_self_attn_layers"),
            ffn_dim=config.get("model.ffn_dim"),
            resample_noise=config.get("training.resample_noise"),
        )

    def as_kwargs(self) -> dict:
        """Return the config as keyword arguments for ``train_model``."""
        return asdict(self)

    def run(self, galaxy_images, labels, rng_key, psf_images=None, checkpoint_fn=None,
            history_fn=None):
        """Train a model with this configuration.

        Equivalent to ``train_model(galaxy_images, labels, rng_key,
        psf_images=psf_images, **cfg.as_kwargs())`` plus the two hooks.
        """
        return train_model(
            galaxy_images, labels, rng_key, psf_images=psf_images,
            checkpoint_fn=checkpoint_fn, history_fn=history_fn, **self.as_kwargs()
        )
