"""Training-matched rendering and inference for the shear-bias benchmarks.

The research harness used to carry its own simulator. Nothing kept it in step
with the one training used, so a change to the draw method, the PSF FWHM or the
shear convention showed up as a multiplicative bias with no way to tell it apart
from a real one. This module removes that second simulator: benchmarks render
through :class:`~shearnet.core.specs.DatasetSpec` built from the *saved training
config*, so the evaluation population is produced by the same code path, on the
same backend, that produced the training population.

Held-out by construction
------------------------
Two things keep the benchmark from measuring the training set:

* **A different seed.** :meth:`TrainingMatchedRenderer.render` refuses a seed
  equal to ``dataset.seed``. The seed drives the per-object noise, sub-pixel
  shift, PSF shear and PSFEx draw.
* **A different catalogue.** The seed does *not* change which catalogue rows are
  used -- object ``i`` is row ``i`` either way -- so with one catalogue file the
  benchmark would re-render the training galaxies with fresh noise. The renderer
  therefore requires an explicit evaluation catalogue whenever the training run
  used one, and says so rather than quietly reusing it.

Pairing
-------
Both backends seed each object from its index, and the applied shear consumes no
random numbers. Rendering the same ``(samples, seed)`` at ``+gamma`` and
``-gamma`` therefore yields the *same galaxies with the same noise*, which is
what :func:`~shearnet.methods.anacal.paired_bias` needs and what makes the
finite-difference response protocol usable at finite S/N.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from .config.config_handler import Config
from .core.dataset import DatasetResult, generate_dataset, split_combined_images
from .core.specs import DatasetSpec
from .logging_utils import get_logger
from .utils.normalization import load_image_normalizer, load_normalizer

logger = get_logger(__name__)

__all__ = [
    "RenderedStamps",
    "SavedModelPredictor",
    "TrainingMatchedRenderer",
    "load_training_config",
    "model_dir_for",
]


@dataclass(frozen=True)
class RenderedStamps:
    """Images, labels and optional ngmix observations from one backend draw."""

    galaxy_images: np.ndarray
    psf_images: Optional[np.ndarray]
    labels: np.ndarray
    observations: Optional[list] = None

    @property
    def n(self):
        return len(self.galaxy_images)


def model_dir_for(model_name: str, config: Optional[Config] = None) -> Path:
    """Directory holding a run's saved config and normalizers."""
    if config is not None:
        plot_path = config.get("output.plot_path")
        if plot_path:
            return Path(plot_path) / model_name
    root = Path(os.getenv("SHEARNET_DATA_PATH", os.path.abspath(".")))
    return root / "plots" / model_name


def load_training_config(model_name: str, config_path: Optional[str] = None) -> Config:
    """Load a run's persisted ``training_config.yaml``, failing rather than guessing.

    Guessing here would silently benchmark one architecture's stamps against
    another's settings, so a missing file is an error with the path in it.
    """
    path = Path(config_path) if config_path else model_dir_for(model_name) / "training_config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved training config at {path}. Training writes it next to the "
            "normalizers; pass --training-config to point at it explicitly."
        )
    return Config(str(path))


class TrainingMatchedRenderer:
    """Render held-out benchmark stamps through the saved training backend."""

    def __init__(self, config: Config, eval_catalog: Optional[str] = None):
        self.config = config
        self.training_seed = int(config.get("dataset.seed"))
        self.gal_type = config.get("dataset.type", "exp")
        spec = DatasetSpec.from_config(config)
        train_catalog = spec.cosmos_cat_fname
        if eval_catalog is not None:
            if train_catalog and os.path.abspath(eval_catalog) == os.path.abspath(train_catalog):
                raise ValueError(
                    "The evaluation catalog is the training catalog "
                    f"({eval_catalog}). A different seed re-renders the same "
                    "galaxies with fresh noise; it does not hold them out."
                )
            spec = replace(spec, cosmos_cat_fname=eval_catalog)
        elif train_catalog:
            raise ValueError(
                "This run trained on a real catalog "
                f"({train_catalog}) but no evaluation catalog was given. Set "
                "paths.eval_catalog in the benchmark config: object i is row i "
                "regardless of seed, so without a separate catalog the benchmark "
                "measures the training galaxies."
            )
        self.spec = spec
        self.eval_catalog = eval_catalog

    @classmethod
    def from_saved_run(cls, model_name, config_path=None, eval_catalog=None):
        return cls(load_training_config(model_name, config_path), eval_catalog=eval_catalog)

    # -- rendering ------------------------------------------------------
    def render(
        self,
        samples: int,
        *,
        seed: int,
        base_shear_g1: float = 0.0,
        base_shear_g2: float = 0.0,
        psf_shear_g1: float = 0.0,
        psf_shear_g2: float = 0.0,
        intrinsic_rotation: float = 0.0,
        add_noise: bool = True,
        observations: bool = False,
    ) -> RenderedStamps:
        """Render held-out stamps, optionally packaged for the ngmix baseline.

        ``base_shear_*`` is the applied shear the bias measurement varies;
        ``psf_shear_*`` is an *extra* shear on the PSF, used only to measure
        ``R^PSF`` by finite difference. Everything else -- galaxies, sub-pixel
        offsets, noise -- is a function of ``(samples, seed)`` alone, so any two
        calls that share those are pair-matched.

        ``intrinsic_rotation`` (degrees) renders the same population with every
        galaxy's *intrinsic* shape rotated on the sky, which is what a ring test
        needs -- see :meth:`shear_pair`.
        """
        if int(seed) == self.training_seed:
            raise ValueError(
                f"Benchmark seed {seed} equals dataset.seed; the benchmark must "
                "not re-render the training population."
            )
        if samples < 1:
            raise ValueError("samples must be positive")

        kwargs = self.spec.as_kwargs()
        kwargs.update(
            {
                "samples": int(samples),
                "seed": int(seed),
                "type": self.gal_type,
                # FPFS and ngmix need the PSF stamp even when a single-branch
                # network ignores it, so benchmarks always keep the channel.
                "return_psf": True,
                "base_shear_g1": float(base_shear_g1),
                "base_shear_g2": float(base_shear_g2),
                "add_noise": bool(add_noise),
                "as_result": True,
            }
        )
        # NOT `apply_psf_shear = True` here. That flag draws a RANDOM per-object
        # PSF shear (uniform +/- psf_shear_range, rms 0.030 at the 0.05 default)
        # into the truth table. The R^PSF offset does not need it: it is added
        # straight onto truth.params below and `_render_jax` already renders
        # with trace_psf_shear=True. Switching it on instead moved the
        # measurement in two ways:
        #
        #   1. It evaluated d e / d e^PSF at a random +/- 0.05 base point rather
        #      than at the object's actual PSF -- while TRAINING measures the
        #      same derivative at zero offset, because the truth table there is
        #      drawn with apply_psf_shear from the config (unset = False). Same
        #      quantity, different point: a train/eval mismatch in exactly the
        #      number psf_weight optimises.
        #   2. The two extra RNG draws are consumed BEFORE dx/dy, so the
        #      perturbed renders got different galaxy centroids from the
        #      unperturbed one -- up to 0.325 arcsec, 2.3 pixels. Both legs of
        #      the difference shifted together so R^PSF itself stayed clean, but
        #      e_<est> and gpsf were then measured on a different object
        #      configuration from Rpsf_<est>_sim in the same LEAKAGE row.
        if self.spec.backend == "jax-galsim":
            # The benchmark sets the applied shear itself; the training-time
            # random base-shear augmentation would randomise what we are
            # measuring against.
            kwargs["base_shear_range"] = 0.0

        extra_psf_shear = (float(psf_shear_g1), float(psf_shear_g2))
        if self.spec.backend == "galsim":
            kwargs["return_obs"] = bool(observations)
            result = self._render_galsim(kwargs, extra_psf_shear, float(intrinsic_rotation))
        else:
            result = self._render_jax(kwargs, extra_psf_shear, float(intrinsic_rotation))

        images, labels, obs = result.images, result.labels, result.obs
        galaxy_images, psf_images = split_combined_images(images, has_psf=True, has_clean=False)
        if observations and obs is None:
            obs = self.observations_from_stamps(galaxy_images, psf_images)
        return RenderedStamps(
            galaxy_images=np.asarray(galaxy_images),
            psf_images=None if psf_images is None else np.asarray(psf_images),
            labels=np.asarray(labels),
            observations=obs,
        )

    def _render_galsim(self, kwargs, extra_psf_shear, intrinsic_rotation=0.0) -> DatasetResult:
        if any(extra_psf_shear) or intrinsic_rotation:
            raise NotImplementedError(
                "A PSF-shear offset for R^PSF and the intrinsic rotation for "
                "the ring test are only wired through the jax-galsim backend, "
                "which carries the per-object truth table (psf_g1/psf_g2, "
                "g1/g2, dx/dy) as explicit arrays. Use "
                "dataset.backend: jax-galsim."
            )
        result = generate_dataset(**kwargs)
        if not isinstance(result, DatasetResult):
            raise RuntimeError("generate_dataset(as_result=True) returned an unexpected value")
        return result

    def _render_jax(self, kwargs, extra_psf_shear, intrinsic_rotation=0.0) -> DatasetResult:
        from .core.dataset_jax import generate_dataset_jax

        if not any(extra_psf_shear) and not intrinsic_rotation:
            result = generate_dataset_jax(**kwargs)
            if not isinstance(result, DatasetResult):
                raise RuntimeError(
                    "generate_dataset_jax(as_result=True) returned an unexpected value"
                )
            return result
        return self._render_jax_perturbed(kwargs, extra_psf_shear, intrinsic_rotation)

    def _render_jax_perturbed(
        self, kwargs, extra_psf_shear, intrinsic_rotation=0.0
    ) -> DatasetResult:
        """Re-render after perturbing the sampled truth rather than the config.

        Two perturbations go through here, and both have to leave everything
        they do not name bit-identical to the unperturbed call:

        ``extra_psf_shear``
            an additive offset on every object's PSF shear, so the finite
            difference that gives ``R^PSF`` isolates the PSF.

        ``intrinsic_rotation``
            a rotation of every galaxy's *intrinsic* shape by ``theta`` degrees
            on the sky, for the ring test. Spelled out:

            * ``g -> g * exp(2 i theta)`` on the shape. Spin-2 doubles the
              angle, and the rotation is *exact* for the elliptical
              Gaussian/Exponential profiles rendered here -- the base profile is
              round, so the shape is all there is to rotate.
            * ``(dx, dy)`` rotates by ``theta`` itself. The sub-pixel offset is
              part of the scene, so rotating it keeps the twin a true rotation
              of the original rather than the same galaxy displaced differently.
            * ``noise -> rot90(noise, theta // 90)``. Rotating an i.i.d.
              Gaussian field about the stamp centre gives a field independent of
              the original at every pixel except the fixed centre one (1 of 2809
              at a 53-pixel stamp), so a 90- or 270-degree twin carries its own
              noise realisation. A grid cannot be rotated by 45 degrees, so the
              45- and 135-degree twins reuse the noise of their 0- and
              90-degree partners; that costs some of the noise averaging and
              nothing else, because the noise field is independent of the galaxy
              either way.
            * the **PSF is not rotated**. That is deliberate and it is the whole
              point: every member of the ring must see the *same* PSF, so that
              averaging over the ring cancels the intrinsic shape while leaving
              any PSF leakage intact and measurable.

            The labels are recomputed from the rotated intrinsic shape composed
            with the applied shear, because ``sample_truth`` fixed them at draw
            time and they would otherwise describe the unrotated galaxy.
        """
        import math

        import numpy as np

        from .core.dataset_jax import (
            JaxRenderConfig,
            _build_labels,
            render_truth,
            sample_truth,
        )
        from .core.shear_algebra import compose_shear

        cfg = JaxRenderConfig(
            npix=kwargs["npix"],
            scale=kwargs["scale"],
            psf_fwhm=kwargs["psf_fwhm"],
            gal_type=kwargs["type"],
            exp=kwargs["exp"],
            fft_size=kwargs["jax_fft_size"],
            batch_size=kwargs["jax_batch_size"],
        )
        truth = sample_truth(
            kwargs["samples"],
            cfg,
            seed=kwargs["seed"],
            nse_sd=kwargs["nse_sd"],
            apply_psf_shear=kwargs.get("apply_psf_shear", False),
            psf_shear_range=kwargs.get("psf_shear_range", 0.05),
            base_shear_g1=kwargs["base_shear_g1"],
            base_shear_g2=kwargs["base_shear_g2"],
            psf_file_or_dir=kwargs["psf_file_or_dir"],
            hlr_type=kwargs["hlr_type"],
            flux_type=kwargs["flux_type"],
            cosmos_cat_fname=kwargs["cosmos_cat_fname"],
            add_noise=kwargs["add_noise"],
        )
        truth.params["psf_g1"] = truth.params["psf_g1"] + extra_psf_shear[0]
        truth.params["psf_g2"] = truth.params["psf_g2"] + extra_psf_shear[1]
        if intrinsic_rotation:
            theta = math.radians(float(intrinsic_rotation))
            cos2, sin2 = math.cos(2.0 * theta), math.sin(2.0 * theta)
            g1, g2 = truth.params["g1"], truth.params["g2"]
            truth.params["g1"] = cos2 * g1 - sin2 * g2
            truth.params["g2"] = sin2 * g1 + cos2 * g2
            cos1, sin1 = math.cos(theta), math.sin(theta)
            dx, dy = truth.params["dx"], truth.params["dy"]
            truth.params["dx"] = cos1 * dx - sin1 * dy
            truth.params["dy"] = sin1 * dx + cos1 * dy
            quarter = int(float(intrinsic_rotation) // 90.0) % 4
            if quarter and truth.noise is not None:
                truth.noise = np.rot90(truth.noise, k=quarter, axes=(-2, -1)).copy()
            obs_g1, obs_g2 = compose_shear(
                truth.params["g1"], truth.params["g2"],
                truth.params["base_g1"], truth.params["base_g2"],
            )
            truth.labels_raw = dict(truth.labels_raw, g1=obs_g1, g2=obs_g2)
        gal, psf = render_truth(truth, cfg, trace_psf_shear=True)
        if kwargs["add_noise"]:
            gal = gal + truth.noise
        labels = _build_labels(truth, psf, kwargs["output_keys"], kwargs["scale"])
        images = np.stack([gal, psf], axis=-1)
        return DatasetResult(images=images, labels=labels, obs=None)

    def observations_from_stamps(self, galaxy_images, psf_images) -> list:
        """Package renderer stamps for ngmix without altering a pixel."""
        import ngmix

        centre = (galaxy_images.shape[1] - 1.0) / 2.0
        noise_sd = float(self.spec.nse_sd)
        jacobian = ngmix.DiagonalJacobian(row=centre, col=centre, scale=self.spec.scale)
        observations = []
        for galaxy, psf in zip(galaxy_images, psf_images):
            psf_noise = max(float(np.max(psf)) / 1000.0, np.finfo(float).tiny)
            psf_obs = ngmix.Observation(
                np.ascontiguousarray(psf, dtype=np.float64),
                weight=np.full(psf.shape, 1.0 / psf_noise**2),
                jacobian=jacobian,
            )
            observations.append(
                ngmix.Observation(
                    np.ascontiguousarray(galaxy, dtype=np.float64),
                    weight=np.full(galaxy.shape, 1.0 / max(noise_sd, 1e-12) ** 2),
                    jacobian=jacobian,
                    psf=psf_obs,
                )
            )
        return observations

    # -- the paired / response helpers the harness actually calls -------
    def shear_pair(
        self,
        samples: int,
        *,
        seed: int,
        shear: float,
        component: int = 0,
        intrinsic_rotation: float = 0.0,
        observations: bool = False,
    ) -> Tuple[RenderedStamps, RenderedStamps]:
        """Matched plus/minus applied-shear populations from one seed.

        ``intrinsic_rotation`` renders the *same* galaxies, under the same
        applied shear and the same PSF, with their intrinsic shapes rotated by
        that many degrees on the sky. Averaging a shape measurement over a ring
        of such rotations is the ring test of Nakajima & Bernstein (2007), and
        the two useful rings cancel different things:

        ``{0, 90}``
            the intrinsic ellipticity itself, since spin-2 makes 90 degrees a
            sign flip. This is shape-noise cancellation, and it is what makes
            ``c`` and the PSF-leakage slope precise.

        ``{0, 45}``
            the ``O(eps^2 gamma)`` term that survives the plus/minus difference,
            since 45 degrees sends ``eps -> i eps`` and so ``eps^2 -> -eps^2``.
            Measured on this composition: each rotation alone leaves
            ``m = +1.35e-04`` with a per-object scatter of 0.125 (an error of
            2.8e-04 at N = 200000), and averaging 0 with 45 cancels it to
            ``-1.2e-08`` with a per-object scatter of 5.4e-06 -- object by
            object, not merely in the mean, and a factor of 23000 down. What
            survives is the ``O(gamma^3 eps^4)`` term the ring does not reach.
            This is what makes the truth-referenced ``m`` precise.

        The full ring ``{0, 45, 90, 135}`` does both.
        """
        if component not in (0, 1):
            raise ValueError("component must be 0 (g1) or 1 (g2)")
        axis = ("base_shear_g1", "base_shear_g2")[component]
        common = dict(
            seed=seed, observations=observations,
            intrinsic_rotation=float(intrinsic_rotation),
        )
        return (
            self.render(samples, **common, **{axis: float(shear)}),
            self.render(samples, **common, **{axis: -float(shear)}),
        )

    def response_renderer(
        self,
        samples: int,
        *,
        seed: int,
        base_shear_g1: float = 0.0,
        base_shear_g2: float = 0.0,
        psf: bool = False,
        intrinsic_rotation: float = 0.0,
        observations: bool = False,
    ) -> Callable[[float, float], Tuple[np.ndarray, np.ndarray]]:
        """A ``render(d1, d2) -> (galaxy, psf)`` closure for the response protocol.

        Offsets go on the applied shear, or on the PSF shear when ``psf`` is set.
        Everything else is pinned, so
        :func:`~shearnet.methods.anacal.renderer_shear_response` differences a
        pure shear perturbation and the noise cancels exactly.
        """

        def render(d1: float, d2: float):
            common = dict(
                seed=seed,
                intrinsic_rotation=float(intrinsic_rotation),
                observations=observations,
            )
            if psf:
                stamps = self.render(
                    samples,
                    base_shear_g1=base_shear_g1,
                    base_shear_g2=base_shear_g2,
                    psf_shear_g1=d1,
                    psf_shear_g2=d2,
                    **common,
                )
            else:
                stamps = self.render(
                    samples,
                    base_shear_g1=base_shear_g1 + d1,
                    base_shear_g2=base_shear_g2 + d2,
                    **common,
                )
            return stamps.galaxy_images, stamps.psf_images

        return render


class SavedModelPredictor:
    """Run a saved model with exactly the input transformations it trained on.

    A benchmark that forgets the image normalizer, the label normalizer or the
    noise-unit conditioning does not measure the model; it measures the omission.
    All three are read from the run directory rather than reconstructed.

    Predictions come back in **physical** label units, so a response taken
    through :meth:`jax_predict` needs no separate normalizer bookkeeping.
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[Config] = None,
        config_path: Optional[str] = None,
        image_shape: Tuple[int, int] = None,
    ):
        import jax

        from .cli.evaluate import load_model
        from .core.models import is_fork_model

        self.config = (
            config if config is not None else load_training_config(model_name, config_path)
        )
        self.model_name = model_name
        self.output_keys = tuple(self.config.get("model.output_keys"))
        nn = self.config.get("model.type", "cnn")
        # A two-branch model always needs the PSF stamp, whatever
        # model.process_psf happens to say; load_model gates on the flag alone.
        self.process_psf = bool(self.config.get("model.process_psf", False)) or is_fork_model(nn)
        self.gap = bool(self.config.get("model.gap", False))
        noise = self.config.get("training.noise", {}) or {}
        self.noise_condition = bool(noise.get("condition", False))
        self.noise_sd = self._evaluation_noise_sd(noise)
        if image_shape is None:
            npix = int(self.config.get("dataset.stamp_size"))
            image_shape = (npix, npix)

        dummy = np.ones((1, *image_shape), dtype=np.float32)
        self.state = load_model(
            {
                "model_name": model_name,
                "seed": self.config.get("evaluation.seed", self.config.get("dataset.seed")),
                "nn": nn,
                "galaxy_type": self.config.get("model.galaxy.type", "research_backed"),
                "psf_type": self.config.get("model.psf.type", "forklens_psf"),
                "fusion": self.config.get("model.fusion", "concat"),
                "head": self.config.get("model.head", "gap"),
                "output_keys": self.output_keys,
                "process_psf": self.process_psf,
                "gap": self.gap,
                "branch_features": self.config.get("model.branch_features", None),
                "save_path": self.config.get("output.save_path"),
            },
            dummy,
            dummy if self.process_psf else None,
        )

        self.model_dir = model_dir_for(model_name, self.config)
        label_path = self.model_dir / "label_normalizer.npz"
        self.label_normalizer = load_normalizer(str(label_path)) if label_path.exists() else None
        image_path = self.model_dir / "image_normalizer.npz"
        self.image_normalizer = (
            load_image_normalizer(str(image_path)) if image_path.exists() else None
        )
        if self.label_normalizer is None:
            logger.warning(
                "No label_normalizer.npz in %s; assuming the run trained on raw "
                "labels. If it did not, every number below is off by the label "
                "scale.",
                self.model_dir,
            )
        self._jitted = jax.jit(self._forward)

    def _evaluation_noise_sd(self, noise) -> float:
        low, high = noise.get("min_sd"), noise.get("max_sd")
        if low is None or high is None:
            return float(self.config.get("dataset.nse_sd"))
        # Matches train_model_inloop's fixed validation depth.
        return (float(low) + float(high)) / 2.0

    def _forward(self, galaxy, psf):
        import jax.numpy as jnp

        if self.noise_condition:
            galaxy = galaxy / self.noise_sd
            psf = None if psf is None else psf / self.noise_sd
        if self.image_normalizer is not None:
            galaxy = (galaxy - self.image_normalizer["gal_mean"]) / self.image_normalizer["gal_std"]
            if psf is not None and "psf_mean" in self.image_normalizer:
                psf = (psf - self.image_normalizer["psf_mean"]) / self.image_normalizer["psf_std"]
        if self.process_psf:
            if psf is None:
                raise ValueError(f"model {self.model_name!r} expects PSF stamps")
            preds = self.state.apply_fn(
                self.state.params,
                galaxy,
                psf,
                output_keys=self.output_keys,
                gap=self.gap,
                deterministic=True,
            )
        else:
            preds = self.state.apply_fn(
                self.state.params,
                galaxy,
                output_keys=self.output_keys,
                gap=self.gap,
                deterministic=True,
            )
        if self.label_normalizer is not None:
            preds = preds * jnp.asarray(self.label_normalizer["std"]) + jnp.asarray(
                self.label_normalizer["mean"]
            )
        return preds

    def jax_predict(self, galaxy_images, psf_images=None):
        """JAX-traceable prediction in physical label units."""
        import jax.numpy as jnp

        galaxy = jnp.asarray(galaxy_images, dtype=jnp.float32)
        psf = None if psf_images is None else jnp.asarray(psf_images, dtype=jnp.float32)
        return self._forward(galaxy, psf)

    def __call__(self, galaxy_images, psf_images=None, batch_size: int = 4096) -> np.ndarray:
        """Predict in batches so a large benchmark does not exhaust the device."""
        import jax.numpy as jnp

        n = len(galaxy_images)
        out = []
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            out.append(
                np.asarray(
                    self._jitted(
                        jnp.asarray(galaxy_images[sl], dtype=jnp.float32),
                        None if psf_images is None else jnp.asarray(psf_images[sl], jnp.float32),
                    )
                )
            )
        return np.concatenate(out, axis=0)

    def shear_measure(self, batch_size: int = 4096):
        """A ``measure(galaxy, psf) -> (N, 2)`` callable for the response protocol."""
        shear_indices = [self.output_keys.index(k) for k in ("g1", "g2")]

        def measure(galaxy, psf):
            return self(galaxy, psf, batch_size=batch_size)[:, shear_indices]

        return measure
