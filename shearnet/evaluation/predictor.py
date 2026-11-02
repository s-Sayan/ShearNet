"""Run a finished model with exactly the transformations it trained with.

A prediction that forgets the image normalizer, the label normalizer or the
noise-unit conditioning does not measure the model; it measures the omission.
All three come from the run directory, and a run that should have one and does
not is an error rather than a guess. Predictions are returned in physical label
units, every output key, in the model's order.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..artifacts.checkpoints import build_model_from_config, init_variables, load_params
from ..artifacts.runs import RunDir
from ..config import Config
from ..core.models import is_fork_model
from ..logging_utils import get_logger
from ..utils.normalization import load_image_normalizer, load_normalizer

logger = get_logger(__name__)

__all__ = ["RunPredictor"]


class RunPredictor:
    """The model of one completed run, as a ``(galaxy, psf) -> predictions`` callable."""

    def __init__(self, run: RunDir, config: Config = None):
        import jax

        run.require_completed()
        self.run = run
        self.config = config if config is not None else Config.from_file(run.config_resolved)
        self.output_keys: Tuple[str, ...] = tuple(self.config.get("model.output_keys"))
        self.model = build_model_from_config(self.config)
        self.variables = load_params(run.best_params,
                                     init_variables(self.model, self.config))
        self.uses_psf = is_fork_model(self.config.get("model.type"))
        self.gap = self.config.get("model.gap")

        self.label_normalizer = load_normalizer(str(run.label_normalizer))
        saved_keys = self.label_normalizer.get("output_keys")
        if saved_keys is not None and tuple(saved_keys) != self.output_keys:
            raise ValueError(f"{run.label_normalizer} was fit for outputs {saved_keys}, "
                             f"the model predicts {self.output_keys}")
        wants_images = self.config.get("training.normalize_images")
        has_images = run.image_normalizer.is_file()
        if wants_images != has_images:
            raise ValueError(f"{run.root}: training.normalize_images is {wants_images} but "
                             f"normalizers/images.npz {'exists' if has_images else 'is missing'}")
        self.image_normalizer = (load_image_normalizer(str(run.image_normalizer))
                                 if has_images else None)
        noise = self.config.get("training.noise")
        self.noise_condition = bool(noise["condition"])
        if noise["min_sd"] is not None:
            # train_model_inloop validates at the middle of the range
            self.noise_sd = 0.5 * (float(noise["min_sd"]) + float(noise["max_sd"]))
        else:
            self.noise_sd = float(self.config.get("simulation.noise_sigma"))
        self._jitted = jax.jit(self._forward)

    def _forward(self, galaxy, psf):
        import jax.numpy as jnp

        if self.noise_condition:
            galaxy = galaxy / self.noise_sd
            psf = None if psf is None else psf / self.noise_sd
        if self.image_normalizer is not None:
            norm = self.image_normalizer
            galaxy = (galaxy - norm["gal_mean"]) / norm["gal_std"]
            if psf is not None and "psf_mean" in norm:
                psf = (psf - norm["psf_mean"]) / norm["psf_std"]
        inputs = (galaxy, psf) if self.uses_psf else (galaxy,)
        preds = self.model.apply(self.variables, *inputs, output_keys=self.output_keys,
                                 gap=self.gap, deterministic=True)
        return (preds * jnp.asarray(self.label_normalizer["std"])
                + jnp.asarray(self.label_normalizer["mean"]))

    def compile(self, row_counts, batch_size: int, galaxy_npix: int, psf_npix: int) -> float:
        """Compile the padded inference batch before measuring its runtime."""
        import time

        shapes = {batch_size} if any(row_counts) else set()
        start = time.perf_counter()
        for rows in sorted(shapes):
            galaxy = np.zeros((rows, galaxy_npix, galaxy_npix), np.float32)
            psf = np.zeros((rows, psf_npix, psf_npix), np.float32) if self.uses_psf else None
            self(galaxy, psf, batch_size)
        return time.perf_counter() - start

    def benchmark(self, galaxy_images, psf_images, batch_sizes, repeats: int = 5) -> dict:
        """Seconds per stamp of the network alone, ``{batch size: seconds}``.

        The stamps are put on the device once, so no host-device copy is timed.
        Each batch size is compiled and run once before it is timed, and every
        timed pass waits for the device to finish (``block_until_ready``). The
        result is the median of ``repeats`` passes over all the stamps that fill
        whole batches, per stamp. A batch size larger than the stamps, or one
        the device cannot hold, is skipped.
        """
        import time

        import jax

        if self.uses_psf and psf_images is None:
            raise ValueError("this model takes PSF stamps")
        results = {}
        for size in batch_sizes:
            batches = len(galaxy_images) // size
            if batches == 0:
                continue
            try:
                gal = [jax.device_put(np.asarray(galaxy_images[i * size:(i + 1) * size],
                                                 dtype=np.float32)) for i in range(batches)]
                psf = [None if psf_images is None else
                       jax.device_put(np.asarray(psf_images[i * size:(i + 1) * size],
                                                 dtype=np.float32)) for i in range(batches)]
                jax.block_until_ready(gal + [p for p in psf if p is not None])

                def one_pass():
                    jax.block_until_ready([self._jitted(g, p) for g, p in zip(gal, psf)])

                one_pass()  # compile and warm up
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    one_pass()
                    times.append(time.perf_counter() - start)
            except Exception as exc:  # out of device memory: this size does not fit
                if "RESOURCE_EXHAUSTED" not in str(exc) and "out of memory" not in str(exc):
                    raise
                logger.info("benchmark: batch size %d does not fit on the device", size)
                continue
            finally:
                gal = psf = None
            results[size] = float(np.median(times)) / (batches * size)
        return results

    def __call__(self, galaxy_images, psf_images, batch_size: int = 4096) -> np.ndarray:
        """``(N, len(output_keys))`` predictions, forwarded ``batch_size`` at a time."""
        import jax.numpy as jnp

        if self.uses_psf and psf_images is None:
            raise ValueError("this model takes PSF stamps")
        n = len(galaxy_images)
        out = []
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            count = sl.stop - sl.start
            galaxy = np.asarray(galaxy_images[sl], dtype=np.float32)
            psf = None if psf_images is None else np.asarray(psf_images[sl], dtype=np.float32)
            if count < batch_size:
                galaxy = np.pad(galaxy, ((0, batch_size - count), (0, 0), (0, 0)))
                if psf is not None:
                    psf = np.pad(psf, ((0, batch_size - count), (0, 0), (0, 0)))
            out.append(np.asarray(self._jitted(jnp.asarray(galaxy),
                       None if psf is None else jnp.asarray(psf)))[:count])
        if not out:
            return np.zeros((0, len(self.output_keys)))
        return np.concatenate(out, axis=0).astype(float)
