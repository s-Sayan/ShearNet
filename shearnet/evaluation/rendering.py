"""Render held-out evaluation stamps with the renderer the model trained on.

Every scene and every ring station goes through the one path here: draw the
per-object truth table (:func:`~shearnet.core.dataset_jax.sample_truth`, the
same draw-for-draw replica of the GalSim path that training uses), rotate the
source shape for a ring station, render, add the noise, cast to float32 like
the training stamps. Nothing about the population comes from anywhere but the
run's own resolved config, the evaluation seed and the evaluation catalog.

Pairing
-------
Object ``i`` is catalog row ``i`` and is seeded from ``i`` alone, and neither
the applied shear nor the rotation consumes a random number. So every scene of
one evaluation is the same galaxies, at the same offsets, behind the same PSFs,
with the same noise (rotated by quarter turns for the 90/180/270 stations, see
:meth:`EvaluationRenderer.render`). That is what lets any two scenes be
differenced object by object downstream.

Held out
--------
The seed must differ from the training seed and the catalog from the training
catalog; :mod:`shearnet.config.schema` refuses both. A different seed alone is
not enough: row ``i`` is row ``i`` whatever the seed.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..core.dataset import PSF_DATA_DIR, search_psf_files
from ..logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["EvaluationRenderer", "RenderedBlock"]


@dataclass
class RenderedBlock:
    """One scene at one ring station: stamps and the truth they were drawn from."""

    galaxy: np.ndarray        # (N, npix, npix) float32, noisy
    psf: np.ndarray           # (N, npix, npix) float32
    labels: np.ndarray        # (N, n_keys) the network targets, physical units
    truth: Dict[str, np.ndarray]

    @property
    def n(self) -> int:
        return len(self.galaxy)


class EvaluationRenderer:
    """Renders scenes for one run's evaluation config."""

    def __init__(self, config):
        if config.get("simulation.backend") != "jax-galsim":
            raise ValueError(
                "shearnet-eval needs simulation.backend: jax-galsim: the ring stations "
                "rotate the sampled truth table and the scenes set the applied shear on "
                f"it, which only the jax-galsim renderer exposes (run uses "
                f"{config.get('simulation.backend')!r})")
        from ..core.dataset_jax import JaxRenderConfig

        self.config = config
        self.output_keys = tuple(config.get("model.output_keys"))
        self.cfg = JaxRenderConfig(
            npix=config.get("simulation.stamp_size"),
            scale=config.get("simulation.pixel_scale"),
            psf_fwhm=config.get("simulation.psf.gaussian_fwhm"),
            gal_type=config.get("simulation.gal_model"),
            exp=config.get("simulation.psf.mode"),
            fft_size=config.get("simulation.jax_fft_size"),
            batch_size=config.get("simulation.jax_batch_size"),
        )
        self.seed = int(config.get("evaluation.seed"))
        self.nobj = int(config.get("evaluation.nobj"))
        self.catalog = config.get("simulation.catalogs.eval_file")
        psf = config.get("simulation.psf.psfex_file") or PSF_DATA_DIR
        self.psf_source = psf
        if self.cfg.exp == "superbit":
            self.psf_files = [psf] if os.path.isfile(psf) else list(search_psf_files(psf))
        else:
            self.psf_files = []
        self._psf_index = {path: i for i, path in enumerate(self.psf_files)}
        self._catalog_extra = self._read_catalog_extras()

    # ------------------------------------------------------------------
    def _read_catalog_extras(self) -> Dict[str, np.ndarray]:
        """The catalog's own axis ratio and position angle, if it has them."""
        if self.catalog is None:
            return {}
        from astropy.io import fits

        with fits.open(self.catalog) as hdul:
            data = hdul[1].data
            names = set(data.columns.names)
            if len(data) < self.nobj:
                raise ValueError(
                    f"evaluation.nobj is {self.nobj} but {self.catalog} has {len(data)} rows")
            out = {}
            for column, key in (("Q", "q_source"), ("PHI", "phi_source")):
                if column in names:
                    out[key] = np.asarray(data[column][: self.nobj], dtype=float)
        return out

    def describe_psf_files(self) -> List[str]:
        return list(self.psf_files)

    # ------------------------------------------------------------------
    def render(self, g1: float, g2: float, rotation_deg: float = 0.0) -> RenderedBlock:
        """The evaluation population under applied reduced shear ``(g1, g2)``.

        ``rotation_deg`` rotates every galaxy's *source* shape on the sky, for a
        ring station, keeping the applied shear and the PSF fixed:

        * the source ellipticity turns by ``exp(2i theta)`` (spin 2);
        * the sub-pixel offset turns by ``theta``, so the station is a rotation
          of the scene rather than the same galaxy moved;
        * the noise field turns by whole quarter turns (``theta // 90``), which
          gives the 90/180/270 stations their own realisation; 45 and 135 reuse
          the noise of 0 and 90 because a pixel grid has no 45-degree rotation;
        * the PSF does not turn -- every station sees the same PSF, so averaging
          over the ring cancels the source shape and keeps any PSF leakage.

        The labels are recomposed from the rotated source shape and the applied
        shear.
        """
        from ..core.dataset_jax import _build_labels, render_truth, sample_truth
        from ..core.shear_algebra import compose_shear

        config = self.config
        truth = sample_truth(
            self.nobj,
            self.cfg,
            seed=self.seed,
            nse_sd=config.get("simulation.noise_sigma"),
            apply_psf_shear=config.get("simulation.apply_psf_shear"),
            psf_shear_range=config.get("simulation.psf_shear_range"),
            base_shear_g1=float(g1),
            base_shear_g2=float(g2),
            base_shear_range=0.0,
            psf_file_or_dir=self.psf_source,
            hlr_type=config.get("simulation.hlr_type"),
            flux_type=config.get("simulation.flux_type"),
            cosmos_cat_fname=self.catalog,
            add_noise=True,
        )
        p = truth.params
        if rotation_deg:
            theta = math.radians(float(rotation_deg))
            cos2, sin2 = math.cos(2.0 * theta), math.sin(2.0 * theta)
            g1s, g2s = p["g1"], p["g2"]
            p["g1"], p["g2"] = cos2 * g1s - sin2 * g2s, sin2 * g1s + cos2 * g2s
            cos1, sin1 = math.cos(theta), math.sin(theta)
            dx, dy = p["dx"], p["dy"]
            p["dx"], p["dy"] = cos1 * dx - sin1 * dy, sin1 * dx + cos1 * dy
            quarter = int(float(rotation_deg) // 90.0) % 4
            if quarter:
                truth.noise = np.rot90(truth.noise, k=quarter, axes=(-2, -1)).copy()
            obs_g1, obs_g2 = compose_shear(p["g1"], p["g2"], p["base_g1"], p["base_g2"])
            truth.labels_raw = dict(truth.labels_raw, g1=obs_g1, g2=obs_g2)

        gal, psf = render_truth(truth, self.cfg,
                                trace_psf_shear=config.get("simulation.apply_psf_shear"))
        gal = gal + truth.noise
        labels = _build_labels(truth, psf, self.output_keys, self.cfg.scale)

        n = self.nobj
        psf_ids = np.full(n, -1, dtype=np.int32)
        if truth.psf_files:
            psf_ids = np.array([self._psf_index[path] for path in truth.psf_files],
                               dtype=np.int32)
        truth_table = {
            "catalog_row": np.arange(n, dtype=np.int64),
            "e_source": np.stack([p["g1"], p["g2"]], axis=1),
            "g_applied": np.stack([p["base_g1"], p["base_g2"]], axis=1),
            "e_prepsf": np.stack([truth.labels_raw["g1"], truth.labels_raw["g2"]], axis=1),
            "hlr": np.asarray(p["hlr"], dtype=float),
            "flux_model": np.asarray(p["flux"], dtype=float),
            "offset": np.stack([p["dx"], p["dy"]], axis=1),
            "psf_shear": np.stack([p["psf_g1"], p["psf_g2"]], axis=1),
            "psf_pos": np.stack([p["psf_x"], p["psf_y"]], axis=1),
            "psf_file_id": psf_ids,
        }
        for key in ("q_source", "phi_source"):
            truth_table[key] = self._catalog_extra.get(key, np.full(n, np.nan))
        return RenderedBlock(
            galaxy=gal.astype(np.float32),
            psf=psf.astype(np.float32),
            labels=np.asarray(labels, dtype=float),
            truth=truth_table,
        )

    def observations(self, galaxy_images, psf_images) -> list:
        """Package stamps for ngmix without altering a pixel.

        The galaxy weight is the inverse pixel-noise variance; the PSF stamp is
        noiseless, so its weight is a fixed fraction of its peak, which only
        regularises the PSF fit.
        """
        import ngmix

        centre = (galaxy_images.shape[1] - 1.0) / 2.0
        noise_sd = float(self.config.get("simulation.noise_sigma"))
        jacobian = ngmix.DiagonalJacobian(row=centre, col=centre, scale=self.cfg.scale)
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


def quarter_turns(rotation_deg: float) -> int:
    """How many quarter turns a station rotates its noise field by."""
    return int(float(rotation_deg) // 90.0) % 4
