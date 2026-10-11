"""PSF moments, stamp observables, and the ngmix galaxy fit."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "FIT_FIELDS",
    "FLAG_MISSING",
    "empty_fit",
    "fit_galaxy",
    "measure_psf",
    "stamp_observables",
]

#: The fit values kept for every ngmix measurement, with their per-object shape.
FIT_FIELDS = {
    "g": (2,),
    "g_cov": (2, 2),
    "T": (),
    "Tpsf": (),
    "flux": (),
    "s2n": (),
    "flags": (),
}

#: Flag for "ngmix returned no result of this type at all".
FLAG_MISSING = 1 << 30


# ----------------------------------------------------------------------
# what the stamps themselves say
# ----------------------------------------------------------------------
def measure_psf(psf_images, scale: float, seed: int = 0) -> Dict[str, np.ndarray]:
    """Adaptive moments of each PSF stamp.

    ngmix starts each fit from a randomized guess; it is seeded here with
    ``seed + i`` per object so the shapes reproduce exactly (unseeded, they
    move at the fit tolerance, ~1e-5, from one call to the next).

    ``psf_g`` is the ngmix adaptive-moment shape converted to the reduced-shear
    (epsilon) convention; ``psf_T_hsm`` is GalSim HSM's determinant size
    ``2 sigma^2`` (the ``psf_T`` label); ``psf_T_admom`` is the ngmix
    adaptive-moment trace. Both sizes are
    arcsec^2. ``psf_flags`` is non-zero where either fit failed.
    """
    import ngmix

    from ..core.moments import get_admoms_ngmix_fit

    n = len(psf_images)
    out = {
        "psf_g": np.full((n, 2), np.nan),
        "psf_T_hsm": np.full(n, np.nan),
        "psf_T_admom": np.full(n, np.nan),
        "psf_flags": np.ones(n, dtype=np.int32),
    }
    centre = (psf_images.shape[1] - 1.0) / 2.0
    jacobian = ngmix.DiagonalJacobian(row=centre, col=centre, scale=scale)
    for i, image in enumerate(psf_images):
        obs = ngmix.Observation(np.ascontiguousarray(image, dtype=np.float64),
                                jacobian=jacobian)
        try:
            fit = get_admoms_ngmix_fit(obs, reduced=True,
                                       rng=np.random.RandomState(seed + i))
        except Exception as exc:  # HSM and admom both raise on hopeless stamps
            logger.debug("PSF moments failed on object %d: %s", i, exc)
            continue
        out["psf_flags"][i] = fit["flags"]
        if fit["flags"] == 0:
            out["psf_g"][i] = (fit["e1"], fit["e2"])
            out["psf_T_hsm"][i] = fit["T"]
            out["psf_T_admom"][i] = fit["T_admom"]
    return out


def stamp_observables(galaxy_images, noise_sigma: float) -> Dict[str, np.ndarray]:
    """``flux_stamp`` (sum of the noisy stamp) and ``s2n_stamp``.

    ``s2n_stamp = sqrt(sum I^2) / sigma`` on the noisy stamp.
    It is not ngmix's ``s2n`` and not a matched
    filter S/N.
    """
    gal = np.asarray(galaxy_images, dtype=float)
    return {
        "flux_stamp": gal.sum(axis=(1, 2)),
        "s2n_stamp": np.sqrt(np.sum(gal**2, axis=(1, 2))) / max(float(noise_sigma), 1e-12),
    }


# ----------------------------------------------------------------------
# ngmix fits
# ----------------------------------------------------------------------
def empty_fit():
    """A fit row that failed: NaN values, flagged :data:`FLAG_MISSING`."""
    row = {name: np.full(shape, np.nan) for name, shape in FIT_FIELDS.items()}
    row["flags"] = FLAG_MISSING
    return row


def fit_galaxy(runner, obs, psf_result):
    """Fit ``obs`` with its PSF already fitted (``psf_result``, flags 0).

    The PSF result is attached to ``obs.psf`` the way ngmix's ``PSFRunner``
    would have, so the galaxy fit is the one a refit of that PSF would give.
    Errors propagate: the caller decides what a failed measurement records.
    """
    obs.psf.meta["result"] = psf_result
    gmix = psf_result.get_gmix()
    obs.psf.gmix = gmix
    row = empty_fit()
    # a single-model PSF fit reports T; an EM mixture only carries the mixture
    row["Tpsf"] = psf_result["T"] if "T" in psf_result else gmix.get_T()
    res = runner.go(obs=obs)
    row["flags"] = int(res.get("flags", FLAG_MISSING))
    if row["flags"] == 0:
        for name in ("g", "g_cov", "T", "flux", "s2n"):
            row[name] = np.asarray(res[name], dtype=float)
    return row
