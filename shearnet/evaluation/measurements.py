"""Raw measurements on rendered stamps: PSF moments, ngmix fits, metacal products.

Everything here returns what the measuring code produced and nothing derived
from it -- no responses, no corrections, no averages over objects. A fit that
fails is a row with its flag set and NaN values, never a dropped row and never a
zero (a zero is a plausible shape).

The ngmix work is done ``NGMIX_CHUNK`` objects at a time, each chunk with a
fresh fitter seeded the same way, so the result for an object does not depend on
how many objects are measured with it (only, through the worker pool, on the
worker count -- see :func:`shearnet.methods.ngmix.mp_fit_one_single`).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "METACAL_TYPES",
    "NGMIX_CHUNK",
    "FIT_FIELDS",
    "measure_psf",
    "stamp_observables",
    "fit_original",
    "metacal",
]

#: Objects per ngmix batch. An ``Observation`` is ~362 kB at a 53x53 stamp, so a
#: 200k population would need ~70 GB to hold at once; nothing downstream of a fit
#: needs more than a handful of numbers per object.
NGMIX_CHUNK = 4096

#: The nine metacal products, in the order their images are stacked: the
#: deconvolve/reconvolve ``noshear``, the four artificial shears of the galaxy,
#: and the four artificial shears of the dilated reconvolution PSF.
METACAL_TYPES = (
    "noshear",
    "1p", "1m", "2p", "2m",
    "1p_psf", "1m_psf", "2p_psf", "2m_psf",
)

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


def _empty_fit(n: int) -> Dict[str, np.ndarray]:
    out = {name: np.full((n,) + shape, np.nan) for name, shape in FIT_FIELDS.items()}
    out["flags"] = np.full(n, FLAG_MISSING, dtype=np.int32)
    return out


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
    import galsim
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
        except galsim.GalSimHSMError as exc:
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
# ngmix on the original stamp
# ----------------------------------------------------------------------
def _fit_one(runner, psf_runner, obs, index=0):
    """One plain fit; a failure is reported in the flags, not raised."""
    row = {name: np.full(shape, np.nan) for name, shape in FIT_FIELDS.items()}
    try:
        psf_runner.go(obs=obs.psf)
        res = runner.go(obs=obs)
    except Exception as exc:  # ngmix raises a zoo of errors on hopeless stamps
        logger.debug("ngmix fit failed on object %d: %s", index, exc)
        row["flags"] = FLAG_MISSING
        return row
    row["flags"] = int(res.get("flags", FLAG_MISSING))
    psf_result = obs.psf.meta.get("result")
    if psf_result is not None and psf_result.get("flags", 1) == 0:
        # a single-model PSF fit reports T; an EM mixture (psf_model: em5, as
        # SuperBIT uses) only carries the fitted mixture
        row["Tpsf"] = (psf_result["T"] if "T" in psf_result
                       else psf_result.get_gmix().get_T())
    if row["flags"] == 0:
        for name in ("g", "g_cov", "T", "flux", "s2n"):
            row[name] = np.asarray(res[name], dtype=float)
    return row


_POOL_RUNNERS = (None, None)


def _pool_init(runner, psf_runner):
    global _POOL_RUNNERS
    _POOL_RUNNERS = (runner, psf_runner)


def _pool_fit(obs):
    return _fit_one(_POOL_RUNNERS[0], _POOL_RUNNERS[1], obs)


def fit_original(observations: Sequence, *, seed: int, psf_model: str, gal_model: str,
                 nproc: Optional[int] = None) -> Dict[str, np.ndarray]:
    """ngmix fit of each original stamp (no metacal), one chunk's worth.

    Same fitter, guesser and tolerances as the metacal fits
    (:func:`shearnet.methods.ngmix.build_runners`), seeded with ``seed`` afresh
    for every call.
    """

    from ..methods.ngmix import build_runners
    from ..parallel import cpu_only_children, resolve_nproc, spawn_map

    n = len(observations)
    out = _empty_fit(n)
    if n == 0:
        return out
    rng = np.random.RandomState(seed)
    tguess = 4 * observations[0]._jacobian.get_scale() ** 2
    runner, psf_runner = build_runners(rng, psf_model=psf_model, gal_model=gal_model,
                                       Tguess=tguess)
    workers = resolve_nproc(nproc, n_tasks=n)
    if workers == 1:
        rows = [_fit_one(runner, psf_runner, obs, i) for i, obs in enumerate(observations)]
    else:
        with cpu_only_children():
            rows = list(spawn_map(_pool_fit, observations, workers, initializer=_pool_init,
                                  initargs=(runner, psf_runner), chunksize=64))
    for i, row in enumerate(rows):
        for name in FIT_FIELDS:
            out[name][i] = row[name]
    return out


# ----------------------------------------------------------------------
# metacal
# ----------------------------------------------------------------------
def metacal(observations: Sequence, *, seed: int, step: float, psf: str, psf_model: str,
            gal_model: str, nproc: Optional[int] = None, return_images: bool = False):
    """ngmix metacal on each observation: every product's fit, raw.

    Returns ``(fits, galaxy_stack, psf_stack)``. ``fits[t]`` holds
    :data:`FIT_FIELDS` for metacal type ``t``; a type ngmix did not return is
    flagged :data:`FLAG_MISSING`. With ``return_images`` the stacks are
    ``(N, 9, npix, npix)`` float32 -- the exact image/PSF pairs ngmix fitted, so
    another estimator can measure the same products -- else ``None``.
    """
    from ..methods.ngmix import _get_priors, mp_fit_one_single

    n = len(observations)
    fits = {t: _empty_fit(n) for t in METACAL_TYPES}
    if n == 0:
        return fits, None, None
    results, _ = mp_fit_one_single(
        observations,
        _get_priors(seed),
        np.random.RandomState(seed),
        psf_model=psf_model,
        gal_model=gal_model,
        mcal_pars={"psf": psf, "mcal_shear": step, "types": METACAL_TYPES},
        nproc=nproc,
        return_images=return_images,
    )
    rows = [r[0] for r in results] if return_images else results
    for i, struct in enumerate(rows):
        for row in struct:
            t = str(row["shear_type"])
            if t not in fits:
                continue
            fits[t]["flags"][i] = int(row["flags"])
            fits[t]["Tpsf"][i] = row["Tpsf"]
            if row["flags"] == 0:
                for name in ("g", "g_cov", "T", "flux", "s2n"):
                    fits[t][name][i] = row[name]
    if not return_images:
        return fits, None, None
    galaxy_stack = np.stack([r[1] for r in results], axis=0)
    psf_stack = np.stack([r[2] for r in results], axis=0)
    return fits, galaxy_stack, psf_stack
