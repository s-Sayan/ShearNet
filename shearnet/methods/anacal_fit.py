"""AnaCal's Gaussian model fit, with its analytical shear response.

Li (2026, arXiv:2506.16607) carries AnaCal's analytical pixel shear response
through a galaxy *model fit* using **quintuple numbers** -- a commutative ring
that propagates ``(value, d/dgamma1, d/dgamma2, d/dx1, d/dx2)`` through every
operation of the fit, the way dual numbers propagate a single derivative in
forward-mode autodiff. The fitted shape therefore arrives with its own exact
shear response attached, rather than needing one estimated by finite difference.

Two things to be clear about, because both are easy to get wrong:

**This is not esheldon/ngmix.** AnaCal's ``ngmix`` submodule is its own C++
Gaussian fitter (``include/anacal/ngmix/fitting.h``) and has no dependency on
the ngmix package. Wiring this does not give the ngmix baseline an analytic
response; it adds a *different* estimator that happens to fit the same model
family. The two are separate columns for that reason.

**The response is a derivative of the fit, not of the estimator you point it
at.** Only an estimator AnaCal can differentiate symbolically gets one. A neural
network would need the chain rule through ``d f_h / d gamma`` (Eq. 5 of the
paper), which requires feeding it the PSF-deconvolved, resmoothed ``f_h`` -- a
different input contract from ShearNet's.

Verified against ``anacal`` 0.8.1, whose ``anacal.ngmix`` submodule carries
this (repo ``mr-superonion/AnaCal`` at ec86c16; the README lists NGMIX Gaussian
model fitting as a supported analytical estimator). The API used here is
``GaussFit.process_block`` + ``table.galNumber`` + ``model.get_shape()``, which
returns a pair of ``math.qnumber`` -- ``(v, g1, g2, x1, x2)``, the quintuple.

Odd stamps are fine here, unlike the FPFS path: ``stamp_size`` is the fitting
window the fitter cuts around each source, not the size of the image handed in,
so a 53-pixel stamp needs no padding and no Fourier-origin alignment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..logging_utils import get_logger
from .anacal import ShapeMeasurement

logger = get_logger(__name__)

__all__ = ["gauss_fit_shapes", "measure_gauss_fit"]

#: Fitting window in pixels. Must be even; AnaCal cuts this around each source.
DEFAULT_STAMP_SIZE = 32
#: Iterations of the fit. AnaCal's own tests use 35.
DEFAULT_EPOCHS = 35


def _anacal_ngmix():
    try:
        import anacal
    except ImportError as exc:  # pragma: no cover - exercised by the skip
        raise ImportError(
            "AnaCal is required for the analytical model-fit response. "
            "Install it with `pip install anacal` (>= 0.8.1)."
        ) from exc
    if not hasattr(anacal, "ngmix"):
        raise ImportError(
            "This AnaCal build has no `ngmix` submodule, so it predates the "
            "quintuple-number model fitting of arXiv:2506.16607. Upgrade to "
            "anacal >= 0.8.1."
        )
    return anacal



#: The configured fitter, set by :func:`_fit_pool_init`.
_POOL_FIT = None


def _fit_pool_init(settings):
    """Build the fitter once, from plain numbers."""
    global _POOL_FIT
    anacal = _anacal_ngmix()
    scale, sigma_arcsec, stamp_size, num_epochs, variance, centre, flux, size = settings
    fitter = anacal.ngmix.GaussFit(
        scale=scale, sigma_arcsec=sigma_arcsec, stamp_size=stamp_size
    )
    _POOL_FIT = (anacal, fitter, anacal.ngmix.modelPrior(), num_epochs, variance,
                 centre, flux, size)


def _fit_one(galaxy, psf):
    """``(e, dedg)`` for one stamp, or None if the fit failed."""
    anacal, fitter, prior, num_epochs, variance, centre, flux, size = _POOL_FIT
    source = anacal.table.galNumber()
    source.model.x1.v = centre
    source.model.x2.v = centre
    source.x1_det = centre
    source.x2_det = centre
    source.model.F.v = flux
    source.model.t.v = size
    try:
        fitted = fitter.process_block(
            catalog=[source], img_array=galaxy, psf_array=psf, prior=prior,
            num_epochs=num_epochs, variance=variance,
        )[0]
        e1, e2 = fitted.model.get_shape()
    except Exception as exc:
        logger.debug("AnaCal model fit failed: %s", exc)
        return None
    # Each component is a quintuple number: .v is the value, .g1/.g2 the shear
    # response, .x1/.x2 the response to a shift of the reference point.
    values = np.array([e1.v, e2.v], dtype=float)
    response = np.array([[e1.g1, e1.g2], [e2.g1, e2.g2]], dtype=float)
    if not (np.isfinite(values).all() and np.isfinite(response).all()):
        return None
    return values, response


def measure_gauss_fit(
    galaxy_images: np.ndarray,
    psf_images: np.ndarray,
    *,
    pixel_scale: float,
    noise_variance: float,
    sigma_arcsec: float = 0.52,
    stamp_size: int = DEFAULT_STAMP_SIZE,
    num_epochs: int = DEFAULT_EPOCHS,
    flux_guess: float = 1.0,
    size_guess: float = -0.5,
    prior: Optional[object] = None,
) -> ShapeMeasurement:
    """Fit each stamp and return its shape *with* the analytic ``de/dgamma``.

    One fit per stamp rather than one per exposure, for the same reason the FPFS
    path works that way: ``process_block`` takes a single ``psf_array`` for the
    whole call, and the SuperBIT PSF varies per object. A correct per-object PSF
    is worth more than the throughput.

    Args:
        galaxy_images: ``(N, npix, npix)``. Odd ``npix`` is fine.
        psf_images: ``(N, npix, npix)``, one PSF per stamp.
        pixel_scale: arcsec/pixel.
        noise_variance: per-pixel variance of ``galaxy_images``.
        sigma_arcsec: width of AnaCal's Gaussian smoothing kernel.
        stamp_size: fitting window in pixels; must be even and no larger than
            the stamp.
        flux_guess, size_guess: the fit's starting point. ``size_guess`` is
            AnaCal's ``t``, a log-size, hence negative.

    The fit costs ~12.6 ms/object against ~6.9 ms for an ngmix plain fit, and
    the evaluation runs it six times over the population, so budget ~4.2 hours
    at 200k objects. See the note in the body for why that is not parallelised.

    Returns a :class:`~shearnet.methods.anacal.ShapeMeasurement` whose ``dedg``
    is the analytical response, not a finite difference.
    """
    _anacal_ngmix()  # fail early and clearly if the build is too old

    galaxy_images = np.ascontiguousarray(galaxy_images, dtype=np.float32)
    psf_images = np.ascontiguousarray(psf_images, dtype=np.float64)
    if galaxy_images.ndim != 3 or psf_images.shape != galaxy_images.shape:
        raise ValueError("galaxy_images and psf_images must share an (N, Y, X) shape")
    if galaxy_images.shape[1] != galaxy_images.shape[2]:
        raise ValueError("the model fit requires square stamps")
    if not noise_variance > 0.0:
        raise ValueError("the model fit requires a positive noise_variance")
    npix = galaxy_images.shape[1]
    if stamp_size % 2:
        raise ValueError(f"stamp_size must be even, got {stamp_size}")
    if stamp_size > npix:
        raise ValueError(f"stamp_size {stamp_size} exceeds the {npix}-pixel stamp")

    # GalSim puts the object on pixel (npix - 1) // 2 of an odd stamp; AnaCal
    # wants that position in arcsec from the image origin.
    centre = ((npix - 1) / 2.0) * float(pixel_scale)

    n = len(galaxy_images)
    settings = (
        float(pixel_scale), float(sigma_arcsec), int(stamp_size), int(num_epochs),
        float(noise_variance), centre, float(flux_guess), float(size_guess),
    )
    logger.info(
        "AnaCal model fit: %d stamps (sigma_arcsec=%.3g, window=%d, epochs=%d)",
        n, sigma_arcsec, stamp_size, num_epochs,
    )

    # Serial, deliberately. Measured at ~12.6 ms/object, and neither route out
    # of that works: a spawn process pool deadlocks initialising the pybind11
    # extension in the child, and a thread pool returns identical results at
    # 0.93x -- the C++ fit does not in fact release the GIL in this build, so
    # threads only add overhead. Left as a plain loop rather than as machinery
    # that looks like it parallelises and does not.
    _fit_pool_init(settings)
    results = [_fit_one(galaxy_images[i], psf_images[i]) for i in range(n)]

    e = np.full((n, 2), np.nan)
    dedg = np.full((n, 2, 2), np.nan)
    flags = np.ones(n, dtype=bool)
    for i, item in enumerate(results):
        if item is None:
            continue
        values, response = item
        e[i] = values
        dedg[i] = response
        flags[i] = False

    failed = int(np.sum(flags))
    if failed:
        logger.info("AnaCal model fit: %d/%d objects failed", failed, n)
    return ShapeMeasurement(e=e, dedg=dedg, flags=flags)


def gauss_fit_shapes(measurement: ShapeMeasurement) -> ShapeMeasurement:
    """Identity, for symmetry with :func:`~shearnet.methods.anacal.fpfs_shapes`.

    :func:`measure_gauss_fit` already returns a :class:`ShapeMeasurement`,
    because the quintuple numbers carry the response out of the fit directly --
    there is no separate catalogue to read columns out of.
    """
    return measurement
