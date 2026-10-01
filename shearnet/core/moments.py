"""Adaptive-moment measurement primitives for ShearNet.

Lives in ``core`` (rather than the metrics module) so that ``core.dataset`` can
measure PSF moments without importing the heavier metrics module, which would
otherwise create a dependency cycle. :mod:`shearnet.metrics` (and the
``shearnet.utils.metrics`` shim) re-export :func:`get_admoms_ngmix_fit` for
backward compatibility.
"""

import galsim
import ngmix
import numpy as np
from ngmix.shape import e1e2_to_g1g2


def get_admoms_ngmix_fit(obs: "ngmix.Observation", reduced: bool = True, rng=None) -> dict:
    """Measure adaptive-moment ellipticity and size for an observation.

    Fits adaptive moments with ngmix (for e1/e2) and GalSim HSM (for size), on a
    flux-normalized copy of the image. Used to characterize PSF shape in
    :func:`shearnet.core.dataset.sim_func`.

    Args:
        obs: The ngmix observation to fit.
        reduced: If ``True``, convert the distortion (e1, e2) to reduced shear
            (g1, g2) before returning.
        rng: ``np.random.RandomState`` for ngmix's randomized starting guess.
            ``None`` lets ngmix seed one from the OS, so the shape is then only
            reproducible to the fit tolerance (~1e-5).

    Returns:
        dict: ``{"e1", "e2", "T", "T_admom", "flags"}`` where ``flags`` is
        non-zero if either fit failed or the image had no positive flux.

    Two sizes come back and they are not the same quantity. ``T`` is GalSim
    HSM's ``2 sigma^2`` with ``sigma = det(M)^(1/4)`` -- a determinant size,
    which is what every ``T`` this helper has ever returned meant, and what the
    ``psf_T`` training label is. ``T_admom`` is ngmix's adaptive-moment trace
    ``Irr + Icc``, the PSF size SuperBIT's catalogs use. They agree for a round
    PSF and differ by ``1/sqrt(1 - |e|^2)`` otherwise.
    """
    jac = obs._jacobian
    scale = jac.get_scale()
    image = obs.image
    norm = np.sum(image[image > 0])
    if norm <= 0:
        return {"e1": np.nan, "e2": np.nan, "T": np.nan, "T_admom": np.nan, "flags": 1}
    obs_norm = ngmix.Observation(image=image / norm, jacobian=jac)
    am = ngmix.admom.AdmomFitter(rng=rng)
    res = am.go(obs_norm, guess=0.5)
    e1, e2 = res["e1"], res["e2"]
    gal_image = galsim.Image(image / norm, scale=scale)
    admoms = galsim.hsm.FindAdaptiveMom(gal_image)
    sigma = admoms.moments_sigma * scale
    T_galsim = 2 * sigma**2
    flag = 0 if (admoms.moments_status == 0 and res["flags"] == 0) else 1
    if reduced:
        e1, e2 = e1e2_to_g1g2(e1, e2)
    return {"e1": e1, "e2": e2, "T": T_galsim, "T_admom": res.get("T", np.nan), "flags": flag}
