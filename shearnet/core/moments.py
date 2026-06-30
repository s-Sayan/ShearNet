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


def get_admoms_ngmix_fit(obs: "ngmix.Observation", reduced: bool = True) -> dict:
    """Measure adaptive-moment ellipticity and size for an observation.

    Fits adaptive moments with ngmix (for e1/e2) and GalSim HSM (for size), on a
    flux-normalized copy of the image. Used to characterize PSF shape in
    :func:`shearnet.core.dataset.sim_func`.

    Args:
        obs: The ngmix observation to fit.
        reduced: If ``True``, convert the distortion (e1, e2) to reduced shear
            (g1, g2) before returning.

    Returns:
        dict: ``{"e1", "e2", "T", "flags"}`` where ``flags`` is non-zero if either
        fit failed or the image had no positive flux.
    """
    jac = obs._jacobian
    scale = jac.get_scale()
    image = obs.image
    norm = np.sum(image[image > 0])
    if norm <= 0:
        return {"e1": np.nan, "e2": np.nan, "T": np.nan, "flags": 1}
    obs_norm = ngmix.Observation(image=image / norm, jacobian=jac)
    am = ngmix.admom.AdmomFitter()
    res = am.go(obs_norm, guess=0.5)
    e1, e2 = res["e1"], res["e2"]
    gal_image = galsim.Image(image / norm, scale=scale)
    admoms = galsim.hsm.FindAdaptiveMom(gal_image)
    sigma = admoms.moments_sigma * scale
    T_galsim = 2 * sigma**2
    flag = 0 if (admoms.moments_status == 0 and res["flags"] == 0) else 1
    if reduced:
        e1, e2 = e1e2_to_g1g2(e1, e2)
    return {"e1": e1, "e2": e2, "T": T_galsim, "flags": flag}
