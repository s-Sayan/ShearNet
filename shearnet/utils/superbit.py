"""Utility functions copied verbatim from the superbit-lensing repository.

These helpers were previously imported as ``superbit_lensing.utils`` by the
research scripts and notebooks under ``research/shear_bias``. To remove the
external ``superbit-lensing`` dependency, the specific functions ShearNet
relies on are copied here **unchanged**.

Source: https://github.com/superbit-collaboration/superbit-lensing
        (``superbit_lensing/utils.py``)

Nothing in this module is original to ShearNet; it is a vendored copy so that
the ``research`` code can run without installing ``superbit-lensing``. Any
change to the upstream implementation should be mirrored from that repository.
"""

import numpy as np
import galsim
import ngmix
from ngmix.gaussmom import GaussMom
from ngmix.shape import e1e2_to_g1g2


def get_admoms(image: np.ndarray, scale: float, mode: str = "ngmix", seed: int = 124, reduced: bool = True) -> dict:
    """
    Measure adaptive moments (ADMOM) of an image using either ngmix or GalSim.

    Parameters
    ----------
    image : ndarray
        Input 2D image array.
    scale : float
        Pixel scale in arcsec/pixel.
    mode : {"ngmix", "galsim"}, optional
        Which backend to use for measuring moments.
    reduced : bool, optional
        If True, return reduced shear (g1, g2) instead of ellipticity (e1, e2).

    Returns
    -------
    result : dict
        Dictionary containing:
            - "e1" / "g1": ellipticity or reduced shear component 1
            - "e2" / "g2": ellipticity or reduced shear component 2
            - "T": size measure (2 * sigma^2)
            - "flag": int (0 = success, nonzero = failure)
    """
    rng = np.random.RandomState(seed)
    
    # image center
    cx, cy = (image.shape[1]-1) / 2, (image.shape[0]-1) / 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cy, col=cx)

    # --- Normalize positive flux ---
    norm = np.sum(image[image > 0])
    if norm <= 0:
        return {"e1": np.nan, "e2": np.nan, "T": np.nan, "flags": 1}

    # --- Moment measurement ---
    if mode == "ngmix":
        obs_im = ngmix.Observation(image=image / norm, jacobian=jac)
        gm = GaussMom(1.2).go(obs_im)
        am = ngmix.admom.AdmomFitter(rng=rng)
        res = am.go(obs_im, guess=gm["T"])
        e1, e2, T, flag = res["e1"], res["e2"], res["T"], res["flags"]

    elif mode == "galsim":
        gal_image = galsim.Image(image / norm, scale=scale)
        admoms = galsim.hsm.FindAdaptiveMom(gal_image)
        e1, e2 = admoms.observed_e1, admoms.observed_e2
        sigma = admoms.moments_sigma * scale
        T = 2 * sigma**2
        flag = 0 if admoms.moments_status == 0 else 1

    else:
        raise ValueError(f"Unknown mode '{mode}', must be 'ngmix' or 'galsim'")

    # --- Convert if reduced shear requested ---
    if reduced:
        e1, e2 = e1e2_to_g1g2(e1, e2)

    return {"e1": e1, "e2": e2, "T": T, "flags": flag}

def get_admoms_ngmix_fit(obs: "ngmix.Observation", seed: int = 124, reduced: bool = True) -> dict:
    """
    Measure adaptive moments (ADMOM) of an image using ngmix and GalSim.

    Parameters
    ----------
    obs : ngmix.Observation
        The observation containing the image and jacobian.
    reduced : bool, optional
        If True, return reduced shear (g1, g2) instead of ellipticity (e1, e2).

    Returns
    -------
    result : dict
        Dictionary containing:
            - "e1" / "g1": ellipticity or reduced shear component 1
            - "e2" / "g2": ellipticity or reduced shear component 2
            - "T": size measure (2 * sigma^2)
            - "flag": int (0 = success, 1 = failure)
    """
    rng = np.random.RandomState(seed)
    jac = obs._jacobian
    scale = jac.get_scale()
    image = obs.image

    # --- Normalize positive flux ---
    norm = np.sum(image[image > 0])
    if norm <= 0:
        return {"e1": np.nan, "e2": np.nan, "T": np.nan, "flags": 1}

    # --- Measure moments with ngmix ---
    obs_norm = ngmix.Observation(image=image / norm, jacobian=jac)
    gm = GaussMom(1.2).go(obs_norm)
    am = ngmix.admom.AdmomFitter(rng=rng)
    res = am.go(obs_norm, guess=gm["T"])
    e1, e2, T_ngmix = res["e1"], res["e2"], res["T"]

    # --- Measure size using GalSim ---
    # gal_image = galsim.Image(image / norm, wcs=jac.get_galsim_wcs())
    # admoms = galsim.hsm.FindAdaptiveMom(gal_image)
    # sigma = admoms.moments_sigma * scale
    # T_galsim = 2 * sigma**2

    # # --- Set flag based on both results ---
    # flag = 0 if (admoms.moments_status == 0 and res["flags"] == 0) else 1

    # --- Convert to reduced shear if requested ---
    if reduced:
        e1, e2 = e1e2_to_g1g2(e1, e2)

    return {"e1": e1, "e2": e2, "T": T_ngmix, "flags": res["flags"]}

def g_from_gal_jac(gal):
    """
    From a transformed GalSim object, return:
      g1, g2  : reduced shear components
      mu      : magnification (1/detJ)
      kappa   : convergence inferred from detJ and |g|
    """
    J = np.asarray(gal.jac, dtype=float)  # [[dudx, dudy],[dvdx, dvdy]]

    # Built-in decomposition to get the Shear object:
    wcs = galsim.JacobianWCS(J[0,0], J[0,1], J[1,0], J[1,1])
    scale, shear, theta, flip = wcs.getDecomposition()
    g1, g2 = shear.g1, shear.g2

    return g1, g2, scale**2, theta.deg, flip

def get_galsim_tanwcs(image_xsize=9600, image_ysize=6400, center_ra=13.3 * galsim.hours, center_dec=33.1 * galsim.degrees, pixel_scale=0.141):
    fiducial_full_image = galsim.ImageF(image_xsize, image_ysize)
    theta = 0.0 * galsim.degrees
    dudx = np.cos(theta) * pixel_scale
    dudy = -np.sin(theta) * pixel_scale
    dvdx = np.sin(theta) * pixel_scale
    dvdy = np.cos(theta) * pixel_scale

    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, origin=fiducial_full_image.true_center)
    sky_center = galsim.CelestialCoord(ra=center_ra, dec=center_dec)
    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)

    return wcs
