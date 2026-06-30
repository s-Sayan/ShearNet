"""Moment-based metacalibration shear estimation for ShearNet."""

import numpy as np

from ..core.dataset import sim_func


def measure_moments(im):
    """Measure second-order moments for a GalSim image object or a NumPy array.

    Args:
        im: Either a GalSim image object or a 2D NumPy array.

    Returns:
        M_xx, M_yy, M_xy: Second-order moments.
    """
    if hasattr(im, "bounds") and hasattr(im, "array"):
        # Handle GalSim image object
        xmin, xmax, ymin, ymax = im.bounds.xmin, im.bounds.xmax, im.bounds.ymin, im.bounds.ymax
        x, y = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        intensity = im.array
    elif isinstance(im, np.ndarray):
        # Handle NumPy array
        ny, nx = im.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        intensity = im
    else:
        raise TypeError("Input 'im' must be a GalSim image object or a 2D NumPy array.")

    intensity_sum = np.sum(intensity)
    M_x = np.sum(intensity * x) / intensity_sum
    M_y = np.sum(intensity * y) / intensity_sum
    M_xx = np.sum(intensity * (x - M_x) ** 2) / intensity_sum
    M_yy = np.sum(intensity * (y - M_y) ** 2) / intensity_sum
    M_xy = np.sum(intensity * (x - M_x) * (y - M_y)) / intensity_sum
    return M_xx, M_yy, M_xy


def measure_e1e2(g1=None, g2=None, im=None, sigma=1, scale=0.2, npix=53):
    """Calculate ellipticity components e1 and e2.

    Args:
        g1, g2: Shear components. The image-rendering path for these is not
            implemented; pass ``im`` instead.
        im: 2D image array. If provided, moments are computed via
            :func:`measure_moments`.
        sigma: Standard deviation for Gaussian smoothing (used with g1, g2).
        scale: Pixel scale (used with g1, g2).
        npix: Number of pixels along one dimension (used with g1, g2).

    Returns:
        e1, e2: Ellipticity components.
    """
    if im is not None:
        M_xx, M_yy, M_xy = measure_moments(im)
    elif g1 is not None and g2 is not None:
        raise NotImplementedError(
            "measure_e1e2 from (g1, g2) is not implemented; pass an image via im=."
        )
    else:
        raise ValueError("Either (g1 and g2) or im must be provided.")

    Mr = M_xx + M_yy
    Mplus = M_xx - M_yy
    Mcross = 2 * M_xy
    e1 = Mplus / Mr
    e2 = Mcross / Mr
    return e1, e2


def calculate_responsivity(psf_sigma, seed, h=0.01):
    """Estimate the shear responsivity (R1, R2) by finite differences.

    Simulates galaxies sheared by +/- ``h`` in each component and measures how the
    recovered ellipticity changes, giving the multiplicative response used to
    de-bias moment-based shear estimates.

    Args:
        psf_sigma: Gaussian PSF FWHM (arcsec) passed to :func:`sim_func`.
        seed: Random seed for the simulated galaxies.
        h: Finite-difference step in shear.

    Returns:
        tuple: ``(R1, R2)`` responsivities for g1 and g2.
    """
    obj_im_p = sim_func(h, 0, seed=seed, psf_fwhm=psf_sigma)
    obj_im_m = sim_func(-h, 0, seed=seed, psf_fwhm=psf_sigma)
    e1p, _ = measure_e1e2(im=obj_im_p.image)
    e1m, _ = measure_e1e2(im=obj_im_m.image)
    R1 = (e1p - e1m) / (2 * h)
    obj_im_p = sim_func(0, h, seed=seed, psf_fwhm=psf_sigma)
    obj_im_m = sim_func(0, -h, seed=seed, psf_fwhm=psf_sigma)
    _, e2p = measure_e1e2(im=obj_im_p.image)
    _, e2m = measure_e1e2(im=obj_im_m.image)
    R2 = (e2p - e2m) / (2 * h)
    return R1, R2


def obs_g1g2(im, psf_sigma):
    """Calculate the observed g1 and g2 values for a given image and PSF FWHM.

    Args:
        im (numpy.ndarray): The input image.
        psf_sigma (float): The Sigma of the PSF.

    Returns:
        tuple: A tuple containing the observed g1 and g2 values.
    """
    R1, R2 = calculate_responsivity(psf_sigma, 1234)
    e1, e2 = measure_e1e2(im=im)
    obs_g1 = e1 / R1
    obs_g2 = e2 / R2

    return obs_g1, obs_g2


def mcal_preds(images, psf_sigma):
    """Calculate observed g1 and g2 for a list of images and a given PSF FWHM.

    Args:
        images (list): A list of input images.
        psf_sigma (float): The Sigma of the PSF.

    Returns:
        tuple: Two lists, the observed g1 values and the observed g2 values.
    """
    preds = []
    for image in images:
        g1, g2 = obs_g1g2(image, psf_sigma)
        g1g2 = np.array([g1, g2])
        preds.append(g1g2)

    return np.array(preds)
