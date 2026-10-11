"""NGmix-based shear estimation and metacalibration utilities for ShearNet."""

import ngmix
import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


# Function to remove NaN values and count them
def clean_and_report_nans(data_list, name):
    """Return ``data_list`` as an array with NaNs removed, printing how many were dropped."""
    clean_list = np.array(data_list)
    nan_count = np.isnan(clean_list).sum()

    if nan_count > 0:
        logger.info(f"Removed {nan_count} NaN values from {name}.")

    return clean_list[~np.isnan(clean_list)]  # Return array without NaNs


def fourier_transform(psf):
    """Return the centered (fftshifted) 2-D Fourier transform of an image."""
    # Compute the Fourier Transform
    ft_psf = np.fft.fft2(psf)
    ft_psf_shifted = np.fft.fftshift(ft_psf)  # Shift zero frequency to center
    return ft_psf_shifted


def inverse_fourier_transform(ft_psf_shifted):
    """Invert :func:`fourier_transform`, returning the real-space magnitude image."""
    # Shift back and compute Inverse Fourier Transform
    ft_psf = np.fft.ifftshift(ft_psf_shifted)
    psf_reconstructed = np.fft.ifft2(ft_psf)
    return np.abs(psf_reconstructed)


def fft_ifft(psf):
    """Round-trip an image through FFT and inverse FFT (a numerical sanity check)."""
    # Compute the Fourier Transform
    ft_psf = np.fft.fft2(psf)
    ft_psf_shifted = np.fft.fftshift(ft_psf)  # Shift zero frequency to center
    # Shift back and compute Inverse Fourier Transform
    ft_psf = np.fft.ifftshift(ft_psf_shifted)
    psf_reconstructed = np.fft.ifft2(ft_psf)
    return np.abs(psf_reconstructed)


def convolve2d(image, psf, mode="same", boundary="wrap"):
    """Convolve an image with a PSF using 2D FFT-based convolution.

    Always returns an image of the same dimensions as the input.

    Parameters:
    image (np.ndarray): The input image.
    psf (np.ndarray): The PSF.
    mode (str): Not used in FFT implementation, kept for API consistency.
    boundary (str): Not used in FFT implementation, kept for API consistency.

    Returns:
    np.ndarray: The convolved image with same dimensions as input.
    """
    # Ensure PSF is centered in an array of the same size as the image
    psf_padded = np.zeros(image.shape, dtype=psf.dtype)
    psf_center = np.array(psf.shape) // 2
    image_center = np.array(image.shape) // 2

    # Calculate the corner positions for placing the PSF
    top = image_center[0] - psf_center[0]
    left = image_center[1] - psf_center[1]

    # Handle PSFs larger than the image
    psf_top = max(0, -top)
    psf_left = max(0, -left)
    psf_bottom = min(psf.shape[0], image.shape[0] - top)
    psf_right = min(psf.shape[1], image.shape[1] - left)

    # Handle image boundaries
    img_top = max(0, top)
    img_left = max(0, left)
    img_bottom = min(image.shape[0], top + psf.shape[0])
    img_right = min(image.shape[1], left + psf.shape[1])

    # Place the PSF in the padded array
    psf_padded[img_top:img_bottom, img_left:img_right] = psf[psf_top:psf_bottom, psf_left:psf_right]

    # Perform FFT convolution
    ft_image = np.fft.fft2(image)
    ft_psf = np.fft.fft2(psf_padded)
    ft_result = ft_image * ft_psf
    result = np.abs(np.fft.ifft2(ft_result))

    return np.fft.fftshift(result)


def sample_half_gaussian(size=1, sigma=0.5, seed=None):
    """Generate samples from a half-Gaussian distribution with given sigma.

    Ensures that all values are strictly greater than zero.

    Parameters:
    size (int): Number of samples to generate.
    sigma (float): Standard deviation of the full Gaussian.
    seed (int, optional): Seed for reproducibility.

    Returns:
    np.ndarray: Array of sampled values.
    """
    rng = np.random.default_rng(seed)  # Use numpy's random generator with seed
    samples = np.abs(rng.normal(loc=0, scale=sigma, size=size))
    samples = samples[samples > 0.14]  # Remove zeros
    while len(samples) < size:
        extra_samples = np.abs(rng.normal(loc=0, scale=sigma, size=size - len(samples)))
        extra_samples = extra_samples[extra_samples > 0.14]
        samples = np.concatenate((samples, extra_samples))
    return samples


def g1_g2_sigma_sample(num_samples=10000, g_std=0.26, sigma_std=0.5, seed=None):
    """Generate samples for g1, g2, and sigma.

    g1 and g2 are sampled from a Gaussian distribution with specified standard
    deviation, and sigma from a half-Gaussian distribution.

    Parameters:
    num_samples (int): Number of samples to generate.
    g_std (float): Standard deviation for the Gaussian distribution of g1 and g2. Default is 0.26.
    sigma_std (float): Standard deviation for the half-Gaussian distribution of
        sigma. Default is 0.5.
    seed (int, optional): Seed for reproducibility.

    Returns:
    tuple: Arrays of sampled g1, g2, and sigma values.
    """
    rng = np.random.default_rng(seed)  # Use numpy's random generator with seed
    sigma = sample_half_gaussian(size=num_samples, sigma=sigma_std, seed=seed)

    # Generate g1 and g2 from Gaussian distribution with specified std
    g1_selected = np.zeros(num_samples)
    g2_selected = np.zeros(num_samples)

    for i in range(num_samples):
        while True:
            # Sample from Gaussian with specified std (mean=0)
            g1, g2 = rng.normal(0, g_std, 2)
            if np.abs(g1 + 1j * g2) < 1.0:  # Check the constraint
                g1_selected[i] = g1
                g2_selected[i] = g2
                break  # Accept only valid values

    return g1_selected, g2_selected, sigma


def _get_priors(seed):

    # This bit is needed for ngmix v2.x.x
    # won't work for v1.x.x
    rng = np.random.RandomState(seed)

    # prior on ellipticity.  The details don't matter, as long
    # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014

    g_sigma = 0.3
    g_prior = ngmix.priors.GPriorBA(g_sigma, rng=rng)

    # 2-d gaussian prior on the center
    # row and column center (relative to the center of the jacobian, which would be zero)
    # and the sigma of the gaussians

    # units same as jacobian, probably arcsec
    row, col = 0.0, 0.0
    row_sigma, col_sigma = 0.2, 0.2  # a bit smaller than pix size of SuperBIT
    cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng=rng)

    # Flat prior on T.

    Tminval = -1.0  # arcsec squared
    Tmaxval = 1000
    T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

    # Flat flux prior; bounds must cover the image flux range.

    Fminval = -1.0e1
    Fmaxval = 1.0e5
    F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

    # now make a joint prior.  This one takes priors
    # for each parameter separately
    priors = ngmix.joint_prior.PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)

    return priors


def get_em_ngauss(name):
    """Parse the Gaussian count from an EM model name, e.g. ``'em3'`` -> ``3``."""
    ngauss = int(name[2:])
    return ngauss


def get_coellip_ngauss(name):
    """Parse the Gaussian count from a coelliptical model name, e.g. ``'coellip3'`` -> ``3``."""
    ngauss = int(name[7:])
    return ngauss


#: FWHM (arcsec) of the GaussMom weight whose T seeds every galaxy fit, as in
#: superbit-lensing's ``metacalibration/ngmix_fit.py``.
TGUESS_GAUSSMOM_FWHM = 1.2


def set_tguess(runner, obs):
    """Centre the galaxy T guess on this object, the way superbit-lensing does.

    superbit-lensing measures ``GaussMom(1.2).go(obs)['T']`` on each object
    before its fit and hands it to ``TPSFFluxAndPriorGuesser``; the metacal
    products are all guessed from the original stamp's T. A stamp GaussMom
    cannot measure (flux <= 0 under the weight) gives a NaN guess there too, and
    the fit is then flagged rather than raised, as it is in superbit-lensing.
    """
    runner.guesser.T = ngmix.gaussmom.GaussMom(fwhm=TGUESS_GAUSSMOM_FWHM).go(obs)["T"]


def build_runners(rng, psf_model="gauss", gal_model="gauss", ntry=20, Tguess=None, prior=None):
    """Construct the ngmix PSF/galaxy runners used by every fit in this module.

    ``Tguess`` is only a placeholder: every fit sets it per object with
    :func:`set_tguess` first.
    """
    lm_pars = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
    psf_lm_pars = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
    if prior is None:
        prior = _get_priors(rng.randint(0, 2**31 - 1) if hasattr(rng, "randint") else 0)

    fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior, fit_pars=lm_pars)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tguess, prior=prior)

    if "em" in psf_model:
        psf_fitter = ngmix.em.EMFitter(maxiter=50000, tol=1.0e-6)
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=get_em_ngauss(psf_model))
    elif "coellip" in psf_model:
        ngauss = get_coellip_ngauss(psf_model)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=ngauss, fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=ngauss)
    elif psf_model == "gauss":
        psf_fitter = ngmix.fitting.Fitter(model="gauss", fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    else:
        raise ValueError("psf_model must be one of emn, coellipn, or gauss")

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=ntry)
    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)
    return runner, psf_runner
