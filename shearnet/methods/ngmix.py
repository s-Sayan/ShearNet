"""NGmix-based shear estimation and metacalibration utilities for ShearNet."""

import multiprocessing as mp

import ngmix
import numpy as np

from ..logging_utils import get_logger
from ..parallel import cpu_only_children, resolve_nproc

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


def make_struct(res, obs, shear_type):
    """Make the data structure.

    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', 'T', and 'g_cov'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type

    Returns
    -------
    1-element array with fields
    """
    dt = [
        ("flags", "i4"),
        ("shear_type", "U7"),
        ("s2n", "f8"),
        ("g", "f8", 2),
        ("T", "f8"),
        ("flux", "f8"),
        ("Tpsf", "f8"),
        ("g_cov", "f8", (2, 2)),
    ]
    data = np.zeros(1, dtype=dt)
    data["shear_type"] = shear_type
    data["flags"] = res["flags"]

    if res["flags"] == 0:
        data["s2n"] = res["s2n"]
        # for Gaussian moments we are actually measuring e, the ellipticity
        try:
            data["g"] = res["e"]
        except KeyError:
            data["g"] = res["g"]
        data["T"] = res["T"]
        data["flux"] = res["flux"]
        data["g_cov"] = res.get("g_cov", np.nan * np.ones((2, 2)))
    else:
        data["s2n"] = np.nan
        data["g"] = np.nan
        data["T"] = np.nan
        data["flux"] = np.nan
        data["Tpsf"] = np.nan
        data["g_cov"] = np.nan * np.ones((2, 2))

    # Get the psf T by averaging over epochs (and eventually bands)
    tpsf_list = []

    try:
        tpsf_list.append(obs.psf.meta["result"]["T"])
    except Exception:
        # No cached PSF fit result; Tpsf is left as NaN (handled below).
        pass
    valid = np.array(tpsf_list)[np.isfinite(tpsf_list)]
    data["Tpsf"] = np.mean(valid) if valid.size > 0 else np.nan

    return data


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

    # T prior.  This one is flat, but another uninformative you might
    # try is the two-sided error function (TwoSidedErf)

    Tminval = -1.0  # arcsec squared
    Tmaxval = 1000
    T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

    # similar for flux.  Make sure the bounds make sense for
    # your images

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


def process_obs(obs, boot):
    """Run a metacal bootstrapper on one observation; return its struct and obs dict."""
    resdict, obsdict = boot.go(obs)
    dlist = [
        make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
        for stype, sres in resdict.items()
    ]
    return np.hstack(dlist), obsdict


def mp_fit_one(
    obslist,
    prior,
    rng,
    psf_model="gauss",
    gal_model="gauss",
    mcal_pars={"psf": "dilate", "mcal_shear": 0.01},
    weight_fwhm=1.0,
):
    """Run metacalibration on an object (multiprocessing version of ``_fit_one``).

    Returns the unsheared ellipticities of each galaxy, as well as entries for
    each shear step.

    inputs:
    - obslist: Observation list for MEDS object of given ID
    - prior: ngmix mcal priors
    - mcal_pars: mcal running parameters
    - weight_fwhm is just for gaussmom psf and gal models

    TO DO: add a label indicating whether the galaxy passed the selection
    cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).
    """
    # Get image pixel scale
    jacobian = obslist[0]._jacobian
    scale = jacobian.get_scale()

    # GaussMom setup (if using moments-based measurement)
    if gal_model == "gaussmom":
        # FWHM of Gaussian weight function (in arcsec, typically)
        # Rule of thumb: ~1-2x your typical PSF FWHM
        weight_fwhm = 1.2  # You can adjust this
        fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
        runner = ngmix.runners.Runner(fitter=fitter)
    else:
        # Original ML fitting setup
        Tguess = 4 * scale**2
        ntry = 20
        lm_pars = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
        fitter = ngmix.fitting.Fitter(model=gal_model, prior=prior, fit_pars=lm_pars)
        guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tguess, prior=prior)
        runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)

    # PSF fitting setup
    if psf_model == "gaussmom":
        weight_fwhm = 1.2  # Same or different from galaxy weight
        psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
        psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter)
    elif "em" in psf_model:
        em_pars = {"tol": 1.0e-6, "maxiter": 50000}
        psf_ngauss = get_em_ngauss(psf_model)
        psf_fitter = ngmix.em.EMFitter(maxiter=em_pars["maxiter"], tol=em_pars["tol"])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)
        psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=20)
    elif "coellip" in psf_model:
        psf_lm_pars = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
        psf_ngauss = get_coellip_ngauss(psf_model)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
        psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=20)
    elif psf_model == "gauss":
        psf_lm_pars = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
        psf_fitter = ngmix.fitting.Fitter(model="gauss", fit_pars=psf_lm_pars)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
        psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=20)
    else:
        raise ValueError("psf_model must be gaussmom, emN, coellipN, or gauss")

    # types = ['noshear', '1p', '1m', '2p', '2m']
    psf = mcal_pars["psf"]
    mcal_shear = mcal_pars["mcal_shear"]
    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner,
        psf_runner=psf_runner,
        rng=rng,
        psf=psf,
        step=mcal_shear,
        # types=types,
    )

    num_cores = resolve_nproc(n_tasks=len(obslist))
    logger.info(
        f"Starting NGmix ML fitting: num_gal: {len(obslist)} | psf_model: {psf_model} | "
        f"gal_model: {gal_model} | num_cores: {num_cores}"
    )

    return _metacal_map(boot, obslist, num_cores), []


def ngmix_pred(data_list, return_bad_indices=False):
    """Extract ``(g1, g2, sigma, flux)`` predictions from NGmix metacal results.

    Reads the unsheared (``noshear``) entry of each galaxy's result struct and
    converts the moment size ``T`` to ``sigma = sqrt(T/2)``.

    Args:
        data_list: Per-galaxy metacal structs as produced by :func:`mp_fit_one`.
        return_bad_indices: If ``True``, also return the indices with invalid
            (non-positive or NaN) ``T``.

    Returns:
        np.ndarray of shape ``(N, 4)``, or ``(preds, bad_indices)`` when requested.
    """
    g1 = np.array([d[0][3][0] for d in data_list])
    g2 = np.array([d[0][3][1] for d in data_list])
    T = np.array([d[0][4] for d in data_list])

    # Detect bad T values
    bad_mask = (T <= 0) | np.isnan(T)
    bad_indices = np.where(bad_mask)[0]

    # T = 2 * sigma**2
    with np.errstate(invalid="ignore"):  # Suppress warning during calculation
        sigma = np.sqrt(T / 2)

    flux = np.array([d[0][5] for d in data_list])
    preds = np.vstack((g1, g2, sigma, flux)).T

    if return_bad_indices:
        return preds, bad_indices
    return preds


def _metacal_bootstrapper(prior, rng, psf_model, gal_model, mcal_pars, Tguess):
    """The metacal bootstrapper, built one way for both the serial and pool paths."""
    runner, psf_runner = build_runners(
        rng, psf_model=psf_model, gal_model=gal_model, Tguess=Tguess, prior=prior
    )
    return ngmix.metacal.MetacalBootstrapper(
        runner=runner,
        psf_runner=psf_runner,
        rng=rng,
        psf=mcal_pars["psf"],
        step=mcal_pars["mcal_shear"],
    )


#: The metacal shear types, in the order their images are stacked when
#: ``return_images`` is set. Fixed so a caller can index the stack by position.
METACAL_TYPES = ("noshear", "1p", "1m", "2p", "2m")


def _metacal_struct(boot, obs, return_images=False):
    """One object's metacal result as a struct, dropping everything else.

    ``boot.go`` returns ``(resdict, obsdict)``. ``obsdict`` holds the five
    sheared reconvolved observations and ``resdict`` holds their fit results
    with the observations still attached: measured at a 53x53 stamp they are
    3.6 MB and 4.3 MB per object respectively, against 1.4 kB for the struct
    that is the actual answer. Building the struct here and returning only that
    is what keeps a 200k-object population from needing hundreds of gigabytes,
    whether the caller is looping or draining a pool.

    ``return_images`` additionally returns the reconvolved stamps as float32 --
    the five galaxy images and the single dilated PSF they all share, 67 kB per
    object rather than the 3.6 MB of the full observations. That is what a
    non-deconvolving estimator needs in order to have a metacal response at all:
    it measures the same sheared images ngmix does. Missing shear types come
    back as NaN planes so the stack shape never depends on the object.
    """
    resdict, obsdict = boot.go(obs)
    struct = np.hstack(
        [make_struct(res=sres, obs=obsdict[stype], shear_type=stype) for stype, sres in resdict.items()]
    )
    if not return_images:
        return struct
    reference = next(iter(obsdict.values()))
    shape = np.asarray(reference.image).shape
    gal = np.full((len(METACAL_TYPES),) + shape, np.nan, dtype=np.float32)
    for i, stype in enumerate(METACAL_TYPES):
        if stype in obsdict:
            gal[i] = np.asarray(obsdict[stype].image, dtype=np.float32)
    psf = np.asarray(reference.psf.image, dtype=np.float32)
    return struct, gal, psf


#: Per-worker bootstrapper, set by :func:`_metacal_pool_init` under ``spawn``.
_POOL_BOOT = None


def _metacal_pool_init(boot):
    """Give each worker one bootstrapper, instead of pickling it per task.

    It is ~11.5 kB, so shipping it alongside all 200k observations was never
    the memory problem; building it once per worker is simply less work. The
    consequence to know about: the bootstrapper's RNG stream now advances
    across the objects a worker handles, so the metacal column is reproducible
    for a fixed worker count and not across different ones.
    """
    global _POOL_BOOT
    _POOL_BOOT = boot


def _metacal_pool_worker(obs):
    return _metacal_struct(_POOL_BOOT, obs)


def _metacal_pool_worker_images(obs):
    return _metacal_struct(_POOL_BOOT, obs, return_images=True)


def _metacal_map(boot, obslist, workers, chunksize=16, return_images=False):
    """``[struct per object]``, serially or over a spawn pool.

    With ``return_images`` each entry is ``(struct, galaxy_stack, psf)`` instead.
    """
    worker = _metacal_pool_worker_images if return_images else _metacal_pool_worker
    if workers == 1:
        return [_metacal_struct(boot, obs, return_images=return_images) for obs in obslist]
    # imap over the observations, not starmap over [(obs, boot), ...]: an
    # Observation pickles to ~352 kB, so the task list alone would be ~70 GB at
    # 200k objects, queued before any fitting starts. imap streams it, and the
    # worker returns only the 1.4 kB struct -- the obsdict it used to send back
    # is 3.6 MB per object, and every caller in this repo discards it.
    ctx = mp.get_context("spawn")
    with cpu_only_children(), ctx.Pool(
        workers, initializer=_metacal_pool_init, initargs=(boot,)
    ) as pool:
        return list(pool.imap(worker, obslist, chunksize=chunksize))


def mp_fit_one_single(
    obslist,
    prior,
    rng,
    psf_model="gauss",
    gal_model="gauss",
    mcal_pars={"psf": "dilate", "mcal_shear": 0.01},
    nproc=None,
    collect_resdict=False,
    return_images=False,
):
    """Run metacalibration over ``obslist``, returning one struct per object.

    Args:
        nproc: worker processes. ``None`` = the SLURM allocation (1 off-cluster).
            Metacal costs ~156 ms/object against ~7 ms for a plain fit, so a
            200k-object population is ~8.7 hours on one core; this is the knob
            that makes the bias benchmark finish. Results are reproducible for a
            fixed ``(rng seed, nproc)`` -- each worker seeds its own bootstrapper
            deterministically from its index -- but the random *guesses* differ
            if you change the worker count, so record ``nproc`` beside any m you
            intend to reproduce bit-for-bit.
        collect_resdict: also return the per-object ``resdict``. Off by default:
            it is 4.3 MB per object (the sheared observations are still attached
            to the fit results), so accumulating it for a 200k population asks
            for ~817 GB, and the only caller in this repo discards it.

    Returns ``(data_list, resdict_list)``; the second is empty unless
    ``collect_resdict``.

    TO DO: add a label indicating whether the galaxy passed the selection
    cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).
    """
    n = len(obslist)
    if n == 0:
        return [], []
    Tguess = 4 * obslist[0]._jacobian.get_scale() ** 2
    boot = _metacal_bootstrapper(prior, rng, psf_model, gal_model, mcal_pars, Tguess)

    if collect_resdict:
        logger.info("metacal: %d objects, serial (collecting resdict)", n)
        data_list, resdict_list = [], []
        for obs in obslist:
            resdict, obsdict = boot.go(obs)
            data_list.append(
                np.hstack(
                    [
                        make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
                        for stype, sres in resdict.items()
                    ]
                )
            )
            resdict_list.append(resdict)
        return data_list, resdict_list

    workers = resolve_nproc(nproc, n_tasks=n)
    logger.info("metacal: %d objects on %d worker(s)", n, workers)
    return _metacal_map(boot, obslist, workers, return_images=return_images), []


def build_runners(rng, psf_model="gauss", gal_model="gauss", ntry=20, Tguess=None, prior=None):
    """Construct the ngmix PSF/galaxy runners used by every fit in this module.

    Factored out of :func:`mp_fit_one_single` so a plain (non-metacal) shape fit
    and the metacal fit are guaranteed to use the *same* fitter, guesser and
    tolerances. If they drifted apart, a comparison between an ngmix baseline
    calibrated by metacal and the same baseline calibrated by a renderer
    response would be measuring the fitter, not the calibration.
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


def _fit_one_shape(runner, psf_runner, obs, index=0):
    """One plain shape fit: ``(e, ok)``, with a failed fit reported not raised."""
    try:
        psf_runner.go(obs=obs.psf)
        res = runner.go(obs=obs)
    except Exception as exc:
        logger.debug("ngmix shape fit failed on object %d: %s", index, exc)
        return np.array([np.nan, np.nan]), False
    if res.get("flags", 1) != 0:
        return np.array([np.nan, np.nan]), False
    value = res.get("e", res.get("g"))
    if value is None or not np.all(np.isfinite(value)):
        return np.array([np.nan, np.nan]), False
    return np.asarray(value, dtype=float), True


#: Per-worker runners, set by :func:`_fit_pool_init` under ``spawn``.
_POOL_RUNNERS = (None, None)


def _fit_pool_init(runner, psf_runner):
    global _POOL_RUNNERS
    _POOL_RUNNERS = (runner, psf_runner)


def _fit_pool_worker(obs):
    return _fit_one_shape(_POOL_RUNNERS[0], _POOL_RUNNERS[1], obs)


def fit_shapes(obslist, seed=42, psf_model="gauss", gal_model="gauss", nproc=None):
    """Fit each observation once, with no metacal, returning ``(e, flags)``.

    This is the measurement half of ngmix on its own: it is what the shared
    renderer-response protocol differentiates, so that the ngmix baseline can be
    calibrated the same way the network and FPFS are and the three numbers mean
    the same thing.

    ``nproc`` follows the same rule as :func:`mp_fit_one_single`: ``None`` means
    the SLURM allocation, 1 off-cluster. At ~7 ms/object a plain fit is 20x
    cheaper than metacal, but the bias task runs it six times over the
    population (two shear populations plus four response re-renders), which is
    still 2.3 hours of one core at 200k objects.

    Returns ``(N, 2)`` ellipticities (NaN where the fit failed) and an ``(N,)``
    failure mask.
    """
    n = len(obslist)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    rng = np.random.RandomState(seed)
    Tguess = 4 * obslist[0]._jacobian.get_scale() ** 2
    runner, psf_runner = build_runners(rng, psf_model=psf_model, gal_model=gal_model, Tguess=Tguess)
    workers = resolve_nproc(nproc, n_tasks=n)

    if workers == 1:
        results = [_fit_one_shape(runner, psf_runner, obs, i) for i, obs in enumerate(obslist)]
    else:
        ctx = mp.get_context("spawn")
        with cpu_only_children(), ctx.Pool(
            workers, initializer=_fit_pool_init, initargs=(runner, psf_runner)
        ) as pool:
            results = list(pool.imap(_fit_pool_worker, obslist, chunksize=64))

    e = np.full((n, 2), np.nan)
    flags = np.ones(n, dtype=bool)
    for i, (value, ok) in enumerate(results):
        if ok:
            e[i] = value
            flags[i] = False
    return e, flags


def get_memory_usage(obj):
    """Print the memory usage (MB) of each attribute of an object, and the total."""
    from pympler import asizeof

    memory_usage = {
        attr: asizeof.asizeof(getattr(obj, attr)) / (1024 * 1024) for attr in obj.__dict__
    }

    for attr, size in memory_usage.items():
        logger.info(f"{attr}: {size:.6f} MB")

    total_memory = sum(memory_usage.values())
    logger.info(f"\nTotal memory used by instance: {total_memory:.6f} MB")


def response_calculation(data_list, mcal_shear):
    """Compute per-galaxy metacalibration response and PSF-leakage terms.

    From the sheared (``1p/1m/2p/2m``) entries of each galaxy's metacal struct,
    finite-differences the shear response matrix elements (R11, R22, R12, R21) and
    the additive/PSF terms, using a step of ``mcal_shear``.

    Returns:
        tuple of eight lists: ``(r11, r22, r12, r21, c1, c2, c1_psf, c2_psf)``.
    """
    r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i in range(len(data_list)):
        g_noshear = data_list[i][0][3]
        g_1p = data_list[i][1][3]
        g_1m = data_list[i][2][3]
        g_2p = data_list[i][3][3]
        g_2m = data_list[i][4][3]
        g_1p_psf = data_list[i][5][3]
        g_1m_psf = data_list[i][6][3]
        g_2p_psf = data_list[i][7][3]
        g_2m_psf = data_list[i][8][3]
        r11 = (g_1p[0] - g_1m[0]) / (2.0 * mcal_shear)
        r22 = (g_2p[1] - g_2m[1]) / (2.0 * mcal_shear)
        r12 = (g_2p[0] - g_2m[0]) / (2.0 * mcal_shear)
        r21 = (g_1p[1] - g_1m[1]) / (2.0 * mcal_shear)
        c1 = (g_1p[0] + g_1m[0]) / 2 - g_noshear[0]
        c2 = (g_2p[1] + g_2m[1]) / 2 - g_noshear[1]
        c1_psf = (g_1p_psf[0] + g_1m_psf[0]) / 2 - g_noshear[0]
        c2_psf = (g_2p_psf[1] + g_2m_psf[1]) / 2 - g_noshear[1]
        r11_list.append(r11)
        r22_list.append(r22)
        r12_list.append(r12)
        r21_list.append(r21)
        c1_list.append(c1)
        c2_list.append(c2)
        c1_psf_list.append(c1_psf)
        c2_psf_list.append(c2_psf)

    return r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list
