"""Galaxy/PSF postage-stamp simulation and dataset generation for ShearNet."""

import functools
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import List, Optional

import galsim
import galsim.des
import ngmix
import numpy as np
from astropy.io import fits
from tqdm import tqdm

from .moments import get_admoms_ngmix_fit
from .wcs import create_wcs_from_params

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetResult:
    """Structured result of :func:`generate_dataset`.

    A stable, self-describing alternative to the tuple return whose arity
    otherwise changes with ``return_obs``. ``obs`` is populated only when the
    dataset was generated with ``return_obs=True``.
    """

    images: np.ndarray
    labels: np.ndarray
    obs: Optional[List["ngmix.Observation"]] = None


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SHEARNET_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

WCS_PARAMS = {
    "image_xsize": 9600,
    "image_ysize": 6422,
    "pixel_scale": 0.1408,  # arcsec/pixel
    "center_ra": 13.3,
    "center_dec": 33.1,
    "theta": 0.0,  # optional
}

MARGIN = 200  # Margins that I wanna use for PSF Rendering

# Directory holding the empirical PSFEx PSF files used for the ``superbit``
# experiment. Defaults to the SuperBIT PSFs bundled with the repository, but can
# be overridden with the ``SHEARNET_PSF_DIR`` environment variable to point at a
# different PSF library.
PSF_DATA_DIR = os.environ.get(
    "SHEARNET_PSF_DIR",
    os.path.join(SHEARNET_ROOT, "psf_data", "emp_psfs_best", "psfex-output"),
)

_cosmos_cat_cache = None


def _load_cosmos_cat(seed=42, cat_path=None):
    """Lazy-load the COSMOS catalog, with a random fallback for CI."""
    global _cosmos_cat_cache
    if _cosmos_cat_cache is not None:
        return _cosmos_cat_cache

    if cat_path is not None and os.path.exists(cat_path):
        with fits.open(cat_path) as hdul:
            _cosmos_cat_cache = hdul[1].data
        return _cosmos_cat_cache

    logger.warning(
        "WARNING: cosmos_catalog_train.fits not found. "
        "Using synthetic random catalog for g1/g2/hlr/flux."
    )
    rng = np.random.RandomState(seed)
    n = 5000
    g_1 = rng.normal(0.0, 0.26, n)
    g_2 = rng.normal(0.0, 0.26, n)

    # Resample the rare tail where the shear magnitude reaches the unphysical
    # |g| >= 1 that GalSim rejects (~0.1% of a sigma=0.26 2D Gaussian). Without
    # this, training on the synthetic fallback crashes once the sample count is
    # large enough to hit one (e.g. the default 10000 samples).
    _max_mag = 0.9
    bad = np.hypot(g_1, g_2) >= _max_mag
    while np.any(bad):
        g_1[bad] = rng.normal(0.0, 0.26, int(bad.sum()))
        g_2[bad] = rng.normal(0.0, 0.26, int(bad.sum()))
        bad = np.hypot(g_1, g_2) >= _max_mag

    class _SyntheticCat:
        def __init__(self):
            self.G1 = g_1
            self.G2 = g_2
            self.HLR = np.full(n, 0.5)
            self.FLUX = np.full(n, 12258.97)

        def __getitem__(self, key):
            return getattr(self, key)

    _cosmos_cat_cache = _SyntheticCat()
    return _cosmos_cat_cache


def _worker_init():
    """Pool-worker initializer: keep generation workers off the GPU.

    Generation is pure galsim/ngmix/numpy and never runs a JAX op, but importing
    the package transitively imports jax. Hiding the GPU here guarantees a worker
    can never grab a CUDA context -- belt-and-suspenders on top of the ``spawn``
    start method (which already avoids the fork-with-live-CUDA deadlock).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _generate_one(task, cfg):
    """Simulate one stamp for :func:`generate_dataset` (serial call or pool worker).

    ``task`` is ``(i, g1, g2, hlr, flux)`` and ``cfg`` bundles the shared
    simulation settings. Returns ``(image, label, obs_or_None)`` -- ``obs`` is
    returned only when ``cfg['return_obs']`` is set, so training runs never pickle
    the heavy ngmix ``Observation`` objects back from the workers.

    For the empirical (``superbit``) PSF the sampling deviate is seeded from
    ``(seed, i)``, so the dataset is identical for any ``nproc`` (the old shared
    deviate depended on loop order and could not be parallelized). Analytic
    (``ideal``) PSFs do not use the deviate.
    """
    i, g1, g2, hlr, flux = task
    ud = None
    if cfg["exp"] == "superbit":
        ud = galsim.UniformDeviate(int(cfg["seed"]) * 1_000_003 + int(i))

    obj_obs = sim_func(
        g1,
        g2,
        hlr=hlr,
        flux=flux,
        psf_fwhm=cfg["psf_fwhm"],
        nse_sd=cfg["nse_sd"],
        type=cfg["type"],
        npix=cfg["npix"],
        scale=cfg["scale"],
        seed=i,
        exp=cfg["exp"],
        apply_psf_shear=cfg["apply_psf_shear"],
        psf_shear_range=cfg["psf_shear_range"],
        ud=ud,
        psf_files=cfg["psf_files"],
        base_shear_g1=cfg["base_shear_g1"],
        base_shear_g2=cfg["base_shear_g2"],
        compute_metacal=cfg["compute_metacal"],
        compute_psf_admom=cfg["compute_psf_admom"],
    )

    if cfg["return_psf"]:
        image = np.stack([obj_obs.image, obj_obs.psf.image], axis=-1)
    else:
        image = obj_obs.image

    available = {
        "g1": g1,
        "g2": g2,
        "hlr": hlr,
        "flux": flux,
        "psf_e1": obj_obs.psf.meta["e1"],
        "psf_e2": obj_obs.psf.meta["e2"],
        "psf_T": obj_obs.psf.meta["T"],
    }
    label = np.array([available[k] for k in cfg["output_keys"]], dtype=np.float32)
    return image, label, (obj_obs if cfg["return_obs"] else None)


def generate_dataset(
    samples,
    psf_fwhm,
    npix=53,
    scale=0.141,
    type="exp",
    exp="ideal",
    nse_sd=1e-5,
    seed=42,
    return_clean=False,
    return_psf=False,
    return_obs=False,
    apply_psf_shear=False,
    psf_shear_range=0.05,
    base_shear_g1=0.0,
    base_shear_g2=0.0,
    psf_file_or_dir=PSF_DATA_DIR,
    output_keys=("g1", "g2"),
    hlr_type="constant",
    flux_type="constant",
    cosmos_cat_fname=None,
    as_result=False,
    compute_metacal=False,
    nproc=1,
):
    """Simulate a dataset of galaxy postage stamps with known shear labels.

    Each sample is a GalSim galaxy (exponential or Gaussian) sheared by values
    drawn from a COSMOS catalog, convolved with either an analytic Gaussian PSF
    (``exp='ideal'``) or an empirical PSFEx SuperBIT PSF (``exp='superbit'``),
    and corrupted with Gaussian noise. See :func:`sim_func` for the per-object
    simulation details.

    Args:
        samples: Number of postage stamps to generate.
        psf_fwhm: FWHM (arcsec) of the analytic Gaussian PSF (``exp='ideal'``).
        npix: Stamp size in pixels (square).
        scale: Pixel scale in arcsec/pixel.
        type: Galaxy light profile, ``'exp'`` or ``'gauss'``.
        exp: Experiment / PSF mode, ``'ideal'`` or ``'superbit'``.
        nse_sd: Standard deviation of the additive Gaussian noise.
        seed: Base random seed for reproducibility.
        return_clean: Deprecated/unsupported; raises ``NotImplementedError`` if
            set (the noise-free clean image was removed from ``sim_func``).
        return_psf: Also stack the PSF image as an extra channel.
        return_obs: Additionally return the list of ngmix ``Observation`` objects.
        apply_psf_shear: Apply a random shear to the PSF (``exp='ideal'`` only).
        psf_shear_range: Half-width of the uniform PSF-shear distribution.
        base_shear_g1, base_shear_g2: Constant shear applied to every galaxy.
        psf_file_or_dir: PSF file or directory of ``.psf`` files for the
            ``superbit`` mode (defaults to :data:`PSF_DATA_DIR`).
        output_keys: Label fields to return per sample; subset of
            ``{"g1", "g2", "hlr", "flux", "psf_e1", "psf_e2", "psf_T"}``.
        hlr_type: ``'constant'`` (0.5) or ``'catalog'`` half-light radius.
        flux_type: ``'constant'`` or ``'catalog'`` flux.
        cosmos_cat_fname: Path to the COSMOS catalog FITS file; a synthetic
            random catalog is used as a fallback (e.g. in CI) when absent.
        as_result: Return a :class:`DatasetResult` (stable shape regardless of
            ``return_obs``) instead of a tuple.
        compute_metacal: Compute and store the four metacal (+/- e1/e2)
            reconvolved images per object. Off by default -- plain training does
            not use them (they only live in each ``Observation``'s metadata).
        nproc: Number of worker processes for generation. ``None`` (default) is
            auto -- ``SLURM_CPUS_PER_TASK`` on a cluster, else ``1`` (serial).
            An explicit ``>1`` uses a ``spawn``-based pool (safe alongside JAX --
            never ``fork``); ``1`` forces serial. The output is identical for any
            ``nproc`` (PSF sampling is seeded per object).

    Returns:
        By default ``(images, labels)`` as numpy arrays, or ``(images, labels,
        obs)`` when ``return_obs=True``. With ``as_result=True`` a
        :class:`DatasetResult` is returned instead. ``images`` is
        ``(samples, npix, npix)`` for a single channel, or has a trailing channel
        axis when ``return_psf``/``return_clean`` are set. ``labels`` has shape
        ``(samples, len(output_keys))``.
    """
    if return_clean:
        raise NotImplementedError(
            "return_clean is no longer supported: the noise-free clean image was "
            "removed from sim_func (it was unused and could trigger oversized FFTs "
            "for compact galaxies)."
        )

    cosmos_cat = _load_cosmos_cat(seed=seed, cat_path=cosmos_cat_fname)
    g1_list = cosmos_cat["G1"]
    g2_list = cosmos_cat["G2"]
    hlr_list = cosmos_cat["HLR"]
    flux_list = cosmos_cat["FLUX"]

    if exp == "superbit":
        if os.path.isfile(psf_file_or_dir):
            psf_files = [psf_file_or_dir]
        elif os.path.isdir(psf_file_or_dir):
            psf_files = search_psf_files(path=psf_file_or_dir)
            if len(psf_files) == 0:
                raise FileNotFoundError(f"No PSF files found in {psf_file_or_dir}")
        else:
            raise FileNotFoundError(f"{psf_file_or_dir} is neither a file nor a directory")
    else:
        psf_files = None

    _valid = {"g1", "g2", "hlr", "flux", "psf_e1", "psf_e2", "psf_T"}
    _requested = set(output_keys)
    if not _requested.issubset(_valid):
        raise ValueError(f"Invalid output_keys: {_requested - _valid}. Must be subset of {_valid}.")
    if hlr_type not in ("catalog", "constant"):
        raise ValueError("hlr can only be 'constant' or 'catalog'")
    if flux_type not in ("catalog", "constant"):
        raise ValueError("flux can only be 'constant' or 'catalog'")

    # generate_dataset indexes the catalog by position, so samples must fit.
    if samples > len(g1_list):
        raise ValueError(
            f"Requested {samples} samples but the catalog has only {len(g1_list)} rows. "
            "Lower `samples` or use a larger catalog."
        )

    # The PSF adaptive-moments fit only feeds the psf_* labels; skip it otherwise.
    do_admom = bool(_requested & {"psf_e1", "psf_e2", "psf_T"})

    cfg = {
        "psf_fwhm": psf_fwhm, "nse_sd": nse_sd, "type": type, "npix": npix, "scale": scale,
        "seed": seed, "exp": exp, "apply_psf_shear": apply_psf_shear,
        "psf_shear_range": psf_shear_range, "psf_files": psf_files,
        "base_shear_g1": base_shear_g1, "base_shear_g2": base_shear_g2,
        "return_psf": return_psf, "return_obs": return_obs, "output_keys": tuple(output_keys),
        "compute_metacal": compute_metacal, "compute_psf_admom": do_admom,
    }

    def _hlr(i):
        return float(hlr_list[i]) if hlr_type == "catalog" else 0.5

    def _flux(i):
        return float(flux_list[i]) if flux_type == "catalog" else 12258.97

    tasks = [(i, float(g1_list[i]), float(g2_list[i]), _hlr(i), _flux(i)) for i in range(samples)]

    if nproc is None:
        # Auto: respect the SLURM allocation, NOT the node's total cores
        # (os.cpu_count would oversubscribe a shared node).
        nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    # Never spawn more workers than there are stamps (matters for tiny runs).
    nproc = max(1, min(int(nproc), len(tasks)))

    _disable = not sys.stderr.isatty()
    worker = functools.partial(_generate_one, cfg=cfg)
    if nproc == 1:
        results = [worker(t) for t in tqdm(tasks, disable=_disable, mininterval=10)]
    else:
        # 'spawn', never 'fork': the parent already holds a JAX/CUDA context
        # (get_device()), and forking it deadlocks. Spawned workers start clean
        # and run pure galsim/ngmix.
        ctx = mp.get_context("spawn")
        chunk = max(1, min(64, (samples // (nproc * 8)) or 1))
        with ctx.Pool(processes=nproc, initializer=_worker_init) as pool:
            results = list(
                tqdm(pool.imap(worker, tasks, chunksize=chunk),
                     total=samples, disable=_disable, mininterval=10)
            )

    images_arr = np.array([r[0] for r in results])
    labels_arr = np.array([r[1] for r in results])
    obs = [r[2] for r in results] if return_obs else None

    if as_result:
        # Opt-in structured return with a stable shape regardless of return_obs.
        return DatasetResult(images_arr, labels_arr, obs)

    if return_obs:
        return images_arr, labels_arr, obs

    return images_arr, labels_arr


def split_combined_images(combined_images, has_psf=False, has_clean=False):
    """
    Split concatenated images back into separate arrays.

    Args:
        combined_images: np.ndarray of shape (samples, height, width, 2 or 3)
        has_psf: bool, whether PSF images are included
        has_clean: bool, whether clean images are included

    Returns:
        Tuple of arrays depending on combination:
        - If has_psf=True, has_clean=True: (galaxy, psf, clean)
        - If has_psf=True, has_clean=False: (galaxy, psf)
        - If has_psf=False, has_clean=True: (galaxy, clean)
    """
    if combined_images.shape[-1] == 2:
        if has_psf and not has_clean:
            # Galaxy + PSF
            galaxy_images = combined_images[..., 0]
            psf_images = combined_images[..., 1]
            return galaxy_images, psf_images
        elif has_clean and not has_psf:
            # Galaxy + Clean
            galaxy_images = combined_images[..., 0]
            clean_images = combined_images[..., 1]
            return galaxy_images, clean_images
        else:
            raise ValueError(
                "Invalid combination: 2 channels requires either PSF or clean images, not both"
            )

    elif combined_images.shape[-1] == 3:
        if has_psf and has_clean:
            # Galaxy + PSF + Clean
            galaxy_images = combined_images[..., 0]
            psf_images = combined_images[..., 1]
            clean_images = combined_images[..., 2]
            return galaxy_images, psf_images, clean_images
        else:
            raise ValueError("3 channels requires both PSF and clean images")
    else:
        raise ValueError(f"Unexpected number of channels: {combined_images.shape[-1]}")


def _safe_draw(obj, npix, scale):
    """Draw ``obj`` onto an ``(npix, npix)`` array.

    Attempt a normal (FFT) draw and fall back to slower real-space rendering if
    the FFT is too large, which can happen for very compact objects.
    """
    try:
        return obj.drawImage(nx=npix, ny=npix, scale=scale).array
    except galsim.errors.GalSimFFTSizeError:
        return obj.drawImage(nx=npix, ny=npix, scale=scale, method="real_space").array


def sim_func(
    g1,
    g2,
    hlr=1.0,
    flux=1.0,
    psf_fwhm=0.5,
    nse_sd=1e-5,
    type="gauss",
    npix=53,
    scale=0.141,
    seed=42,
    exp="ideal",
    apply_psf_shear=False,
    psf_shear_range=0.05,
    ud=None,
    psf_files=None,
    base_shear_g1=0.0,
    base_shear_g2=0.0,
    compute_metacal=True,
    compute_psf_admom=True,
):
    """Simulate a single galaxy observation and return an ngmix ``Observation``.

    Builds a sheared, randomly-shifted galaxy, convolves it with the chosen PSF,
    draws the noisy image, fits the PSF adaptive moments, and packages everything
    into an ngmix ``Observation``. The metadata also stores the +/- e1/e2 sheared
    counterparts used for metacalibration-style responses.

    ``compute_metacal`` and ``compute_psf_admom`` gate the two expensive pieces of
    per-object work that plain network *training* does not use: the four metacal
    (+/- e1/e2) reconvolutions/draws and the ngmix adaptive-moments fit of the
    PSF. Both default to ``True`` (unchanged behavior); pass ``False`` to skip
    them for a large speed-up. When the admom fit is skipped the PSF ``e1/e2/T``
    metadata are set to NaN (they are only needed for the ``psf_*`` labels).

    Args:
        g1, g2: Intrinsic shear of the galaxy.
        hlr: Half-light radius (arcsec).
        flux: Total flux of the galaxy.
        psf_fwhm: Gaussian PSF FWHM (arcsec) for ``exp='ideal'``.
        nse_sd: Standard deviation of the additive Gaussian noise.
        type: Galaxy profile, ``'exp'`` or ``'gauss'``.
        npix: Stamp size in pixels (square).
        scale: Pixel scale in arcsec/pixel.
        seed: Random seed for this object's noise and shifts.
        exp: PSF mode, ``'ideal'`` (analytic Gaussian) or ``'superbit'`` (PSFEx).
        apply_psf_shear: Apply a random shear to the analytic PSF.
        psf_shear_range: Half-width of the uniform PSF-shear distribution.
        ud: A ``galsim.UniformDeviate`` used to sample the SuperBIT PSF.
        psf_files: List of ``.psf`` files to draw from for ``exp='superbit'``.
        base_shear_g1, base_shear_g2: Constant shear applied before the PSF.

    Returns:
        ngmix.Observation: The noisy galaxy observation, with its ``psf`` set and
        ``meta`` populated with ``snr`` and the metacal images.
    """
    rng = np.random.RandomState(seed=seed)

    gsp = galsim.GSParams(maximum_fft_size=32768)

    # Create a galaxy object
    if type == "exp":
        gal = galsim.Exponential(half_light_radius=hlr, flux=flux).shear(g1=g1, g2=g2)
    elif type == "gauss":
        gal = galsim.Gaussian(half_light_radius=hlr, flux=flux).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    # Generate PSF shear values if requested
    psf_g1, psf_g2 = 0.0, 0.0
    if apply_psf_shear:
        psf_g1 = rng.uniform(-psf_shear_range, psf_shear_range)
        psf_g2 = rng.uniform(-psf_shear_range, psf_shear_range)

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    sheared_gal = gal.shift(dx, dy)

    sheared_gal = sheared_gal.shear(g1=base_shear_g1, g2=base_shear_g2)

    # Convolve with PSF
    if exp == "ideal":
        psf = galsim.Gaussian(fwhm=psf_fwhm)

        if apply_psf_shear:
            psf = psf.shear(g1=psf_g1, g2=psf_g2)

        obj = galsim.Convolve([sheared_gal, psf], gsparams=gsp)

    elif exp == "superbit":
        psf = import_psf(psf_files, ud)
        obj = galsim.Convolve([psf, sheared_gal], gsparams=gsp)

    else:
        raise ValueError("For now only supported experiments are 'ideal' or 'superbit'")

    # Draw images
    obj_im = _safe_draw(obj.withGSParams(gsp), npix, scale)
    psf_im = _safe_draw(psf.withGSParams(gsp), npix, scale)

    # Metacal (+/- e1/e2) reconvolutions + draws -- skipped unless requested,
    # since plain training never uses them. Identical for ideal/superbit once the
    # PSF is defined, so it lives here rather than duplicated per branch.
    if compute_metacal:
        obj_e1_positive = galsim.Convolve([sheared_gal.shear(g1=0.01, g2=0.0), psf], gsparams=gsp)
        obj_e1_negative = galsim.Convolve([sheared_gal.shear(g1=-0.01, g2=0.0), psf], gsparams=gsp)
        obj_e2_positive = galsim.Convolve([sheared_gal.shear(g1=0.0, g2=0.01), psf], gsparams=gsp)
        obj_e2_negative = galsim.Convolve([sheared_gal.shear(g1=0.0, g2=-0.01), psf], gsparams=gsp)
        e1_positive_im = _safe_draw(obj_e1_positive.withGSParams(gsp), npix, scale)
        e1_negative_im = _safe_draw(obj_e1_negative.withGSParams(gsp), npix, scale)
        e2_positive_im = _safe_draw(obj_e2_positive.withGSParams(gsp), npix, scale)
        e2_negative_im = _safe_draw(obj_e2_negative.withGSParams(gsp), npix, scale)

    # Add noise
    nse = rng.normal(size=obj_im.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.shape, scale=nse_sd)

    cen = npix // 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen + dy / scale, col=cen + dx / scale)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)

    # Add small noise to PSF for stability
    target_psf_noise = psf_im.max() / 1000.0

    psf_obs = ngmix.Observation(
        image=psf_im,
        weight=np.ones_like(psf_im) / target_psf_noise**2,
        jacobian=psf_jac,
    )
    if compute_psf_admom:
        admom_psf_measurement = get_admoms_ngmix_fit(obs=psf_obs, reduced=True)
        psf_obs.update_meta_data(
            {
                "e1": admom_psf_measurement["e1"],
                "e2": admom_psf_measurement["e2"],
                "T": admom_psf_measurement["T"],
                "admom_flags": admom_psf_measurement["flags"],
            }
        )
    else:
        # These feed only the psf_e1/psf_e2/psf_T labels; when those are not
        # requested, skip the (expensive) fit and leave NaN placeholders so the
        # metadata keys still exist.
        psf_obs.update_meta_data({"e1": np.nan, "e2": np.nan, "T": np.nan, "admom_flags": -1})

    obj_obs = ngmix.Observation(
        image=obj_im + nse,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    # Calculate SNR using ngmix built-in method
    snr = obj_obs.get_s2n()

    obj_meta = {"snr": snr}
    if compute_metacal:
        obj_meta.update(
            {
                "e1_positive": e1_positive_im,
                "e1_negative": e1_negative_im,
                "e2_positive": e2_positive_im,
                "e2_negative": e2_negative_im,
            }
        )
    obj_obs.update_meta_data(obj_meta)

    return obj_obs


def search_psf_files(path):
    """Return a list of all ``*.psf`` files directly under ``path``."""
    all_psf_files = []
    search_path = os.path.join(path, "*.psf")
    all_psf_files.extend(glob(search_path))
    return all_psf_files


def get_background_file(psf_file):
    """Map a PSFEx ``.psf`` file path to its matching sky-background FITS file.

    Replaces the ``_starcat.psf`` suffix with ``.bkg_rms.fits`` and swaps the
    ``psfex-output`` directory for the sibling ``sky_backgrounds`` directory.
    """
    # Extract base name without directory
    fname = os.path.basename(psf_file)

    # Remove the "_starcat.psf" suffix
    stem = fname.replace("_starcat.psf", "")

    # Build the new file name
    new_fname = stem + ".bkg_rms.fits"

    # Replace psfex-output with sky_backgrounds
    new_dir = os.path.dirname(psf_file).replace("psfex-output", "sky_backgrounds")

    return os.path.join(new_dir, new_fname)


def import_psf(
    psf_files, ud, xsize=WCS_PARAMS["image_xsize"], ysize=WCS_PARAMS["image_ysize"], margin=MARGIN
):
    """Sample an empirical PSFEx PSF at a random image position.

    Picks a random exposure from ``psf_files`` and a random position within the
    image (inset by ``margin``), then evaluates the PSFEx model there using the
    WCS built from :data:`WCS_PARAMS`.

    Args:
        psf_files: List of PSFEx ``.psf`` files to choose from.
        ud: A ``galsim.UniformDeviate`` providing the random draws.
        xsize, ysize: Image dimensions (pixels) used to sample the position.
        margin: Pixel margin kept clear of the image edges.

    Returns:
        A ``galsim`` PSF object evaluated at the sampled position.
    """
    maxexp = len(psf_files)
    # random position
    x = margin + (xsize - 2 * margin) * ud()
    y = margin + (ysize - 2 * margin) * ud()

    # random integer between 1 and maxexp
    exp = int(maxexp * ud())
    image_pos = galsim.PositionD(x=x, y=y)
    psf_file = psf_files[exp]

    wcs = create_wcs_from_params(WCS_PARAMS)
    psf = galsim.des.DES_PSFEx(psf_file, wcs=wcs)
    this_psf = psf.getPSF(image_pos)

    return this_psf
