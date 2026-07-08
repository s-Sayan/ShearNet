# main.py or notebook cell

# ============================================================
# CONFIGURATION
# ============================================================
import numpy as np
import ngmix
import galsim
import galsim.des
import matplotlib.pyplot as plt
import ipdb
from IPython.display import display, Math
import matplotlib.colors as colors
from astropy.table import Table
from astropy.io import fits
import shearnet.utils.superbit as utils  # vendored from superbit-lensing (see shearnet/utils/superbit.py)
from multiprocessing import Pool
import os

nproc = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

from helpers import (
    _get_priors,
    progress,
    make_struct,
    process_obs,
    process_obs_response,
    shear_data_to_table,
    leakage_response_to_table,
    get_init_guess,
    get_em_ngauss,
    get_coellip_ngauss
)

norm2 = colors.SymLogNorm(linthresh=1e-4, base=np.e)


import yaml
import argparse

# ========================== CONFIG LOADING ==========================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Shear calibration simulation with ngmix")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    return parser.parse_args()

args = parse_args()
_config = load_config(args.config)

# ----- General -----
SEED  = _config["eval"]["seed"]
NOISE = _config["image"]["noise_sd"]
SCALE = _config["image"]["pixel_scale"]
NPIX  = _config["image"]["stamp_size"]
PSF_NPIX = _config["psf"]["stamp_size"]
GALSIM_PSF = galsim.des.DES_PSFEx(_config["paths"]["psfex_model_file"], wcs=utils.get_galsim_tanwcs())

# ----- GAL settings -----
GAL_HLR  = _config["galaxy"]["hlr"]  if _config["galaxy"]["hlr_type"]  == "constant" else "catalog"
GAL_FLUX = _config["galaxy"]["flux"] if _config["galaxy"]["flux_type"] == "constant" else "catalog"

NOBS = _config["eval"]["n_obs"]

# ----- ngmix fit models -----
LEAKAGE_CFG = _config["eval"]["leakage"]
PSF_MODEL = LEAKAGE_CFG["psf_model"]
GAL_MODEL = _config["eval"]["gal_model"]

# ----- PSF response (metacal leakage) correction -----
# When enabled, the measured shear (ngmix and ShearNet) is corrected for PSF
# leakage using the metacal PSF response, following
#   ngmix/tests/test_metacal_galsim_psf_response.py :
#       R11_psf = (g['1p_psf'] - g['1m_psf']) / (2*step)
#       g_corrected = g['noshear'] - g_psf * R11_psf
PSF_RESPONSE = LEAKAGE_CFG.get("psf_response", False)
RECONV_PSF   = LEAKAGE_CFG.get("reconv_psf", "dilate")
MCAL_STEP    = LEAKAGE_CFG.get("metacal_step", 0.01)
# The metacal PSF response is only physical for a PSF-deconvolving estimator
# (ngmix). For ShearNet it measures OOD sensitivity to the reconvolution, not
# leakage, so it is reported as a diagnostic but NOT applied unless requested.
PSF_RESPONSE_SHEARNET = LEAKAGE_CFG.get("psf_response_shearnet", False)
# PSF-shear response terms ('*_psf') require the 'dilate' reconvolution PSF.
MCAL_TYPES = (
    ["noshear", "1p_psf", "1m_psf"] if RECONV_PSF == "dilate" else ["noshear"]
)

# ----- ShearNet -----
INCLUDE_SN    = _config["eval"]["include_shearnet"]
SN_MODEL_NAME = _config["meta"]["model_name"]
OUTPUT_KEYS   = tuple(_config["model"]["output_keys"])
GAP           = _config["model"].get("gap", False)

# ----- Catalog -----
COSMOS_CAT_FNAME = _config["paths"]["eval_catalog"]
with fits.open(COSMOS_CAT_FNAME) as hdul:
    cosmos_cat = hdul[1].data

# ----- Output -----
OUTPUT_FITS = os.path.join(_config["paths"]["root"], LEAKAGE_CFG["output"])

NTRY = 20
LM_PARS = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
PSF_LM_PARS = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
EM_PARS={'tol': 1.0e-6, 'maxiter': 50000}

# ========== Load up ShearNet state here =========
import jax
import jax.numpy as jnp
if INCLUDE_SN:
    from shearnet.cli.evaluate import load_model as _initialize_model, load_config as _eval_load_config
    from shearnet.utils.normalization import load_normalizer, inverse_transform_labels
    
    eval_config = argparse.Namespace(
        model_name=SN_MODEL_NAME, config=None, seed=None,
        test_samples=None, mcal=False, plot=False, plot_animation=False,
        process_psf=None, galaxy_type=None, psf_type=None,
        apply_psf_shear=None, psf_shear_range=None
    )
    eval_config = _eval_load_config(eval_config)
    
    dummy_images = jnp.ones((1, NPIX, NPIX))
    STATE = _initialize_model(eval_config, dummy_images, dummy_images)
    
    # load normalizer from same directory as training config
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
    _normalizer_path = os.path.join(data_path, 'plots', SN_MODEL_NAME, 'label_normalizer.npz')
    NORM_PARAMS = load_normalizer(_normalizer_path) if os.path.exists(_normalizer_path) else None

    def _sn_predict(params, gal, psf, output_keys, gap):
         return STATE.apply_fn(params, gal, psf, output_keys=output_keys, gap=gap, deterministic=True)
    SN_PREDICT = jax.jit(_sn_predict, static_argnames=("output_keys", "gap"))
else:
    STATE = None
    NORM_PARAMS = None
    SN_PREDICT = None

# ============================================================
# FUNCTIONS THAT NEED galsim + cosmos_cat stay here
# ============================================================
def make_data(rng, noise):

    scale = SCALE

    index = rng.randint(len(cosmos_cat))
    q    = cosmos_cat['Q'][index]
    phi  = cosmos_cat['PHI'][index] * galsim.radians
    if GAL_HLR == "catalog":
        gal_hlr = cosmos_cat['HLR'][index]
    else:
        gal_hlr = GAL_HLR
    if GAL_FLUX == "catalog":
        gal_flux = cosmos_cat['FLUX'][index]
    else:
        gal_flux = GAL_FLUX

    npix_psf = PSF_NPIX
    npix = NPIX
    dy, dx = rng.uniform(low=-scale / 2, high=scale / 2, size=2)

    image_xsize=9600 
    image_ysize=6400
    margin = 500
    # random integer pixel coordinates inside [margin, size - margin - 1]
    x_im = rng.randint(margin, image_xsize - margin)
    y_im = rng.randint(margin, image_ysize - margin)
    image_position = galsim.PositionD(x_im, y_im)
    psf = GALSIM_PSF.getPSF(image_position)
    gsp=galsim.GSParams(maximum_fft_size=32768)
    obj0 = galsim.Exponential(half_light_radius=gal_hlr, flux=gal_flux).shear(q=q, beta=phi)

    g1_th, g2_th, _, _, _ = utils.g_from_gal_jac(obj0)
    g_th = np.array([g1_th, g2_th])

    obj0 = obj0.shift(dx=dx, dy=dy)


    obj0_psf = galsim.Convolve(psf, obj0, gsparams=gsp)

    psf_im = psf.withGSParams(gsp).drawImage(nx=npix_psf, ny=npix_psf, scale=scale).array
    im_0 = obj0_psf.withGSParams(gsp).drawImage(nx=npix, ny=npix, scale=scale).array

    im_noise = rng.normal(scale=noise, size=im_0.shape)
    im_0 += im_noise

    cen = (np.array(im_0.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy/scale, col=cen[1] + dx/scale, scale=scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
    )

    wt = im_0*0 + 1.0/noise**2
    psf_noise = psf_im.max() / 1000.0
    psf_wt = psf_im*0 + 1.0/psf_noise**2

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian,
    )

    obs0 = ngmix.Observation(im_0, weight=wt, jacobian=jacobian, psf=psf_obs)

    return obs0, g_th, gal_hlr, gal_flux

def process_single_object(args):
    i, base_seed, noise = args

    # independent RNG per worker
    rng = np.random.RandomState(base_seed + i)

    obs, g_th, gal_hlr, gal_flux = make_data(rng=rng, noise=noise)

    # create priors & runners locally (safe)
    prior = _get_priors(base_seed + i)

    TGUESS, _ = get_init_guess(obs)

    fitter = ngmix.fitting.Fitter(model=GAL_MODEL, prior=prior, fit_pars=LM_PARS)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng, T=TGUESS, prior=prior
    )

    if 'em' in PSF_MODEL:
        psf_ngauss = get_em_ngauss(PSF_MODEL)
        psf_fitter = ngmix.em.EMFitter(maxiter=EM_PARS['maxiter'], tol=EM_PARS['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif 'coellip' in PSF_MODEL:
        psf_ngauss = get_coellip_ngauss(PSF_MODEL)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=PSF_LM_PARS)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif PSF_MODEL == 'gauss':
        psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=PSF_LM_PARS)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)


    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter, guesser=psf_guesser, ntry=NTRY
    )
    runner = ngmix.runners.Runner(
        fitter=fitter, guesser=guesser, ntry=NTRY
    )

    boot = ngmix.bootstrap.Bootstrapper(
        runner=runner,
        psf_runner=psf_runner,
    )

    data, gal_im, psf_im = process_obs(obs, boot, return_images=True)
    return data, g_th, gal_im, psf_im, gal_hlr, gal_flux


def process_single_object_response(args):
    """
    Metacal PSF-response variant of ``process_single_object``.

    Runs a metacal bootstrap with the PSF-shear types ('1p_psf', '1m_psf') so
    the per-object PSF response can be measured, and returns the sheared images
    for each shear type so ShearNet can be evaluated on them in the main
    process (its PSF response is measured the same way).
    """
    i, base_seed, noise = args

    rng = np.random.RandomState(base_seed + i)

    obs, g_th, gal_hlr, gal_flux = make_data(rng=rng, noise=noise)

    prior = _get_priors(base_seed + i)

    TGUESS, _ = get_init_guess(obs)

    fitter = ngmix.fitting.Fitter(model=GAL_MODEL, prior=prior, fit_pars=LM_PARS)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng, T=TGUESS, prior=prior
    )

    if 'em' in PSF_MODEL:
        psf_ngauss = get_em_ngauss(PSF_MODEL)
        psf_fitter = ngmix.em.EMFitter(maxiter=EM_PARS['maxiter'], tol=EM_PARS['tol'])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif 'coellip' in PSF_MODEL:
        psf_ngauss = get_coellip_ngauss(PSF_MODEL)
        psf_fitter = ngmix.fitting.CoellipFitter(ngauss=psf_ngauss, fit_pars=PSF_LM_PARS)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=psf_ngauss)
    elif PSF_MODEL == 'gauss':
        psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=PSF_LM_PARS)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)

    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter, guesser=psf_guesser, ntry=NTRY
    )
    runner = ngmix.runners.Runner(
        fitter=fitter, guesser=guesser, ntry=NTRY
    )

    boot = ngmix.metacal.MetacalBootstrapper(
        runner=runner,
        psf_runner=psf_runner,
        rng=rng,
        psf=RECONV_PSF,
        step=MCAL_STEP,
        types=MCAL_TYPES,
    )

    data, mcal_images = process_obs_response(obs, boot)
    return data, g_th, mcal_images, gal_hlr, gal_flux


args_list = [(i, SEED, NOISE) for i in range(NOBS)]

data_list = []
gth_list = []
g_sn_raw_list = []
img_buffer = []

BATCH_SIZE = 500

_ok = list(OUTPUT_KEYS)
g_idx = [_ok.index("g1"), _ok.index("g2")]


def _run_shearnet_batch(buf):
    if STATE is None:
        return
    n = len(buf)
    base = len(data_list) - n
    gal = jnp.stack([r[2] for r in buf])
    psf = jnp.stack([r[3] for r in buf])
    preds = np.array(SN_PREDICT(STATE.params, gal, psf, OUTPUT_KEYS, GAP))
    if NORM_PARAMS is not None:
        preds = inverse_transform_labels(preds, NORM_PARAMS)
    for k in range(n):
        data_list[base + k][0]["g_sn"] = preds[k][g_idx]
    g_sn_raw_list.append(preds[:, g_idx])


def _run_shearnet_batch_response(buf):
    """
    Evaluate ShearNet on every metacal shear type (noshear, 1p_psf, 1m_psf)
    so its per-object PSF response can be formed downstream. Predictions are
    written back into the per-type struct records.
    """
    if STATE is None:
        return
    n = len(buf)
    base = len(data_list) - n
    stypes = list(buf[0][2].keys())  # mcal_images keys

    for stype in stypes:
        gal = jnp.stack([r[2][stype][0] for r in buf])
        psf = jnp.stack([r[2][stype][1] for r in buf])
        preds = np.array(SN_PREDICT(STATE.params, gal, psf, OUTPUT_KEYS, GAP))
        if NORM_PARAMS is not None:
            preds = inverse_transform_labels(preds, NORM_PARAMS)
        for k in range(n):
            idx = np.where(data_list[base + k]["shear_type"] == stype)[0][0]
            data_list[base + k][idx]["g_sn"] = preds[k][g_idx]


hlr_list_out, flux_list_out = [], []

if PSF_RESPONSE:
    print(
        f"[psf_leakage] PSF-response correction ON "
        f"(reconv_psf={RECONV_PSF}, metacal_step={MCAL_STEP}); "
        f"'g' and 'g_sn' hold corrected values, 'g_raw'/'g_sn_raw' the uncorrected ones."
    )
    _worker = process_single_object_response
    _batch = _run_shearnet_batch_response
    # result layout: (data, g_th, mcal_images, gal_hlr, gal_flux)
    _hlr_idx, _flux_idx = 3, 4
else:
    _worker = process_single_object
    _batch = _run_shearnet_batch
    # result layout: (data, g_th, gal_im, psf_im, gal_hlr, gal_flux)
    _hlr_idx, _flux_idx = 4, 5

with Pool(processes=nproc) as pool:
    for result in pool.imap(_worker, args_list):
        data_list.append(result[0])
        gth_list.append(result[1])
        hlr_list_out.append(result[_hlr_idx])
        flux_list_out.append(result[_flux_idx])
        img_buffer.append(result)
        if len(img_buffer) == BATCH_SIZE:
            _batch(img_buffer)
            img_buffer.clear()
    if img_buffer:
        _batch(img_buffer)
        img_buffer.clear()

if PSF_RESPONSE:
    tab, rbar_psf, rbar_psf_sn = leakage_response_to_table(
        data_list, step=MCAL_STEP, apply_shearnet=PSF_RESPONSE_SHEARNET
    )
    print(
        f"[psf_leakage] ensemble metacal PSF response  "
        f"Rbar_psf(ngmix)={rbar_psf:.4f}  Rbar_psf(shearnet)={rbar_psf_sn:.4f}"
    )
    if STATE is not None and not PSF_RESPONSE_SHEARNET:
        print(
            "[psf_leakage] NOTE: ShearNet g_sn left UNCORRECTED "
            "(metacal PSF response is not physical leakage for a network; "
            "set eval.leakage.psf_response_shearnet: true to override)."
        )
    tab["g_th"] = np.asarray(gth_list)
    tab["gal_hlr_th"]  = np.array(hlr_list_out)
    tab["gal_flux_th"] = np.array(flux_list_out)
    if STATE is None:
        # no ShearNet: keep corrected/raw g_sn columns explicitly NaN
        tab["g_sn"]     = np.full((len(tab), 2), np.nan)
        tab["g_sn_raw"] = np.full((len(tab), 2), np.nan)
else:
    tab = shear_data_to_table(data_list)
    tab["g_th"] = np.asarray(gth_list)
    tab["gal_hlr_th"]  = np.array(hlr_list_out)
    tab["gal_flux_th"] = np.array(flux_list_out)

    if STATE is not None:
        tab["g_sn"] = np.concatenate(g_sn_raw_list, axis=0)
    else:
        tab["g_sn"] = np.full((len(tab), 2), np.nan)

fname = OUTPUT_FITS

# Create directory if needed
outdir = os.path.dirname(fname)
if outdir != "":
    os.makedirs(outdir, exist_ok=True)

hdu_primary = fits.PrimaryHDU()
hdu = fits.BinTableHDU(tab, name="results")

hdul = fits.HDUList([hdu_primary, hdu])
hdul.writeto(fname, overwrite=True)
