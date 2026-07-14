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
    shear_data_to_table,
    jackknife_mc_v2,
    get_em_ngauss,
    get_coellip_ngauss,
    get_init_guess,
    superscript,
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

# General
SEED       = _config["eval"]["seed"]
SHEAR_TRUE = _config["eval"]["bias"]["shear_true"]

NOISE      = _config["image"]["noise_sd"]
PSF_NOISE  = _config["psf"]["noise"]
SCALE      = _config["image"]["pixel_scale"]
NPIX       = _config["image"]["stamp_size"]

# GAL settings
GAL_HLR  = _config["galaxy"]["hlr"]  if _config["galaxy"]["hlr_type"]  == "constant" else "catalog"
GAL_FLUX = _config["galaxy"]["flux"] if _config["galaxy"]["flux_type"] == "constant" else "catalog"

# PSF settings
PSF_FWHM          = _config["psf"]["mode"]
PSF_GAUSSIAN_FWHM = _config["psf"]["gaussian_fwhm"]
PSF_NPIX          = _config["psf"]["stamp_size"]

NOBS = _config["eval"]["n_obs"]
NJAC = _config["eval"]["bias"]["n_jackknife"]

GALSIM_PSF = galsim.des.DES_PSFEx(_config["paths"]["psfex_model_file"], wcs=utils.get_galsim_tanwcs())

# ----- ngmix fit models -----
PSF_MODEL = _config["eval"]["bias"]["psf_model"]
GAL_MODEL = _config["eval"]["gal_model"]

# ----- PSF response (metacal leakage) correction -----
# When enabled, the measured shear (ngmix and ShearNet) is corrected for PSF
# leakage using the metacal PSF response, following
#   ngmix/tests/test_metacal_galsim_psf_response.py :
#       R11_psf = (g['1p_psf'] - g['1m_psf']) / (2*step)
#       g_corrected = g['noshear'] - g_psf * R11_psf
# This mostly shifts the additive bias c (and slightly m), which is exactly
# the PSF-leakage term the leakage benchmark isolates.
PSF_RESPONSE = _config["eval"]["bias"].get("psf_response", False)
RECONV_PSF   = _config["eval"]["bias"].get("reconv_psf", "dilate")
MCAL_STEP    = _config["eval"]["bias"].get("metacal_step", 0.01)
# metacal PSF response is not physical leakage for a network -> ShearNet
# correction is opt-in only (see the correction block below).
PSF_RESPONSE_SHEARNET = _config["eval"]["bias"].get("psf_response_shearnet", False)

# ----- Skip-deconvolution ("direct") PSF-response correction -----
# Measure R11_psf by shearing the real PSF by +/- step and convolving the galaxy
# with it (no metacal deconvolution/dilation), then subtract gpsf*Rbar_psf from
# g_noshear / g_sn_noshear. Switchable per shape-measurement software; when on it
# takes precedence over the metacal psf_response correction. This is the
# physically-correct PSF response for a non-deconvolving network like ShearNet.
PSF_RESPONSE_DIRECT_NGMIX = _config["eval"]["bias"].get("psf_response_direct_ngmix", False)
PSF_RESPONSE_DIRECT_SN    = _config["eval"]["bias"].get("psf_response_direct_shearnet", False)
PSF_RESPONSE_DIRECT = PSF_RESPONSE_DIRECT_NGMIX or PSF_RESPONSE_DIRECT_SN
PSF_DIRECT_STEP = _config["eval"]["bias"].get("psf_response_direct_step", MCAL_STEP)

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
OUTPUT_FITS = os.path.join(_config["paths"]["root"], _config["eval"]["bias"]["output"])

TGUESS = 4 * SCALE**2
NTRY = 20
LM_PARS = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
PSF_LM_PARS = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
EM_PARS={'tol': 1.0e-6, 'maxiter': 50000}
MCAL_PARS = {"psf": RECONV_PSF, "mcal_shear": MCAL_STEP}
# PSF-shear response terms ('*_psf') require the 'dilate' reconvolution PSF.
if PSF_RESPONSE and RECONV_PSF == "dilate":
    TYPES = ["noshear", "1p", "1m", "1p_psf", "1m_psf"]
else:
    TYPES = ["noshear", "1p", "1m"]

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
def make_data(rng, noise, shear_true):
    psf_noise = PSF_NOISE

    scale = SCALE
    psf_fwhm = PSF_FWHM

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

    if psf_fwhm == "superbit":
        image_xsize=9600 
        image_ysize=6400
        margin = 500
        # random integer pixel coordinates inside [margin, size - margin - 1]
        x_im = rng.randint(margin, image_xsize - margin)
        y_im = rng.randint(margin, image_ysize - margin)
        image_position = galsim.PositionD(x_im, y_im)
        psf = GALSIM_PSF.getPSF(image_position)
    else:
        psf = galsim.Gaussian(fwhm=PSF_GAUSSIAN_FWHM)

    gsp=galsim.GSParams(maximum_fft_size=32768)
    
    obj0 = galsim.Exponential(half_light_radius=gal_hlr, flux=gal_flux).shear(q=q, beta=phi)
    objp = obj0.shear(g1=shear_true, g2=0.0)
    objm = obj0.shear(g1=-shear_true, g2=0.0)

    # Inferring input theory values from galsim jacobians
    g1_th_p, g2_th_p, _, _, _ = utils.g_from_gal_jac(objp)
    g1_th_m, g2_th_m, _, _, _ = utils.g_from_gal_jac(objm)

    g_th_p = np.array([g1_th_p, g2_th_p])
    g_th_m = np.array([g1_th_m, g2_th_m])

    objp = objp.shift(dx=dx, dy=dy)
    objm = objm.shift(dx=dx, dy=dy)

    obj0_psf = galsim.Convolve(psf, obj0, gsparams=gsp)
    objp_psf = galsim.Convolve(psf, objp, gsparams=gsp)
    objm_psf = galsim.Convolve(psf, objm, gsparams=gsp)

    psf_im = psf.withGSParams(gsp).drawImage(nx=npix_psf, ny=npix_psf, scale=scale).array
    im_0 = obj0_psf.withGSParams(gsp).drawImage(nx=npix, ny=npix, scale=scale).array
    im_p = objp_psf.withGSParams(gsp).drawImage(nx=npix, ny=npix, scale=scale).array
    im_m = objm_psf.withGSParams(gsp).drawImage(nx=npix, ny=npix, scale=scale).array

    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    im_noise = rng.normal(scale=noise, size=im_0.shape)
    im_0 += im_noise
    im_p += im_noise
    im_m += im_noise

    cen = (np.array(im_0.shape) - 1.0) / 2.0
    psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0

    jacobian = ngmix.DiagonalJacobian(
        row=cen[0] + dy / scale, col=cen[1] + dx / scale, scale=scale
    )
    psf_jacobian = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

    wt = im_0 * 0 + 1.0 / noise**2
    psf_wt = psf_im * 0 + 1.0 / psf_noise**2

    psf_obs = ngmix.Observation(psf_im, weight=psf_wt, jacobian=psf_jacobian)

    obs0 = ngmix.Observation(im_0, weight=wt, jacobian=jacobian, psf=psf_obs)
    obsp = ngmix.Observation(im_p, weight=wt, jacobian=jacobian, psf=psf_obs)
    obsm = ngmix.Observation(im_m, weight=wt, jacobian=jacobian, psf=psf_obs)

    # ---- skip-deconvolution ("direct") PSF-shear variants ----
    # Shear the REAL psf by +/- PSF_DIRECT_STEP in g1 and convolve the (already
    # shear_true-sheared, shifted) galaxies objp/objm with it. Reuse the same
    # galaxy noise field im_noise so it cancels in the +/- difference. No metacal.
    direct_p, direct_m = None, None
    if PSF_RESPONSE_DIRECT:
        direct_p, direct_m = {}, {}
        for _s, _key in ((+PSF_DIRECT_STEP, "1p_psf_direct"),
                         (-PSF_DIRECT_STEP, "1m_psf_direct")):
            psf_s = psf.shear(g1=_s, g2=0.0)
            psf_s_im = psf_s.withGSParams(gsp).drawImage(
                nx=npix_psf, ny=npix_psf, scale=scale
            ).array
            psf_s_im = psf_s_im + rng.normal(scale=psf_noise, size=psf_s_im.shape)
            psf_s_wt = psf_s_im * 0 + 1.0 / psf_noise**2
            psf_s_obs = ngmix.Observation(psf_s_im, weight=psf_s_wt, jacobian=psf_jacobian)
            for _obj, _store in ((objp, direct_p), (objm, direct_m)):
                im_s = galsim.Convolve(psf_s, _obj, gsparams=gsp).withGSParams(gsp).drawImage(
                    nx=npix, ny=npix, scale=scale
                ).array + im_noise
                _store[_key] = ngmix.Observation(
                    im_s, weight=wt, jacobian=jacobian, psf=psf_s_obs
                )

    if psf_fwhm == "superbit":
        return obs0, obsp, obsm, g_th_p, g_th_m, gal_hlr, gal_flux, x_im, y_im, direct_p, direct_m
    else:
        return obs0, obsp, obsm, g_th_p, g_th_m, gal_hlr, gal_flux, direct_p, direct_m

def process_single_object(args):
    i, base_seed, noise, shear_true = args

    # independent RNG per worker
    rng = np.random.RandomState(base_seed + i)

    if PSF_FWHM == "superbit":
        obs0, obsp, obsm, g_th_p, g_th_m, gal_hlr, gal_flux, x_im, y_im, direct_p, direct_m = make_data(rng=rng, noise=noise, shear_true=shear_true)
    else:
        obs0, obsp, obsm, g_th_p, g_th_m, gal_hlr, gal_flux, direct_p, direct_m = make_data(rng=rng, noise=noise, shear_true=shear_true)

    # create priors & runners locally (safe)
    prior = _get_priors(base_seed + i)

    TGUESS, _ = get_init_guess(obs0)

    fitter = ngmix.fitting.Fitter(model=GAL_MODEL, prior=prior, fit_pars=LM_PARS)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng, T=TGUESS, prior=prior
    )

    psf_fitter = ngmix.fitting.Fitter(model=PSF_MODEL, fit_pars=PSF_LM_PARS)
    psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)

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
        psf=MCAL_PARS["psf"],
        step=MCAL_PARS["mcal_shear"],
        types=TYPES,
    )


    # Always return images so main process can run ShearNet
    data_p, mcal_images_p = process_obs(obsp, boot, return_images=True)
    data_m, mcal_images_m = process_obs(obsm, boot, return_images=True)
    raw_p   = obsp.image.copy()
    raw_m   = obsm.image.copy()
    psf_im  = obsp.psf.image.copy()

    # ---- skip-deconvolution ("direct") PSF response ----
    # Append the PSF-sheared variants as extra shear types ('1p_psf_direct',
    # '1m_psf_direct') so they flow through the existing ShearNet batch + table
    # machinery. ngmix is fit here (plain, non-metacal bootstrapper) only when
    # requested; otherwise its shape is left NaN and only ShearNet evaluates them.
    if PSF_RESPONSE_DIRECT:
        plain_boot = (
            ngmix.bootstrap.Bootstrapper(runner=runner, psf_runner=psf_runner)
            if PSF_RESPONSE_DIRECT_NGMIX else None
        )

        def _augment(direct_obs, data_struct, images):
            extra = []
            for key, obs_d in direct_obs.items():
                res = plain_boot.go(obs_d) if plain_boot is not None else {"flags": 1}
                extra.append(make_struct(res=res, obs=obs_d, shear_type=key))
                images[key] = (obs_d.image.copy(), obs_d.psf.image.copy())
            return np.hstack([data_struct, *extra])

        data_p = _augment(direct_p, data_p, mcal_images_p)
        data_m = _augment(direct_m, data_m, mcal_images_m)

    if PSF_FWHM == "superbit":
        return data_p, data_m, g_th_p, g_th_m, mcal_images_p, mcal_images_m, raw_p, raw_m, psf_im, gal_hlr, gal_flux, x_im, y_im
    else:
        return data_p, data_m, g_th_p, g_th_m, mcal_images_p, mcal_images_m, raw_p, raw_m, psf_im, gal_hlr, gal_flux

args = [(i, SEED, NOISE, SHEAR_TRUE, STATE) for i in range(NOBS)]

BATCH_SIZE = 500

args_list = [(i, SEED, NOISE, SHEAR_TRUE) for i in range(NOBS)]

data_list_p, data_list_m = [], []
gth_list_p, gth_list_m   = [], []
g_sn_raw_list_p, g_sn_raw_list_m = [], []
img_buffer = []
sn_sigma_raw_list = []
sn_flux_raw_list  = []

_ok = list(OUTPUT_KEYS)
g_idx = [_ok.index("g1"), _ok.index("g2")]

def _run_shearnet_batch(buf):
    if STATE is None:
        return
    n = len(buf)
    base = len(data_list_p) - n
    stypes = list(buf[0][4].keys())  # mcal_images_p keys

    for stype in stypes:
        gal_p = jnp.stack([r[4][stype][0] for r in buf])
        psf_p = jnp.stack([r[4][stype][1] for r in buf])
        gal_m = jnp.stack([r[5][stype][0] for r in buf])
        psf_m = jnp.stack([r[5][stype][1] for r in buf])

        preds_p = np.array(SN_PREDICT(STATE.params, gal_p, psf_p, OUTPUT_KEYS, GAP))
        preds_m = np.array(SN_PREDICT(STATE.params, gal_m, psf_m, OUTPUT_KEYS, GAP))

        if NORM_PARAMS is not None:
            preds_p = inverse_transform_labels(preds_p, NORM_PARAMS)
            preds_m = inverse_transform_labels(preds_m, NORM_PARAMS)

        for k in range(n):
            idx_p = np.where(data_list_p[base + k]["shear_type"] == stype)[0][0]
            idx_m = np.where(data_list_m[base + k]["shear_type"] == stype)[0][0]
            data_list_p[base + k][idx_p]["g_sn"] = preds_p[k][g_idx]
            data_list_m[base + k][idx_m]["g_sn"] = preds_m[k][g_idx]

    raw_p  = jnp.stack([r[6] for r in buf])
    raw_m  = jnp.stack([r[7] for r in buf])
    psf_raw = jnp.stack([r[8] for r in buf])
    raw_preds_p = np.array(SN_PREDICT(STATE.params, raw_p, psf_raw, OUTPUT_KEYS, GAP))
    raw_preds_m = np.array(SN_PREDICT(STATE.params, raw_m, psf_raw, OUTPUT_KEYS, GAP))
    
    if NORM_PARAMS is not None:
        raw_preds_p = inverse_transform_labels(raw_preds_p, NORM_PARAMS)
        raw_preds_m = inverse_transform_labels(raw_preds_m, NORM_PARAMS)

    g_sn_raw_list_p.append(raw_preds_p[:, g_idx])
    g_sn_raw_list_m.append(raw_preds_m[:, g_idx])
   
    _ok = list(OUTPUT_KEYS)
    if "hlr" in _ok:
        sn_sigma_raw_list.append(raw_preds_p[:, _ok.index("hlr")])
    if "flux" in _ok:
        sn_flux_raw_list.append(raw_preds_p[:, _ok.index("flux")])

hlr_list_out, flux_list_out = [], []
x_im_list, y_im_list = [], []

with Pool(processes=nproc) as pool:
    for result in pool.imap(process_single_object, args_list):
        data_list_p.append(result[0])
        data_list_m.append(result[1])
        gth_list_p.append(result[2])
        gth_list_m.append(result[3])
        hlr_list_out.append(result[9])
        flux_list_out.append(result[10])
        if PSF_FWHM == "superbit":
            x_im_list.append(result[11])
            y_im_list.append(result[12])
        img_buffer.append(result)
        if len(img_buffer) == BATCH_SIZE:
            _run_shearnet_batch(img_buffer)
            img_buffer.clear()
    if img_buffer:
        _run_shearnet_batch(img_buffer)
        img_buffer.clear()

tab_p = shear_data_to_table(data_list_p, mcal_shear=MCAL_STEP)
tab_m = shear_data_to_table(data_list_m, mcal_shear=MCAL_STEP)

tab_p["g_th"] = np.asarray(gth_list_p)
tab_m["g_th"] = np.asarray(gth_list_m)
tab_p["gal_hlr_th"]  = np.array(hlr_list_out)
tab_p["gal_flux_th"] = np.array(flux_list_out)
tab_m["gal_hlr_th"]  = np.array(hlr_list_out)
tab_m["gal_flux_th"] = np.array(flux_list_out)
if PSF_FWHM == "superbit":
    tab_p["psf_x_im"] = np.array(x_im_list)
    tab_p["psf_y_im"] = np.array(y_im_list)
    tab_m["psf_x_im"] = np.array(x_im_list)
    tab_m["psf_y_im"] = np.array(y_im_list)

N = len(tab_p)

# default: NaNs
g_sn_raw_p = np.full((N, 2), np.nan)
g_sn_raw_m = np.full((N, 2), np.nan)

sn_sigma_raw = np.full(N, np.nan)
sn_flux_raw  = np.full(N, np.nan)

if sn_sigma_raw_list:
    sn_sigma_raw[:] = np.concatenate(sn_sigma_raw_list)
if sn_flux_raw_list:
    sn_flux_raw[:]  = np.concatenate(sn_flux_raw_list)

tab_p["g_sn_sigma"] = sn_sigma_raw
tab_p["g_sn_flux"]  = sn_flux_raw
tab_m["g_sn_sigma"] = sn_sigma_raw
tab_m["g_sn_flux"]  = sn_flux_raw

# overwrite if available
if STATE is not None:
    g_sn_raw_p[:] = np.concatenate(g_sn_raw_list_p, axis=0)
    g_sn_raw_m[:] = np.concatenate(g_sn_raw_list_m, axis=0)

# always add columns
tab_p["g_sn_raw"] = g_sn_raw_p
tab_m["g_sn_raw"] = g_sn_raw_m

# ---- skip-deconvolution ("direct") PSF-response correction ----
# Same ensemble-Rbar philosophy as the metacal block below, but Rbar_psf comes
# from the '*_psf_direct' columns (real PSF sheared +/- step, galaxy convolved
# with it -- no deconvolution/dilation). This is the physically-correct PSF
# response for ShearNet. ngmix and ShearNet are corrected independently per the
# config flags. Takes precedence over the metacal psf_response block.
def _rbar_pm(c1p, c1m, step):
    dg = 2.0 * step
    g1p = np.concatenate([np.asarray(tab_p[c1p], float)[:, 0],
                          np.asarray(tab_m[c1p], float)[:, 0]])
    g1m = np.concatenate([np.asarray(tab_p[c1m], float)[:, 0],
                          np.asarray(tab_m[c1m], float)[:, 0]])
    return (np.nanmean(g1p) - np.nanmean(g1m)) / dg

if PSF_RESPONSE_DIRECT and "g_1p_psf_direct" not in tab_p.colnames:
    print("[bias/m] direct PSF-response requested but no *_psf_direct columns "
          "present; skipping correction.")
elif PSF_RESPONSE_DIRECT:
    rbar_psf_direct = (
        _rbar_pm("g_1p_psf_direct", "g_1m_psf_direct", PSF_DIRECT_STEP)
        if PSF_RESPONSE_DIRECT_NGMIX else np.nan
    )
    rbar_psf_sn_direct = (
        _rbar_pm("g_sn_1p_psf_direct", "g_sn_1m_psf_direct", PSF_DIRECT_STEP)
        if (STATE is not None and PSF_RESPONSE_DIRECT_SN) else np.nan
    )
    print(
        f"[bias/m] DIRECT (skip-deconvolution) PSF-response correction ON "
        f"(step={PSF_DIRECT_STEP}; ngmix={PSF_RESPONSE_DIRECT_NGMIX}, "
        f"shearnet={PSF_RESPONSE_DIRECT_SN}); ensemble "
        f"Rbar_psf(ngmix)={rbar_psf_direct:.4f}, "
        f"Rbar_psf(shearnet)={rbar_psf_sn_direct:.4f}. Applying constant Rbar to "
        f"g_noshear / g_sn_noshear; raw kept as *_raw."
    )
    for _tab in (tab_p, tab_m):
        _gpsf = np.asarray(_tab["gpsf_noshear"], dtype=float)   # (N, 2)
        if PSF_RESPONSE_DIRECT_NGMIX:
            _g = np.asarray(_tab["g_noshear"], dtype=float)
            _tab["g_noshear_raw"] = _g.copy()
            _tab["g_noshear"]     = _g - _gpsf * rbar_psf_direct
        if STATE is not None and PSF_RESPONSE_DIRECT_SN:
            _gsn = np.asarray(_tab["g_sn_noshear"], dtype=float)
            _tab["g_sn_noshear_raw"] = _gsn.copy()
            _tab["g_sn_noshear"]     = _gsn - _gpsf * rbar_psf_sn_direct

# ---- PSF-response (metacal leakage) correction ----
# The metacal PSF response is an ENSEMBLE quantity (a ratio of ensemble means,
# exactly as in ngmix/tests/test_metacal_galsim_psf_response.py), NOT a
# per-object finite difference. A per-object response (g_1p_psf-g_1m_psf)/(2*step)
# has huge variance and, since the metacal shear types share each object's noise
# realization, is correlated with g_noshear -- subtracting gpsf*(per-object R)
# therefore injects that variance/noise-bias into the shear and does not cancel
# between the +/- shear pair (so it corrupts m, which should be untouched by an
# additive PSF-leakage term). We instead form a single constant response Rbar
# over the combined p/m ensemble and apply it; with a constant Rbar and the same
# gpsf for the pair, the correction cancels exactly in (g_p - g_m)/2, so it moves
# only the additive bias c, as PSF leakage physically should.
#
# NOTE: this response is only physical for a PSF-deconvolving estimator (ngmix).
# For ShearNet it measures OOD sensitivity to the reconvolution, not leakage, so
# it is applied only when eval.bias.psf_response_shearnet is set.
elif PSF_RESPONSE and "g_1p_psf" not in tab_p.colnames:
    print("[bias/m] PSF-response requested but no *_psf metacal types were run "
          "(reconv_psf must be 'dilate'); skipping correction.")
elif PSF_RESPONSE:
    dg = 2.0 * MCAL_STEP

    def _rbar(c1p, c1m):
        g1p = np.concatenate([np.asarray(tab_p[c1p], float)[:, 0],
                              np.asarray(tab_m[c1p], float)[:, 0]])
        g1m = np.concatenate([np.asarray(tab_p[c1m], float)[:, 0],
                              np.asarray(tab_m[c1m], float)[:, 0]])
        return (np.nanmean(g1p) - np.nanmean(g1m)) / dg

    rbar_psf = _rbar("g_1p_psf", "g_1m_psf")
    rbar_psf_sn = (
        _rbar("g_sn_1p_psf", "g_sn_1m_psf") if STATE is not None else np.nan
    )
    print(
        f"[bias/m] PSF-response correction ON (reconv_psf={RECONV_PSF}, "
        f"metacal_step={MCAL_STEP}); ensemble Rbar_psf(ngmix)={rbar_psf:.4f}, "
        f"Rbar_psf(shearnet)={rbar_psf_sn:.4f}. Applying constant Rbar to "
        f"g_noshear; raw kept as g_noshear_raw."
    )
    if STATE is not None and not PSF_RESPONSE_SHEARNET:
        print("[bias/m] NOTE: ShearNet g_sn_noshear left UNCORRECTED "
              "(metacal response is not physical leakage for a network).")
    for _tab in (tab_p, tab_m):
        _gpsf = np.asarray(_tab["gpsf_noshear"], dtype=float)   # (N, 2)
        _g    = np.asarray(_tab["g_noshear"], dtype=float)      # (N, 2)
        _tab["g_noshear_raw"] = _g.copy()
        _tab["g_noshear"]     = _g - _gpsf * rbar_psf
        if STATE is not None:
            _gsn = np.asarray(_tab["g_sn_noshear"], dtype=float)
            _tab["g_sn_noshear_raw"] = _gsn.copy()
            _tab["g_sn_noshear"] = (
                _gsn - _gpsf * rbar_psf_sn if PSF_RESPONSE_SHEARNET else _gsn
            )

fname = OUTPUT_FITS

# Create directory if needed
outdir = os.path.dirname(fname)
if outdir != "":
    os.makedirs(outdir, exist_ok=True)

hdu_primary = fits.PrimaryHDU()
hdu_p = fits.BinTableHDU(tab_p, name="TAB_P")
hdu_m = fits.BinTableHDU(tab_m, name="TAB_M")

hdul = fits.HDUList([hdu_primary, hdu_p, hdu_m])
hdul.writeto(fname, overwrite=True)


print(f"Saved tables to {fname}")

(
    m_full,
    c_full,
    m,
    merr,
    c,
    cerr,
    m_jk,
    c_jk,
    r11_mean,
    r11_err,
    _,
    _,
    _,
    _,
) = jackknife_mc_v2(tab_p, tab_m, SHEAR_TRUE, njac=NJAC)

exp_m = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
exp_c = int(np.floor(np.log10(abs(c)))) if c != 0 else 0

m_scaled = m / 10**exp_m
merr_scaled = merr / 10**exp_m

c_scaled = c / 10**exp_c
cerr_scaled = cerr / 10**exp_c

print(f"Shear Bias results for {NOBS} objects")
print(f"m = ({m_scaled:.3f} ± {merr_scaled:.3f}) × 10{superscript(exp_m)}")
print(f"c = ({c_scaled:.3f} ± {cerr_scaled:.3f}) × 10{superscript(exp_c)}")

if STATE is not None:
    (
        _,
        _,
        m_sn,
        merr_sn,
        c_sn,
        cerr_sn,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = jackknife_mc_v2(tab_p, tab_m, SHEAR_TRUE, njac=NJAC, g_col="g_sn_noshear", r11_col="r11_sn")

    exp_m_sn = int(np.floor(np.log10(abs(m_sn)))) if m_sn != 0 else 0
    exp_c_sn = int(np.floor(np.log10(abs(c_sn)))) if c_sn != 0 else 0

    print(f"\nShearNet Bias results for {NOBS} objects")
    print(f"m = ({m_sn/10**exp_m_sn:.3f} ± {merr_sn/10**exp_m_sn:.3f}) × 10{superscript(exp_m_sn)}")
    print(f"c = ({c_sn/10**exp_c_sn:.3f} ± {cerr_sn/10**exp_c_sn:.3f}) × 10{superscript(exp_c_sn)}")
