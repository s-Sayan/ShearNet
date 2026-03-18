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
import superbit_lensing.utils as utils
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

# ----- Simulation controls -----
SEED = _config["simulation"]["seed"]
SHEAR_TRUE = _config["simulation"]["shear_true"]

NOISE = _config["simulation"]["nse_sd"]
PSF_NOISE = _config["simulation"]["psf_noise"]
SCALE = _config["simulation"]["scale"]
NPIX = _config["simulation"]["npix"]

GAL_HLR = _config["simulation"]["hlr"]
GAL_FLUX = _config["simulation"]["flux"]

PSF_FWHM = _config["simulation"]["psf_fwhm"]
PSF_NPIX = _config["simulation"]["psf_npix"]

NOBS = _config["simulation"]["n_obs"]
NJAC = _config["simulation"]["Njack"]

GALSIM_PSF = galsim.des.DES_PSFEx(_config["simulation"]["psfex_model_file"], wcs=utils.get_galsim_tanwcs())

# ----- Models -----
PSF_MODEL = _config["models"]["psf_model"]
GAL_MODEL = _config["models"]["gal_model"]

# ShearNet
INCLUDE_SN = _config["ShearNet"]["include_sn"]
SN_MODEL_NAME = _config["ShearNet"]["sn_model_name"]
N_OUTPUTS = _config["ShearNet"]["n_outputs"]

# ----- Catalog -----
COSMOS_CAT_FNAME = _config["catalog"]["cosmos_cat_fname"]
with fits.open(COSMOS_CAT_FNAME) as hdul:
    cosmos_cat = hdul[1].data

# ========= Output ============
OUTPUT_FITS = _config["output"]["results_fits"]

TGUESS = 4 * SCALE**2
NTRY = 20
LM_PARS = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
PSF_LM_PARS = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
EM_PARS={'tol': 1.0e-6, 'maxiter': 50000}
MCAL_PARS = {"psf": "dilate", "mcal_shear": 0.01}
TYPES = ["noshear", "1p", "1m"]

# ========== Load up ShearNet state here =========
import jax.numpy as jnp
if INCLUDE_SN:
    from shearnet.cli.evaluate import initialize_model as _initialize_model, load_config as _eval_load_config
    eval_args = argparse.Namespace(
        model_name=SN_MODEL_NAME, config=None, seed=None,
        test_samples=None, mcal=False, plot=False, plot_animation=False,
        process_psf=None, galaxy_type=None, psf_type=None,
        apply_psf_shear=None, psf_shear_range=None
    )
    eval_config = _eval_load_config(eval_args)
    dummy_images = jnp.ones((1, NPIX, NPIX))
    STATE = _initialize_model(eval_config, dummy_images, dummy_images)
else:
    STATE = None

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
        psf = galsim.Gaussian(fwhm=psf_fwhm)

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

    psf_im = psf.drawImage(nx=npix_psf, ny=npix_psf, scale=scale).array
    im_0 = obj0_psf.drawImage(nx=npix, ny=npix, scale=scale).array
    im_p = objp_psf.drawImage(nx=npix, ny=npix, scale=scale).array
    im_m = objm_psf.drawImage(nx=npix, ny=npix, scale=scale).array

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

    if psf_fwhm == "superbit":
        return obs0, obsp, obsm, g_th_p, g_th_m, gal_hlr, gal_flux, x_im, y_im
    else:
        return obs0, obsp, obsm, g_th_p, g_th_m, gal_hlr, gal_flux

def process_single_object(args):
    i, base_seed, noise, shear_true = args

    # independent RNG per worker
    rng = np.random.RandomState(base_seed + i)

    if PSF_FWHM == "superbit":
        obs0, obsp, obsm, g_th_p, g_th_m, gal_hlr, gal_flux, x_im, y_im = make_data(rng=rng, noise=noise, shear_true=shear_true)
    else:
        obs0, obsp, obsm, g_th_p, g_th_m, gal_hlr, gal_flux = make_data(rng=rng, noise=noise, shear_true=shear_true)

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

        preds_p = np.array(STATE.apply_fn(STATE.params, gal_p, psf_p, n_outputs=N_OUTPUTS, deterministic=True))
        preds_m = np.array(STATE.apply_fn(STATE.params, gal_m, psf_m, n_outputs=N_OUTPUTS, deterministic=True))

        for k in range(n):
            idx_p = np.where(data_list_p[base + k]["shear_type"] == stype)[0][0]
            idx_m = np.where(data_list_m[base + k]["shear_type"] == stype)[0][0]
            data_list_p[base + k][idx_p]["g_sn"] = preds_p[k][:2]
            data_list_m[base + k][idx_m]["g_sn"] = preds_m[k][:2]

    raw_p  = jnp.stack([r[6] for r in buf])
    raw_m  = jnp.stack([r[7] for r in buf])
    psf_raw = jnp.stack([r[8] for r in buf])
    raw_preds_p = np.array(STATE.apply_fn(STATE.params, raw_p, psf_raw, n_outputs=N_OUTPUTS, deterministic=True))
    raw_preds_m = np.array(STATE.apply_fn(STATE.params, raw_m, psf_raw, n_outputs=N_OUTPUTS, deterministic=True))

    g_sn_raw_list_p.append(raw_preds_p[:, :2])
    g_sn_raw_list_m.append(raw_preds_m[:, :2])

    if N_OUTPUTS > 2:
        sn_sigma_raw_list.append(raw_preds_p[:, 2])
    if N_OUTPUTS > 3:
        sn_flux_raw_list.append(raw_preds_p[:, 3])

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

tab_p = shear_data_to_table(data_list_p)
tab_m = shear_data_to_table(data_list_m)

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
