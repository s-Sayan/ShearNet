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

# ----- Simulation controls -----
SEED = _config["simulation"]["seed"]

NOISE = _config["simulation"]["nse_sd"]
SCALE = _config["simulation"]["scale"]
NPIX = _config["simulation"]["npix"]
PSF_NPIX = _config["simulation"]["psf_npix"]
GALSIM_PSF = galsim.des.DES_PSFEx(_config["simulation"]["psfex_model_file"], wcs=utils.get_galsim_tanwcs())

GAL_HLR = _config["simulation"]["hlr"]
GAL_FLUX = _config["simulation"]["flux"]

NOBS = _config["simulation"]["n_obs"]

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

NTRY = 20
LM_PARS = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
PSF_LM_PARS = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
EM_PARS={'tol': 1.0e-6, 'maxiter': 50000}

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

    psf_im = psf.drawImage(nx=npix_psf, ny=npix_psf, scale=scale).array
    im_0 = obj0_psf.drawImage(nx=npix, ny=npix, scale=scale).array

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


args_list = [(i, SEED, NOISE) for i in range(NOBS)]

data_list = []
gth_list = []
g_sn_raw_list = []
img_buffer = []

BATCH_SIZE = 500

def _run_shearnet_batch(buf):
    if STATE is None:
        return
    n = len(buf)
    base = len(data_list) - n
    gal = jnp.stack([r[2] for r in buf])
    psf = jnp.stack([r[3] for r in buf])
    preds = np.array(STATE.apply_fn(STATE.params, gal, psf, n_outputs=N_OUTPUTS, deterministic=True))
    for k in range(n):
        data_list[base + k][0]["g_sn"] = preds[k][:2]
    g_sn_raw_list.append(preds[:, :2])

hlr_list_out, flux_list_out = [], []

with Pool(processes=nproc) as pool:
    for result in pool.imap(process_single_object, args_list):
        data_list.append(result[0])
        gth_list.append(result[1])
        hlr_list_out.append(result[4])
        flux_list_out.append(result[5])
        img_buffer.append(result)
        if len(img_buffer) == BATCH_SIZE:
            _run_shearnet_batch(img_buffer)
            img_buffer.clear()
    if img_buffer:
        _run_shearnet_batch(img_buffer)
        img_buffer.clear()

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
