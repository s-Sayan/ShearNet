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

# ----- Catalog -----
COSMOS_CAT_FNAME = _config["catalog"]["cosmos_cat_fname"]
cosmos_cat = Table.read(COSMOS_CAT_FNAME, format="csv")

# ========= Output ============
OUTPUT_FITS = _config["output"]["results_fits"]

NTRY = 20
LM_PARS = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
PSF_LM_PARS = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
EM_PARS={'tol': 1.0e-6, 'maxiter': 50000}

# ========== Load up ShearNet state here =========
# state = 

STATE = None

# ============================================================
# FUNCTIONS THAT NEED galsim + cosmos_cat stay here
# ============================================================
def make_data(rng, noise):

    scale = SCALE
    gal_hlr = GAL_HLR
    gal_flux = GAL_FLUX

    index = rng.randint(len(cosmos_cat))
    phi = cosmos_cat[index]["c10_sersic_fit_phi"] * galsim.radians
    q = cosmos_cat[index]["c10_sersic_fit_q"]
    if q > 1.0:
        q = 1 / q

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
    obj0 = galsim.Exponential(half_light_radius=gal_hlr).shear(q=q, beta=phi)

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

    return obs0, g_th

def process_single_object(args):
    i, base_seed, noise, state = args

    # independent RNG per worker
    rng = np.random.RandomState(base_seed + i)

    obs, g_th= make_data(rng=rng, noise=noise)

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

    data = process_obs(obs, boot)
    return data, g_th


args = [(i, SEED, NOISE, STATE) for i in range(NOBS)]

with Pool(processes=nproc) as pool:
    results = list(pool.imap(process_single_object, args))

data_list = [r[0] for r in results]
gth_list  = [r[1] for r in results] 


tab = shear_data_to_table(data_list)

tab["g_th"] = np.asarray(gth_list)

fname = OUTPUT_FITS

# Create directory if needed
outdir = os.path.dirname(fname)
if outdir != "":
    os.makedirs(outdir, exist_ok=True)

hdu_primary = fits.PrimaryHDU()
hdu = fits.BinTableHDU(tab, name="results")

hdul = fits.HDUList([hdu_primary, hdu])
hdul.writeto(fname, overwrite=True)