# main.py or notebook cell

# ============================================================
# CONFIGURATION
# ============================================================
import numpy as np
import ngmix
import galsim
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

# ----- Models -----
PSF_MODEL = _config["models"]["psf_model"]
GAL_MODEL = _config["models"]["gal_model"]

# ----- Catalog -----
COSMOS_CAT_FNAME = _config["catalog"]["cosmos_cat_fname"]
cosmos_cat = Table.read(COSMOS_CAT_FNAME, format="csv")

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
# state = 

STATE = None

# ============================================================
# FUNCTIONS THAT NEED galsim + cosmos_cat stay here
# ============================================================
def make_data(rng, noise, shear_true):
    psf_noise = PSF_NOISE

    scale = SCALE
    psf_fwhm = PSF_FWHM
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

    psf = galsim.Gaussian(fwhm=psf_fwhm)

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

    obj0_psf = galsim.Convolve(psf, obj0)
    objp_psf = galsim.Convolve(psf, objp)
    objm_psf = galsim.Convolve(psf, objm)

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

    return obs0, obsp, obsm, g_th_p, g_th_m

def process_single_object(args):
    i, base_seed, noise, shear_true, state = args

    # independent RNG per worker
    rng = np.random.RandomState(base_seed + i)

    obs0, obsp, obsm, g_th_p, g_th_m = make_data(rng=rng, noise=noise, shear_true=shear_true)

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


    if state is not None:
        data_p, g_sn_raw_p = process_obs(obsp, boot, state=state)
        data_m, g_sn_raw_m = process_obs(obsm, boot, state=state)
        return data_p, data_m, g_th_p, g_th_m, g_sn_raw_p, g_sn_raw_m,
    else:
        data_p = process_obs(obsp, boot, state=state)
        data_m = process_obs(obsm, boot, state=state)
        return data_p, data_m, g_th_p, g_th_m


args = [(i, SEED, NOISE, SHEAR_TRUE, STATE) for i in range(NOBS)]

with Pool(processes=nproc) as pool:
    results = list(pool.imap(process_single_object, args))

data_list_p = [r[0] for r in results]
data_list_m = [r[1] for r in results]
gth_list_p  = [r[2] for r in results] 
gth_list_m  = [r[3] for r in results] 
if STATE is not None:
    g_sn_raw_list_p = [r[4] for r in results]
    g_sn_raw_list_m = [r[5] for r in results]

tab_p = shear_data_to_table(data_list_p)
tab_m = shear_data_to_table(data_list_m)

tab_p["g_th"] = np.asarray(gth_list_p)
tab_m["g_th"] = np.asarray(gth_list_m)

N = len(tab_p)

# default: NaNs
g_sn_raw_p = np.full((N, 2), np.nan)
g_sn_raw_m = np.full((N, 2), np.nan)

# overwrite if available
if STATE is not None:
    g_sn_raw_p[:] = np.asarray(g_sn_raw_list_p)
    g_sn_raw_m[:] = np.asarray(g_sn_raw_list_m)

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
) = jackknife_mc_v2(tab_p, tab_m, shear_true, njac=NJAC)

exp_m = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
exp_c = int(np.floor(np.log10(abs(c)))) if c != 0 else 0

m_scaled = m / 10**exp_m
merr_scaled = merr / 10**exp_m

c_scaled = c / 10**exp_c
cerr_scaled = cerr / 10**exp_c

print(f"Shear Bias results for {NOBS} objects")
print(f"m = ({m_scaled:.3f} ± {merr_scaled:.3f}) × 10{superscript(exp_m)}")
print(f"c = ({c_scaled:.3f} ± {cerr_scaled:.3f}) × 10{superscript(exp_c)}")
