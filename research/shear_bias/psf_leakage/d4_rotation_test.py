import numpy as np
import jax
import jax.numpy as jnp
import galsim
import galsim.des
import ngmix
import os
from astropy.io import fits
import superbit_lensing.utils as utils
from helpers import (
    _get_priors, make_struct, process_obs, shear_data_to_table,
    get_init_guess, get_em_ngauss, get_coellip_ngauss
)
import yaml
import argparse

# --- 1. GLOBALS & CONFIG (Copied from main.py) ---
config_path = "config_full.yaml"
with open(config_path, "r") as f:
    _config = yaml.safe_load(f)

SEED  = _config["eval"]["seed"]
NOISE = _config["image"]["noise_sd"]
SCALE = _config["image"]["pixel_scale"]
NPIX  = _config["image"]["stamp_size"]
PSF_NPIX = _config["psf"]["stamp_size"]
GALSIM_PSF = galsim.des.DES_PSFEx(_config["paths"]["psfex_model_file"], wcs=utils.get_galsim_tanwcs())

GAL_HLR  = _config["galaxy"]["hlr"]  if _config["galaxy"]["hlr_type"]  == "constant" else "catalog"
GAL_FLUX = _config["galaxy"]["flux"] if _config["galaxy"]["flux_type"] == "constant" else "catalog"

PSF_MODEL = _config["eval"]["leakage"]["psf_model"]
GAL_MODEL = _config["eval"]["gal_model"]
INCLUDE_SN = _config["eval"]["include_shearnet"]
SN_MODEL_NAME = _config["meta"]["model_name"]
OUTPUT_KEYS = tuple(_config["model"]["output_keys"])
GAP = _config["model"].get("gap", False)

COSMOS_CAT_FNAME = _config["paths"]["eval_catalog"]
with fits.open(COSMOS_CAT_FNAME) as hdul:
    cosmos_cat = hdul[1].data

NTRY = 20
LM_PARS = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
PSF_LM_PARS = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
EM_PARS={'tol': 1.0e-6, 'maxiter': 50000}

# --- 2. MODEL INITIALIZATION (Copied from main.py) ---
if INCLUDE_SN:
    from shearnet.cli.evaluate import load_model as _initialize_model, load_config as _eval_load_config
    eval_config = argparse.Namespace(
        model_name=SN_MODEL_NAME, config=None, seed=None,
        test_samples=None, mcal=False, plot=False, plot_animation=False,
        process_psf=None, galaxy_type=None, psf_type=None,
        apply_psf_shear=None, psf_shear_range=None
    )
    eval_config = _eval_load_config(eval_config)
    dummy_images = jnp.ones((1, NPIX, NPIX))
    STATE = _initialize_model(eval_config, dummy_images, dummy_images)
    
    def _sn_predict(params, gal, psf, output_keys, gap):
         return STATE.apply_fn(params, gal, psf, output_keys=output_keys, gap=gap, deterministic=True)
    SN_PREDICT = jax.jit(_sn_predict, static_argnames=("output_keys", "gap"))
else:
    STATE, SN_PREDICT = None, None

# --- 3. FUNCTIONS (Copied from main.py) ---
def make_data(rng, noise):
    scale = SCALE
    index = rng.randint(len(cosmos_cat))
    q    = cosmos_cat['Q'][index]
    phi  = cosmos_cat['PHI'][index] * galsim.radians
    gal_hlr  = cosmos_cat['HLR'][index] if GAL_HLR == "catalog" else GAL_HLR
    gal_flux = cosmos_cat['FLUX'][index] if GAL_FLUX == "catalog" else GAL_FLUX
    
    npix_psf, npix = PSF_NPIX, NPIX
    dy, dx = rng.uniform(low=-scale / 2, high=scale / 2, size=2)
    
    x_im = rng.randint(500, 9600 - 500)
    y_im = rng.randint(500, 6400 - 500)
    psf = GALSIM_PSF.getPSF(galsim.PositionD(x_im, y_im))
    
    obj0 = galsim.Exponential(half_light_radius=gal_hlr, flux=gal_flux).shear(q=q, beta=phi)
    g1_th, g2_th, _, _, _ = utils.g_from_gal_jac(obj0)
    obj0 = obj0.shift(dx=dx, dy=dy)
    obj0_psf = galsim.Convolve(psf, obj0, gsparams=galsim.GSParams(maximum_fft_size=32768))

    psf_im = psf.withGSParams(gsp).drawImage(nx=npix_psf, ny=npix_psf, scale=scale).array
    im_0   = obj0_psf.withGSParams(gsp).drawImage(nx=npix, ny=npix, scale=scale).array
    im_0  += rng.normal(scale=noise, size=im_0.shape)
    
    cen = (np.array(im_0.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0
    jacobian = ngmix.DiagonalJacobian(row=cen[0]+dy/scale, col=cen[1]+dx/scale, scale=scale)
    psf_jacobian = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)
    
    obs0 = ngmix.Observation(im_0, weight=im_0*0+1.0/noise**2, jacobian=jacobian, 
                             psf=ngmix.Observation(psf_im, weight=psf_im*0+1.0/(psf_im.max()/1000.0)**2, jacobian=psf_jacobian))
    return obs0, np.array([g1_th, g2_th])

def process_single_object(args):
    i, base_seed, noise = args
    rng = np.random.RandomState(base_seed + i)
    obs, g_th = make_data(rng=rng, noise=noise)
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
    return data, g_th, gal_im, psf_im

# --- 4. D4 ORBIT LOGIC ---
def apply_op(A, k, flip):
    B = np.flip(A, axis=1) if flip else A
    return np.rot90(B, k, axes=(1, 2))

def Dmap(eg, k, flip):
    sk = (-1)**k
    e1 = sk * eg[:, 0]
    e2 = sk * (-1 if flip else 1) * eg[:, 1]
    return np.stack([e1, e2], 1)

def run_orbit(gals, psfs):
    def estimator(g, p):
        preds = np.array(SN_PREDICT(STATE.params, jnp.array(g), jnp.array(p), OUTPUT_KEYS, GAP))
        g_idx = [OUTPUT_KEYS.index("g1"), OUTPUT_KEYS.index("g2")]
        return preds[:, g_idx]
    
    base = estimator(gals, psfs)
    res = {}
    for (k, f) in [(k, f) for f in (False, True) for k in range(4)]:
        if k == 0 and not f: continue
        res[(k, f)] = estimator(apply_op(gals, k, f), apply_op(psfs, k, f)) - Dmap(base, k, f)
    return res

# --- 5. EXECUTION ---
if __name__ == "__main__":
    n_samples = 10000
    # Fixed the unpacking error: (i, SEED, NOISE)
    results = [process_single_object((i, SEED, NOISE)) for i in range(n_samples)]
    gals = np.stack([r[2] for r in results])
    psfs = np.stack([r[3] for r in results])
    
    allres = np.concatenate(list(run_orbit(gals, psfs).values()), 0)
    print(f"\nRMS e1: {np.sqrt((allres[:, 0]**2).mean()):.4e}")
    print(f"RMS e2: {np.sqrt((allres[:, 1]**2).mean()):.4e}")