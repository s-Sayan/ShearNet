"""
Smoke test for the skip-deconvolution ("direct") PSF-response path.

Exercises the real helpers (render_psf_shear_variants, process_obs_direct,
leakage_response_to_table) on synthetic galsim data -- no COSMOS catalog, no
PSFEx file, no trained ShearNet needed. It validates:

  1. ngmix path: for a ROUND galaxy with the PSF sheared by +/- step, a
     PSF-deconvolving estimator (ngmix) should recover ~round -> r11_psf ~ 0.
     This confirms the 3-render + finite-difference machinery runs and gives a
     physically sensible number.
  2. ngmix-disabled path: process_obs_direct(do_ngmix=False) returns NaN ngmix
     shapes but valid gpsf + the 3 (galaxy, psf) image pairs ShearNet would eat.
"""
import os, sys, types, importlib.util
import numpy as np

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# --- preload shearnet.utils.superbit WITHOUT importing the heavy shearnet pkg ---
for pkg in ["shearnet", "shearnet.utils"]:
    m = types.ModuleType(pkg)
    m.__path__ = []
    sys.modules[pkg] = m
spec = importlib.util.spec_from_file_location(
    "shearnet.utils.superbit", os.path.join(HERE, "shearnet/utils/superbit.py")
)
_superbit = importlib.util.module_from_spec(spec)
sys.modules["shearnet.utils.superbit"] = _superbit
spec.loader.exec_module(_superbit)

sys.path.insert(0, PKG_DIR)
import galsim
import ngmix
from helpers import (
    render_psf_shear_variants,
    process_obs_direct,
    leakage_response_direct_to_table,
    _get_priors,
    get_init_guess,
)

SCALE = 0.141
NPIX = 53
NPIX_PSF = 53
NOISE = 1e-4          # tiny noise -> clean response
STEP = 0.01
NGAL = 150
LM_PARS = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}


def build_boot(rng):
    prior = _get_priors(int(rng.randint(1 << 30)))
    fitter = ngmix.fitting.Fitter(model="gauss", prior=prior, fit_pars=LM_PARS)

    def _make(obs):
        TGUESS, _ = get_init_guess(obs)
        guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=TGUESS, prior=prior)
        psf_fitter = ngmix.fitting.Fitter(model="gauss", fit_pars=LM_PARS)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
        psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=10)
        runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=10)
        return ngmix.bootstrap.Bootstrapper(runner=runner, psf_runner=psf_runner)

    return _make


def run(do_ngmix, psf, gal_factory):
    data_list = []
    images_last = None
    for i in range(NGAL):
        rng = np.random.RandomState(1000 + i)
        obj0 = gal_factory()
        PSF = psf
        obs_dict = render_psf_shear_variants(
            obj0, PSF, step=STEP, noise=NOISE, scale=SCALE,
            npix=NPIX, npix_psf=NPIX_PSF, rng=rng,
        )
        boot = build_boot(rng)(obs_dict["noshear"]) if do_ngmix else None
        struct, images = process_obs_direct(obs_dict, boot=boot, do_ngmix=do_ngmix)
        data_list.append(struct)
        images_last = images

    tab, rbar_psf, rbar_psf_sn = leakage_response_direct_to_table(data_list, step=STEP)
    return tab, rbar_psf, rbar_psf_sn, images_last, data_list


EPSF = galsim.Gaussian(half_light_radius=0.5).shear(g1=0.03, g2=-0.02)  # anisotropic PSF

print("=" * 70)
print("CASE 1a: MATCHED models (round Gaussian gal + Gaussian PSF + gauss fit)")
print("         -> a perfect deconvolving estimator must give r11_psf = 0")
print("=" * 70)
tab, rbar_psf, _, _, _ = run(
    do_ngmix=True, psf=EPSF,
    gal_factory=lambda: galsim.Gaussian(half_light_radius=0.5, flux=4000.0),
)
r11 = np.asarray(tab["r11_psf"])
print(f"n galaxies fit           : {np.isfinite(r11).sum()} / {len(tab)}")
print(f"ensemble Rbar_psf(ngmix) : {rbar_psf:+.5f}   (EXPECT ~ 0.000)")

print()
print("=" * 70)
print("CASE 1b: MISMATCHED galaxy profile (Exponential gal, gauss fit)")
print("         -> small nonzero leakage (deconvolving regime, |R| << 1)")
print("=" * 70)
tabm, rbar_m, _, _, _ = run(
    do_ngmix=True, psf=EPSF,
    gal_factory=lambda: galsim.Exponential(half_light_radius=0.5, flux=4000.0),
)
print(f"ensemble Rbar_psf(ngmix) : {rbar_m:+.5f}   (small, ~ metacal ngmix value)")

print()
print("=" * 70)
print("CASE 2: ngmix DISABLED (ShearNet-only plumbing check)")
print("=" * 70)
tab2, rbar2, rbar2_sn, images, dlist2 = run(
    do_ngmix=False, psf=EPSF,
    gal_factory=lambda: galsim.Exponential(half_light_radius=0.5, flux=4000.0),
)
print(f"r11_psf all NaN (ngmix off)    : {np.all(np.isnan(np.asarray(tab2['r11_psf'])))}")
print(f"gpsf still populated           : {np.isfinite(np.asarray(tab2['gpsf'])).all()}")
print(f"image dict keys                : {list(images.keys())}")
print(f"galaxy stamp / psf stamp shapes: {images['1p_psf'][0].shape} / {images['1p_psf'][1].shape}")
# confirm the +g1 and -g1 PSF stamps actually differ (i.e. we really sheared it)
dpsf = np.abs(images['1p_psf'][1] - images['1m_psf'][1]).max()
dgal = np.abs(images['1p_psf'][0] - images['1m_psf'][0]).max()
print(f"max|psf(+g1) - psf(-g1)|        : {dpsf:.3e}  (must be > 0)")
print(f"max|gal(+g1) - gal(-g1)|        : {dgal:.3e}  (must be > 0)")

print()
print("=" * 70)
print("CASE 3: ShearNet-ONLY (ngmix off) -> rbar_psf_sn must be FINITE")
print("        (regression guard for the flags-coupling bug in the reused")
print("         metacal assembler; here ngmix flags=1 for every object)")
print("=" * 70)
# emulate a ShearNet-only run: ngmix disabled (flags=1, g=NaN), but inject a
# synthetic per-type g_sn with a known r11_psf_sn = 0.5 so we can check both the
# per-object value and the ensemble come out finite and correct.
KNOWN_R = 0.5
for arr in dlist2:
    for rec in arr:
        st = str(rec["shear_type"])
        if st == "1p_psf":
            rec["g_sn"] = np.array([+KNOWN_R * STEP, 0.0])
        elif st == "1m_psf":
            rec["g_sn"] = np.array([-KNOWN_R * STEP, 0.0])
        else:
            rec["g_sn"] = np.array([0.0, 0.0])
tab3, rbar3, rbar3_sn = leakage_response_direct_to_table(dlist2, step=STEP)
r11sn = np.asarray(tab3["r11_psf_sn"])
print(f"r11_psf (ngmix)  all NaN : {np.all(np.isnan(np.asarray(tab3['r11_psf'])))}  (ngmix off)")
print(f"rbar_psf_sn              : {rbar3_sn:+.5f}   (EXPECT {KNOWN_R:+.5f}, must be finite)")
print(f"<r11_psf_sn> per-object  : {np.nanmean(r11sn):+.5f}   (EXPECT {KNOWN_R:+.5f})")

ok = (
    abs(rbar_psf) < 0.01                                  # matched models -> ~0 response
    and np.isfinite(r11).sum() > 0.8 * NGAL               # fits converge
    and abs(rbar_m) < 0.3                                 # mismatch: small, deconvolving regime
    and np.all(np.isnan(np.asarray(tab2["r11_psf"])))     # ngmix-off -> NaN
    and np.isfinite(np.asarray(tab2["gpsf"])).all()       # gpsf still populated
    and dpsf > 0 and dgal > 0                             # PSF really sheared +/-
    and np.isfinite(rbar3_sn) and abs(rbar3_sn - KNOWN_R) < 1e-6   # SN ensemble finite & correct w/ ngmix off
    and np.all(np.isnan(np.asarray(tab3["r11_psf"])))     # ngmix column still NaN
)
print()
print("SMOKE TEST:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
