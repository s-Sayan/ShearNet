"""
Smoke test for the skip-deconvolution ("direct") PSF-response correction in the
m/bias pipeline. Synthetic galsim + ngmix (no COSMOS/PSFEx/trained model).

Validates:
  A. Direct variants ('1p_psf_direct','1m_psf_direct') fit with a plain (non-
     metacal) bootstrapper flow through make_struct + shear_data_to_table and
     produce finite g_1p_psf_direct / r11_psf_direct columns (and no KeyError on
     'flux').
  B. Correction math: applying a constant ensemble Rbar to g_noshear for both p
     and m with the same per-object gpsf leaves the multiplicative bias m
     UNCHANGED (it cancels in (g_p-g_m)/2) and shifts only the additive bias c --
     exactly the behaviour the direct correction relies on.
"""
import os, sys, types, importlib.util
import numpy as np

HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PKG_DIR = os.path.dirname(os.path.abspath(__file__))

for pkg in ["shearnet", "shearnet.utils"]:
    m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m
spec = importlib.util.spec_from_file_location(
    "shearnet.utils.superbit", os.path.join(HERE, "shearnet/utils/superbit.py"))
_sb = importlib.util.module_from_spec(spec); sys.modules["shearnet.utils.superbit"] = _sb
spec.loader.exec_module(_sb)

sys.path.insert(0, PKG_DIR)
import galsim, ngmix
from helpers import (
    make_struct, process_obs, shear_data_to_table, jackknife_mc_v2,
    _get_priors, get_init_guess,
)

SCALE, NPIX, NPIXP = 0.141, 53, 53
NOISE, PSF_NOISE = 1e-3, 1e-6
STEP = 0.01           # metacal + direct step
SHEAR_TRUE = 0.01
NGAL = 80
LM = {"maxfev": 2000, "xtol": 5e-5, "ftol": 5e-5}
PSF = galsim.Gaussian(half_light_radius=0.5).shear(g1=0.05, g2=-0.03)  # anisotropic
GSP = galsim.GSParams(maximum_fft_size=32768)


def make_boots(rng):
    prior = _get_priors(int(rng.randint(1 << 30)))
    fitter = ngmix.fitting.Fitter(model="gauss", prior=prior, fit_pars=LM)
    psf_fitter = ngmix.fitting.Fitter(model="gauss", fit_pars=LM)
    psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=10)

    def _for(obs):
        TG, _ = get_init_guess(obs)
        guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=TG, prior=prior)
        runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=10)
        mcal = ngmix.metacal.MetacalBootstrapper(
            runner=runner, psf_runner=psf_runner, rng=rng,
            psf="dilate", step=STEP, types=["noshear", "1p", "1m"])
        plain = ngmix.bootstrap.Bootstrapper(runner=runner, psf_runner=psf_runner)
        return mcal, plain
    return _for


def build_tables():
    data_p, data_m = [], []
    for i in range(NGAL):
        rng = np.random.RandomState(500 + i)
        obj0 = galsim.Exponential(half_light_radius=0.5, flux=6000.0).shear(g1=0.0, g2=0.0)
        objp = obj0.shear(g1=SHEAR_TRUE, g2=0.0)
        objm = obj0.shear(g1=-SHEAR_TRUE, g2=0.0)

        psf_im = PSF.drawImage(nx=NPIXP, ny=NPIXP, scale=SCALE).array
        psf_im = psf_im + rng.normal(scale=PSF_NOISE, size=psf_im.shape)
        im_noise = rng.normal(scale=NOISE, size=(NPIX, NPIX))
        cen = (np.array([NPIX, NPIX]) - 1.0) / 2.0
        pcen = (np.array([NPIXP, NPIXP]) - 1.0) / 2.0
        jac = ngmix.DiagonalJacobian(row=cen[0], col=cen[1], scale=SCALE)
        pjac = ngmix.DiagonalJacobian(row=pcen[0], col=pcen[1], scale=SCALE)
        wt = np.zeros((NPIX, NPIX)) + 1.0 / NOISE**2
        pwt = np.zeros((NPIXP, NPIXP)) + 1.0 / PSF_NOISE**2
        psf_obs = ngmix.Observation(psf_im, weight=pwt, jacobian=pjac)

        def obs_of(obj):
            im = galsim.Convolve(PSF, obj, gsparams=GSP).drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array + im_noise
            return ngmix.Observation(im, weight=wt, jacobian=jac, psf=psf_obs)

        obsp, obsm = obs_of(objp), obs_of(objm)

        make = make_boots(rng)
        mcal_p, plain_p = make(obsp)
        mcal_m, plain_m = make(obsm)
        struct_p = process_obs(obsp, mcal_p)   # noshear,1p,1m
        struct_m = process_obs(obsm, mcal_m)

        # direct PSF-shear variants (skip deconvolution)
        for _store, _obj, _plain, _base in ((data_p, objp, plain_p, struct_p),
                                            (data_m, objm, plain_m, struct_m)):
            extra = []
            for key, s in (("1p_psf_direct", +STEP), ("1m_psf_direct", -STEP)):
                psf_s = PSF.shear(g1=s, g2=0.0)
                ps_im = psf_s.drawImage(nx=NPIXP, ny=NPIXP, scale=SCALE).array
                ps_obs = ngmix.Observation(ps_im, weight=pwt, jacobian=pjac)
                im = galsim.Convolve(psf_s, _obj, gsparams=GSP).drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array + im_noise
                obs_d = ngmix.Observation(im, weight=wt, jacobian=jac, psf=ps_obs)
                res = _plain.go(obs_d)
                extra.append(make_struct(res=res, obs=obs_d, shear_type=key))
            _store.append(np.hstack([_base, *extra]))

    tab_p = shear_data_to_table(data_p, mcal_shear=STEP)
    tab_m = shear_data_to_table(data_m, mcal_shear=STEP)
    tab_p["g_th"] = np.tile([SHEAR_TRUE, 0.0], (len(tab_p), 1))
    tab_m["g_th"] = np.tile([-SHEAR_TRUE, 0.0], (len(tab_m), 1))
    return tab_p, tab_m


print("Building synthetic tables (metacal + direct fits)...")
tab_p, tab_m = build_tables()

print("=" * 68)
print("PART A: direct columns present & finite")
print("=" * 68)
colsA = ["g_1p_psf_direct", "g_1m_psf_direct", "r11_psf_direct", "r11"]
for c in colsA:
    print(f"  {c:20s} present={c in tab_p.colnames}")
r11d = np.asarray(tab_p["r11_psf_direct"])
r11  = np.asarray(tab_p["r11"])
print(f"  <r11>            (shear resp) = {np.nanmean(r11):+.4f}   (metacal shear response)")
print(f"  <r11_psf_direct> (psf  resp) = {np.nanmean(r11d):+.4f}   (small, deconvolving)")

print()
print("=" * 68)
print("PART B: correction moves c, leaves m unchanged")
print("=" * 68)
dg = 2.0 * STEP
def _rbar_pm(c1p, c1m):
    g1p = np.concatenate([np.asarray(tab_p[c1p], float)[:, 0], np.asarray(tab_m[c1p], float)[:, 0]])
    g1m = np.concatenate([np.asarray(tab_p[c1m], float)[:, 0], np.asarray(tab_m[c1m], float)[:, 0]])
    return (np.nanmean(g1p) - np.nanmean(g1m)) / dg
rbar = _rbar_pm("g_1p_psf_direct", "g_1m_psf_direct")
print(f"  ensemble Rbar_psf(ngmix, direct) = {rbar:+.5f}")

m0, c0 = jackknife_mc_v2(tab_p, tab_m, SHEAR_TRUE, njac=10)[2:6:2]  # m_mean, c_mean
# apply the constant-Rbar direct correction (same code path as main.py)
for _tab in (tab_p, tab_m):
    _gpsf = np.asarray(_tab["gpsf_noshear"], float)
    _g = np.asarray(_tab["g_noshear"], float)
    _tab["g_noshear_raw"] = _g.copy()
    _tab["g_noshear"] = _g - _gpsf * rbar
m1, c1 = jackknife_mc_v2(tab_p, tab_m, SHEAR_TRUE, njac=10)[2:6:2]

print(f"  m  before = {m0:+.5f}   after = {m1:+.5f}   |dm| = {abs(m1-m0):.2e}  (EXPECT ~0)")
print(f"  c  before = {c0:+.5e}   after = {c1:+.5e}   dc  = {c1-c0:+.2e}  (shifted)")

ok = (
    all(c in tab_p.colnames for c in colsA)
    and np.isfinite(r11d).sum() > 0.8 * NGAL
    and np.isfinite(rbar)
    and abs(m1 - m0) < 1e-6          # m unchanged by constant-Rbar correction
    and abs(c1 - c0) > 0             # c actually moved
)
print()
print("SMOKE TEST:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
