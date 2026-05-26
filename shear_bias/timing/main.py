"""
ShearNet/time_analysis/main.py

Two timing modes:

  Fair (default)
    Single CPU for ngmix (sequential, no Pool), batch=1 for ShearNet.
    Isolates per-galaxy latency on equal hardware footing.

  Realistic (--realistic)
    ngmix gets every CPU SLURM allocated (multiprocessing Pool).
    ShearNet gets the full GPU with large batches.
    Measures effective throughput — galaxies per second — for each
    method operating at its natural optimum, which is what actually
    matters for a production weak-lensing pipeline.
    Per-galaxy time is reported as total_wall / n_obs.

Benchmarking notes
------------------
* time.perf_counter() — highest-resolution monotonic timer in Python.
* jax.block_until_ready() — flushes JAX async GPU dispatch before
  stopping the clock; without this the measured ShearNet time is
  meaninglessly short.
* Warm-up phase discards JIT-compilation overhead for JAX and any
  one-time initialisation in ngmix.
* Realistic mode generates all observations in the main process first
  (not timed) so we benchmark fitting, not simulation.
"""

import os
import time
import argparse
from multiprocessing import Pool

import numpy as np
import yaml
import ngmix
import galsim
import galsim.des
import matplotlib.pyplot as plt
from astropy.io import fits

import superbit_lensing.utils as utils


# ============================================================
# HELPERS  (inlined — no helpers.py dependency)
# ============================================================

def _get_priors(seed):
    rng = np.random.RandomState(seed)
    return ngmix.joint_prior.PriorSimpleSep(
        ngmix.priors.CenPrior(0.0, 0.0, 0.2, 0.2, rng=rng),
        ngmix.priors.GPriorBA(0.3, rng=rng),
        ngmix.priors.FlatPrior(-1.0, 1000, rng=rng),
        ngmix.priors.FlatPrior(-1.0e1, 1.0e5, rng=rng),
    )


def _get_init_guess(obs):
    gm = ngmix.gaussmom.GaussMom(1.2).go(obs)
    if gm["flags"] == 0:
        return gm["T"], gm["flux"]
    gm = ngmix.gaussmom.GaussMom(1.2).go(obs.psf)
    T  = 2 * gm["T"] if gm["flags"] == 0 else 4 * obs._jacobian.get_scale() ** 2
    return T, np.sum(obs.image)


def _get_em_ngauss(name):
    return int(name[2:])


def _get_coellip_ngauss(name):
    return int(name[7:])


def _make_struct(res, obs):
    dt = [
        ("flags", "i4"), ("s2n", "f8"), ("g", "f8", 2),
        ("T", "f8"), ("Tpsf", "f8"), ("gpsf", "f8", 2),
    ]
    data = np.zeros(1, dtype=dt)
    data["flags"] = res["flags"]
    if res["flags"] == 0:
        data["s2n"] = res["s2n"]
        data["g"]   = res.get("e", res.get("g", np.nan))
        data["T"]   = res["T"]
    else:
        data["s2n"] = data["g"] = data["T"] = data["Tpsf"] = np.nan
    admom = utils.get_admoms_ngmix_fit(obs.psf, reduced=True)
    if admom["flags"] == 0:
        data["gpsf"] = [admom["e1"], admom["e2"]]
        data["Tpsf"] = admom["T"]
    else:
        data["gpsf"] = data["Tpsf"] = np.nan
    return data


# ============================================================
# CONFIG / ARGS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Timing benchmark: ngmix vs ShearNet"
    )
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument(
        "--realistic",
        action="store_true",
        help=(
            "Best-of-both benchmark: ngmix runs with a full multiprocessing Pool "
            "(all SLURM CPUs), ShearNet runs with large GPU batches. "
            "Reports effective per-galaxy throughput for each method at its optimum."
        ),
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


args    = parse_args()
_config = load_config(args.config)
REALISTIC = args.realistic

SEED       = _config["simulation"]["seed"]
NOISE      = _config["simulation"]["nse_sd"]
SCALE      = _config["simulation"]["scale"]
NPIX       = _config["simulation"]["npix"]
PSF_NPIX   = _config["simulation"]["psf_npix"]
GAL_HLR    = _config["simulation"]["hlr"]
GAL_FLUX   = _config["simulation"]["flux"]
NOBS       = _config["simulation"]["n_obs"]
N_WARMUP   = _config["simulation"]["n_warmup"]
PSFEX_FILE = _config["simulation"]["psfex_model_file"]

PSF_MODEL  = _config["models"]["psf_model"]
GAL_MODEL  = _config["models"]["gal_model"]

INCLUDE_SN    = _config["ShearNet"]["include_sn"]
SN_MODEL_NAME = _config["ShearNet"]["sn_model_name"]
OUTPUT_KEYS   = tuple(_config["ShearNet"]["output_keys"])
GAP           = _config["ShearNet"].get("gap", False)

COSMOS_CAT_FNAME = _config["catalog"]["cosmos_cat_fname"]
OUTPUT_NPZ       = _config["output"]["results_npz"]

# Realistic-mode settings (with safe defaults if section absent)
_real_cfg = _config.get("realistic", {})
SN_BATCH_SIZE = _real_cfg.get("shearnet_batch_size", 512)
_cfg_nproc    = _real_cfg.get("nproc", None)
# honour explicit config value; otherwise use SLURM allocation; fallback to 8
NPROC = (
    int(_cfg_nproc)
    if _cfg_nproc is not None
    else int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
)

NTRY        = 20
LM_PARS     = {"maxfev": 2000, "xtol": 5.0e-5, "ftol": 5.0e-5}
PSF_LM_PARS = {"maxfev": 4000, "xtol": 5.0e-5, "ftol": 5.0e-5}
EM_PARS     = {"tol": 1.0e-6, "maxiter": 50000}

GALSIM_PSF = galsim.des.DES_PSFEx(PSFEX_FILE, wcs=utils.get_galsim_tanwcs())

with fits.open(COSMOS_CAT_FNAME) as hdul:
    cosmos_cat = hdul[1].data


# ============================================================
# SHEARNET  — deferred initialisation
#
# JAX starts background threads on import.  If those threads are alive
# when multiprocessing.Pool forks, each worker inherits the full JAX
# memory footprint, triggering the os.fork() deadlock warning and
# exhausting RAM with 32 workers.
#
# Fix: don't import JAX or initialise STATE until after the Pool has
# been created and closed.  _init_shearnet() is called at the top of
# run_fair() (no Pool involved) and after pool.map() in run_realistic().
# ============================================================

STATE = NORM_PARAMS = None


def _init_shearnet():
    """Import JAX and load ShearNet weights into global STATE/NORM_PARAMS."""
    global STATE, NORM_PARAMS
    import jax.numpy as jnp  # first JAX import — starts threads here
    if not INCLUDE_SN:
        return
    from shearnet.cli.evaluate import (
        load_model as _initialize_model,
        load_config as _eval_load_config,
    )
    from shearnet.utils.normalization import load_normalizer

    eval_cfg = argparse.Namespace(
        model_name=SN_MODEL_NAME, config=None, seed=None,
        test_samples=None, mcal=False, plot=False, plot_animation=False,
        process_psf=None, galaxy_type=None, psf_type=None,
        apply_psf_shear=None, psf_shear_range=None,
    )
    eval_cfg    = _eval_load_config(eval_cfg)
    dummy       = jnp.ones((1, NPIX, NPIX))
    STATE       = _initialize_model(eval_cfg, dummy, dummy)
    data_path   = os.getenv("SHEARNET_DATA_PATH", os.path.abspath("."))
    norm_path   = os.path.join(data_path, "plots", SN_MODEL_NAME, "label_normalizer.npz")
    NORM_PARAMS = load_normalizer(norm_path) if os.path.exists(norm_path) else None


# ============================================================
# SIMULATION
# ============================================================

def make_data(rng):
    scale    = SCALE
    index    = rng.randint(len(cosmos_cat))
    q        = cosmos_cat["Q"][index]
    phi      = cosmos_cat["PHI"][index] * galsim.radians
    gal_hlr  = GAL_HLR  if GAL_HLR  != "catalog" else cosmos_cat["HLR"][index]
    gal_flux = GAL_FLUX if GAL_FLUX != "catalog" else cosmos_cat["FLUX"][index]

    dy, dx = rng.uniform(low=-scale / 2, high=scale / 2, size=2)
    x_im = rng.randint(500, 9100)
    y_im = rng.randint(500, 5900)
    psf  = GALSIM_PSF.getPSF(galsim.PositionD(x_im, y_im))

    gsp  = galsim.GSParams(maximum_fft_size=32768)
    obj0 = galsim.Exponential(half_light_radius=gal_hlr, flux=gal_flux).shear(q=q, beta=phi)
    obj0 = obj0.shift(dx=dx, dy=dy)

    psf_im = psf.drawImage(nx=PSF_NPIX, ny=PSF_NPIX, scale=scale).array
    im_0   = galsim.Convolve(psf, obj0, gsparams=gsp).drawImage(nx=NPIX, ny=NPIX, scale=scale).array
    im_0  += rng.normal(scale=NOISE, size=im_0.shape)

    cen     = (np.array(im_0.shape)   - 1.0) / 2.0
    psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0
    jac     = ngmix.DiagonalJacobian(row=cen[0] + dy / scale, col=cen[1] + dx / scale, scale=scale)
    psf_jac = ngmix.DiagonalJacobian(row=psf_cen[0], col=psf_cen[1], scale=scale)

    psf_noise_val = psf_im.max() / 1000.0
    psf_obs = ngmix.Observation(psf_im, weight=psf_im * 0 + 1.0 / psf_noise_val ** 2, jacobian=psf_jac)
    obs     = ngmix.Observation(im_0,   weight=im_0  * 0 + 1.0 / NOISE ** 2,           jacobian=jac, psf=psf_obs)
    return obs


# ============================================================
# PER-GALAXY RUNNERS  (used in both modes)
# ============================================================

def run_ngmix(obs, seed):
    prior   = _get_priors(seed)
    rng     = np.random.RandomState(seed)
    Tguess, _ = _get_init_guess(obs)

    fitter  = ngmix.fitting.Fitter(model=GAL_MODEL, prior=prior, fit_pars=LM_PARS)
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(rng=rng, T=Tguess, prior=prior)

    if "em" in PSF_MODEL:
        psf_fitter  = ngmix.em.EMFitter(maxiter=EM_PARS["maxiter"], tol=EM_PARS["tol"])
        psf_guesser = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=_get_em_ngauss(PSF_MODEL))
    elif "coellip" in PSF_MODEL:
        n = _get_coellip_ngauss(PSF_MODEL)
        psf_fitter  = ngmix.fitting.CoellipFitter(ngauss=n, fit_pars=PSF_LM_PARS)
        psf_guesser = ngmix.guessers.CoellipPSFGuesser(rng=rng, ngauss=n)
    else:
        psf_fitter  = ngmix.fitting.Fitter(model="gauss", fit_pars=PSF_LM_PARS)
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)

    psf_runner = ngmix.runners.PSFRunner(fitter=psf_fitter, guesser=psf_guesser, ntry=NTRY)
    runner     = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=NTRY)
    boot       = ngmix.bootstrap.Bootstrapper(runner=runner, psf_runner=psf_runner)
    return _make_struct(boot.go(obs), obs)


def run_shearnet_single(obs):
    """Batch=1 with GPU sync.  Used in fair mode."""
    import jax
    import jax.numpy as jnp
    from shearnet.utils.normalization import inverse_transform_labels
    gal   = jnp.array(obs.image[np.newaxis])
    psf   = jnp.array(obs.psf.image[np.newaxis])
    preds = STATE.apply_fn(STATE.params, gal, psf, output_keys=OUTPUT_KEYS, gap=GAP, deterministic=True)
    jax.block_until_ready(preds)
    preds = np.array(preds)
    if NORM_PARAMS is not None:
        preds = inverse_transform_labels(preds, NORM_PARAMS)
    return preds


# Top-level function required for multiprocessing pickling
def _ngmix_pool_worker(args):
    obs, seed = args
    return run_ngmix(obs, seed)


# ============================================================
# SUMMARY HELPER
# ============================================================

def report(label, arr_s):
    ms = arr_s * 1e3
    print(f"\n{label}")
    print(f"  n              : {len(arr_s)}")
    print(f"  total          : {arr_s.sum():.3f} s")
    print(f"  mean +/- std   : {ms.mean():.3f} +/- {ms.std():.3f} ms")
    print(f"  median         : {np.median(ms):.3f} ms")
    print(f"  95th pct       : {np.percentile(ms, 95):.3f} ms")
    print(f"  min / max      : {ms.min():.3f} / {ms.max():.3f} ms")


# ============================================================
# FAIR MODE
# ============================================================

def run_fair():
    _init_shearnet()  # safe: no Pool, JAX threads are fine here
    import jax.numpy as jnp

    print("=" * 60)
    print(f"WARM-UP  ({N_WARMUP} galaxies -- results discarded)")
    print("=" * 60)
    rng_wu = np.random.RandomState(SEED + 99999)
    for i in range(N_WARMUP):
        obs_w = make_data(rng_wu)
        run_ngmix(obs_w, seed=SEED + i)
        if INCLUDE_SN:
            run_shearnet_single(obs_w)
    print("Warm-up complete.\n")

    print("=" * 60)
    print(f"FAIR BENCHMARK  ({NOBS} galaxies, 1 CPU, batch=1)")
    print("=" * 60)

    ngmix_times    = np.empty(NOBS)
    shearnet_times = np.empty(NOBS)
    rng_main = np.random.RandomState(SEED)

    for i in range(NOBS):
        obs = make_data(rng_main)

        t0 = time.perf_counter()
        run_ngmix(obs, seed=SEED + i)
        ngmix_times[i] = time.perf_counter() - t0

        if INCLUDE_SN:
            t0 = time.perf_counter()
            run_shearnet_single(obs)
            shearnet_times[i] = time.perf_counter() - t0

        if (i + 1) % 100 == 0 or i == 0:
            sn_str = f"  |  ShearNet {shearnet_times[i]*1e3:.2f} ms" if INCLUDE_SN else ""
            print(f"  [{i+1:>5}/{NOBS}]  ngmix {ngmix_times[i]*1e3:.1f} ms{sn_str}")

    print("\n" + "=" * 60)
    print("RESULTS — FAIR")
    print("=" * 60)
    report("ngmix  (1 CPU, sequential, no metacal)", ngmix_times)
    if INCLUDE_SN:
        report("ShearNet  (GPU, batch=1, post-JIT)", shearnet_times)
        print(f"\n  Speed-up : {ngmix_times.mean() / shearnet_times.mean():.1f}x")

    save_dict = dict(
        mode           = "fair",
        ngmix_times_s  = ngmix_times,
        ngmix_mean_s   = ngmix_times.mean(),
        ngmix_std_s    = ngmix_times.std(),
        ngmix_total_s  = ngmix_times.sum(),
        ngmix_median_s = np.median(ngmix_times),
    )
    if INCLUDE_SN:
        save_dict.update(dict(
            shearnet_times_s  = shearnet_times,
            shearnet_mean_s   = shearnet_times.mean(),
            shearnet_std_s    = shearnet_times.std(),
            shearnet_total_s  = shearnet_times.sum(),
            shearnet_median_s = np.median(shearnet_times),
            speedup           = ngmix_times.mean() / shearnet_times.mean(),
        ))
    return save_dict, ngmix_times, shearnet_times if INCLUDE_SN else None


# ============================================================
# REALISTIC MODE
# ============================================================

def run_realistic():
    print("=" * 60)
    print(f"REALISTIC BENCHMARK  ({NOBS} galaxies)")
    print(f"  ngmix   : {NPROC} CPU workers via multiprocessing Pool")
    if INCLUDE_SN:
        print(f"  ShearNet: GPU, batch_size={SN_BATCH_SIZE}")
    print("=" * 60)

    # Generate all observations up front (not part of the timed region;
    # both methods would face identical simulation overhead in a real pipeline).
    # JAX has NOT been imported yet, so forked workers inherit a lean process.
    print("\nGenerating observations...")
    rng_main = np.random.RandomState(SEED)
    all_obs  = [make_data(rng_main) for _ in range(NOBS)]
    seeds    = [SEED + i for i in range(NOBS)]
    print("Done.\n")

    # ---- ngmix: Pool  (JAX not yet imported — clean fork) ----
    print(f"Running ngmix Pool ({NPROC} workers)...")
    t0_ngmix = time.perf_counter()
    with Pool(processes=NPROC) as pool:
        _ = pool.map(_ngmix_pool_worker, zip(all_obs, seeds))
    ngmix_wall = time.perf_counter() - t0_ngmix
    ngmix_eff  = np.full(NOBS, ngmix_wall / NOBS)
    print(f"  ngmix wall time  : {ngmix_wall:.2f} s  ({ngmix_wall/NOBS*1e3:.2f} ms/galaxy effective)\n")

    # ---- ShearNet: initialise JAX now that the Pool is closed ----
    _init_shearnet()
    import jax
    import jax.numpy as jnp
    from shearnet.utils.normalization import inverse_transform_labels

    shearnet_eff = None
    if INCLUDE_SN:
        # Warm-up (JIT compilation) — discarded
        print(f"ShearNet warm-up ({N_WARMUP} galaxies)...")
        rng_wu = np.random.RandomState(SEED + 99999)
        for i in range(N_WARMUP):
            run_shearnet_single(make_data(rng_wu))
        print("ShearNet warm-up complete.\n")

        # Batched inference
        print(f"Running ShearNet batched (batch_size={SN_BATCH_SIZE})...")
        gal_stack = np.stack([obs.image     for obs in all_obs])
        psf_stack = np.stack([obs.psf.image for obs in all_obs])

        t0_sn = time.perf_counter()
        for start in range(0, NOBS, SN_BATCH_SIZE):
            sl    = slice(start, min(start + SN_BATCH_SIZE, NOBS))
            gal_b = jnp.array(gal_stack[sl])
            psf_b = jnp.array(psf_stack[sl])
            preds = STATE.apply_fn(
                STATE.params, gal_b, psf_b,
                output_keys=OUTPUT_KEYS, gap=GAP, deterministic=True,
            )
            jax.block_until_ready(preds)
        shearnet_wall = time.perf_counter() - t0_sn
        shearnet_eff  = np.full(NOBS, shearnet_wall / NOBS)
        print(f"  ShearNet wall time : {shearnet_wall:.2f} s  ({shearnet_wall/NOBS*1e3:.2f} ms/galaxy effective)\n")

    print("\n" + "=" * 60)
    print("RESULTS — REALISTIC")
    print("=" * 60)
    print(f"\nngmix  ({NPROC} CPU cores, no metacal)")
    print(f"  total wall     : {ngmix_wall:.3f} s")
    print(f"  effective/gal  : {ngmix_wall/NOBS*1e3:.3f} ms")
    print(f"  throughput     : {NOBS/ngmix_wall:.1f} galaxies/s")

    if INCLUDE_SN:
        print(f"\nShearNet  (GPU, batch={SN_BATCH_SIZE})")
        print(f"  total wall     : {shearnet_wall:.3f} s")
        print(f"  effective/gal  : {shearnet_wall/NOBS*1e3:.3f} ms")
        print(f"  throughput     : {NOBS/shearnet_wall:.1f} galaxies/s")
        print(f"\n  Speed-up (throughput) : {(NOBS/ngmix_wall) / (NOBS/shearnet_wall):.1f}x")

    save_dict = dict(
        mode               = "realistic",
        nproc              = NPROC,
        shearnet_batch_size= SN_BATCH_SIZE,
        ngmix_wall_s       = ngmix_wall,
        ngmix_eff_per_gal_s= ngmix_wall / NOBS,
        ngmix_throughput   = NOBS / ngmix_wall,
        ngmix_times_s      = ngmix_eff,
    )
    if INCLUDE_SN:
        save_dict.update(dict(
            shearnet_wall_s        = shearnet_wall,
            shearnet_eff_per_gal_s = shearnet_wall / NOBS,
            shearnet_throughput    = NOBS / shearnet_wall,
            shearnet_times_s       = shearnet_eff,
            speedup_throughput     = (NOBS / ngmix_wall) / (NOBS / shearnet_wall),
        ))
    return save_dict, ngmix_eff, shearnet_eff


# ============================================================
# MAIN
# ============================================================

if REALISTIC:
    save_dict, ngmix_times, shearnet_times = run_realistic()
    mode_tag = "realistic"
else:
    save_dict, ngmix_times, shearnet_times = run_fair()
    mode_tag = "fair"


# ============================================================
# SAVE
# ============================================================

outdir = os.path.dirname(OUTPUT_NPZ)
if outdir:
    os.makedirs(outdir, exist_ok=True)

tagged_path = OUTPUT_NPZ.replace(".npz", f"_{mode_tag}.npz")
np.savez(tagged_path, **save_dict)
print(f"\nResults saved -> {tagged_path}")


# ============================================================
# PLOT
# ============================================================

ncols = 2 if (INCLUDE_SN and shearnet_times is not None) else 1
fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
if ncols == 1:
    axes = [axes]

x_label = "Time per galaxy (ms)"
title_suffix = "(realistic: throughput / N)" if REALISTIC else "(fair: per-galaxy latency)"

axes[0].hist(ngmix_times * 1e3, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(ngmix_times.mean() * 1e3, color="red", ls="--",
                label=f"{'effective ' if REALISTIC else ''}mean = {ngmix_times.mean()*1e3:.1f} ms")
axes[0].set_xlabel(x_label)
axes[0].set_ylabel("Count")
axes[0].set_title(f"ngmix {title_suffix}")
axes[0].legend()

if INCLUDE_SN and shearnet_times is not None:
    axes[1].hist(shearnet_times * 1e3, bins=40, color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].axvline(shearnet_times.mean() * 1e3, color="red", ls="--",
                    label=f"{'effective ' if REALISTIC else ''}mean = {shearnet_times.mean()*1e3:.2f} ms")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"ShearNet {title_suffix}")
    axes[1].legend()

plt.tight_layout()
plot_path = tagged_path.replace(".npz", "_histogram.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Histogram saved -> {plot_path}")
plt.close()