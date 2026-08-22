"""Training-matched shear-bias, PSF-leakage and timing benchmarks.

What this fixes relative to ``m/main.py`` and ``psf_leakage/main.py``: those
carry their own simulator, so nothing keeps them in step with the backend,
draw method, PSF settings or shear convention the model actually trained on.
Here every stamp comes from :class:`shearnet.benchmarking.TrainingMatchedRenderer`,
which builds a :class:`~shearnet.core.specs.DatasetSpec` from the *saved*
``training_config.yaml`` -- so a model trained with ``backend: jax-galsim`` and
``generation: inloop`` is benchmarked on jax-galsim stamps without the harness
knowing anything about backends.

Every estimator is calibrated the same way
------------------------------------------
``m`` and ``c`` are the pair-matched jackknife estimates of
:func:`shearnet.methods.anacal.paired_bias`, and the response that divides out
comes from one protocol for all estimators: re-render the scene at
``gamma +/- step`` with the noise held fixed and re-measure. metacal is kept
alongside it for ngmix (it estimates the same derivative by shearing the image
rather than the scene) and FPFS is additionally reported with its own analytic
response, so the protocol has two independent checks on it.

The legacy scripts still run: ``m/main.py`` and ``psf_leakage/main.py`` are
untouched, and disagreement between them and this entry point is a finding, not
a nuisance -- it is the drift this harness exists to expose.

Usage::

    python run.py --task m -c config.yaml
    python run.py --task psf-leakage -c config.yaml
    python run.py --task timing -c config.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from shearnet.benchmarking import (
    SavedModelPredictor,
    TrainingMatchedRenderer,
    load_training_config,
)
from shearnet.config.config_handler import Config
from shearnet.logging_utils import get_logger
from shearnet.parallel import resolve_nproc
from shearnet.methods.anacal import (
    ShapeMeasurement,
    leakage,
    paired_bias,
    renderer_shear_response,
)
from shearnet.methods.anacal_fit import measure_gauss_fit

logger = get_logger(__name__)

#: Estimators this harness runs.
#:
#:   ngmix     esheldon/ngmix, the baseline the field reads
#:   anacal    AnaCal's own C++ Gaussian model fit, which carries an ANALYTIC
#:             shear response out of the fit via quintuple numbers
#:             (Li 2026, arXiv:2506.16607). Not a wrapper around ngmix: a
#:             different fitter of the same model family, which is why it is
#:             its own column rather than a correction applied to ngmix.
#:   shearnet  the model under test
#:
#: FPFS was removed from the evaluation. Its columns are a subset of the same
#: AnaCal catalogue this model fit comes from, and now that the analytical
#: response reaches model fitting, the shapelet estimator is no longer the
#: thing AnaCal has to be compared through. The measurement code is untouched
#: in ``shearnet.methods.anacal`` and stays under test; the evaluation no
#: longer runs it.
ESTIMATORS = ("ngmix", "anacal", "shearnet")

#: The baselines ShearNet can be measured against. ShearNet is always in the
#: run -- a benchmark of it alone says nothing -- so the only choice is which
#: of these it is compared to, via ``eval.evaluate.baseline`` or ``--baseline``.
#: ``anacal`` is the one worth switching off when time is short: its fit is
#: serial at ~12.6 ms/object (see shearnet/methods/anacal_fit.py for why), so
#: it costs ~4.2 h at n_obs 200000 against ngmix's ~1.2 h on 18 workers.
BASELINES = ("ngmix", "anacal")

#: Objects per ngmix batch. An ngmix ``Observation`` is ~362 kB resident at a
#: 53x53 stamp -- the four float64 arrays are only 88 kB of that, the rest is
#: the ``pixels`` structured array ngmix builds eagerly -- so a 200k population
#: is 69 GB, and the bias task needs two of them. Everything downstream of the
#: fit is 2 numbers per object, so there is no reason to hold more than a batch.
NGMIX_CHUNK = 4096


def _parser():
    parser = argparse.ArgumentParser(description="Training-matched shear-bias benchmark")
    parser.add_argument("-c", "--config", required=True, help="Benchmark YAML configuration")
    parser.add_argument(
        "--training-config",
        default=None,
        help="Saved training_config.yaml; resolved from the model name when omitted.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        choices=BASELINES + ("both",),
        help=(
            "Which baseline to measure ShearNet against. ShearNet is always in "
            f"the run; this picks the comparison. Default is the config's "
            "eval.evaluate.baseline."
        ),
    )
    return parser


# ----------------------------------------------------------------------
# config plumbing (the unit-test 'eval' schema and the package schema)
# ----------------------------------------------------------------------
def _eval(config: Config, key: str, default=None):
    value = config.get(f"eval.{key}")
    return config.get(f"evaluation.{key}", default) if value is None else value


def _section(config: Config, task: str = "evaluate") -> dict:
    """The evaluation block, accepting the older ``eval.bias`` spelling."""
    for key in (task, "bias"):
        section = config.get(f"eval.{key}", config.get(f"evaluation.{key}"))
        if section:
            return section
    return {}


def _model_name(config: Config) -> str:
    name = config.get("meta.model_name") or config.get("output.model_name")
    if not name:
        raise ValueError("Benchmark config must set meta.model_name or output.model_name")
    return name


def _estimators(section: dict, override: Optional[str]) -> Sequence[str]:
    """``[baseline..., 'shearnet']`` -- the run is always ShearNet vs something.

    A benchmark of ShearNet alone has nothing to say, so the only choice is
    which baseline it is measured against. ``both`` runs the pair.

    Order is fixed here rather than taken from the caller, so the FITS columns
    and the SUMMARY rows come out in the same order whatever the config said.
    """
    requested = override if override is not None else section.get("baseline", "ngmix")
    if isinstance(requested, str):
        requested = list(BASELINES) if requested == "both" else [requested.strip()]
    requested = [str(name).strip() for name in requested]
    unknown = sorted(set(requested) - set(BASELINES))
    if unknown:
        raise ValueError(
            f"unknown baseline {unknown}; choose from {list(BASELINES)} or 'both'"
        )
    if not requested:
        raise ValueError(f"pick a baseline: one of {list(BASELINES)}, or 'both'")
    return [name for name in ESTIMATORS if name in set(requested) | {"shearnet"}]


def _renderer(benchmark: Config, training: Config) -> TrainingMatchedRenderer:
    return TrainingMatchedRenderer(training, eval_catalog=benchmark.get("paths.eval_catalog"))


def _noise_sd(training: Config) -> float:
    noise = training.get("training.noise", {}) or {}
    low, high = noise.get("min_sd"), noise.get("max_sd")
    if low is not None and high is not None:
        return (float(low) + float(high)) / 2.0
    return float(training.get("dataset.nse_sd"))


def _flatten(prefix: str, bias, extra=None) -> dict:
    out = {
        f"{prefix}_m": bias.m,
        f"{prefix}_m_err": bias.m_err,
        f"{prefix}_c": bias.c,
        f"{prefix}_c_err": bias.c_err,
        f"{prefix}_response": bias.response,
        f"{prefix}_n_used": bias.n_used,
    }
    if bias.bins is not None:
        out.update({f"{prefix}_bin_{k}": v for k, v in bias.bins.items()})
    if extra:
        out.update({f"{prefix}_{k}": v for k, v in extra.items()})
    logger.info(bias.describe(prefix))
    return out


# ----------------------------------------------------------------------
# per-estimator measurement
# ----------------------------------------------------------------------
def _ngmix_batches(renderer, galaxy, psf, chunk=NGMIX_CHUNK):
    """Yield ``(start, observations)`` a batch at a time.

    The observations for a whole population never exist at once. That is the
    difference between the bias task fitting in an allocation and not: at 200k
    objects the plus and minus populations alone would be 138 GB of ngmix
    ``Observation``, before a single fit runs.
    """
    for start in range(0, len(galaxy), int(chunk)):
        stop = start + int(chunk)
        yield start, renderer.observations_from_stamps(galaxy[start:stop], psf[start:stop])


def _measure_callables(
    renderer, estimators, *, seed, psf_model, gal_model, anacal_kw, predictor, nproc=None
):
    """``{name: measure(galaxy, psf) -> (N, 2)}`` for the requested estimators.

    One callable per estimator, so the shared response pass can hand every one
    of them the same rendered stamps instead of each re-rendering the
    population for itself.
    """
    from shearnet.methods.ngmix import fit_shapes

    measures = {}
    if "shearnet" in estimators:
        measures["shearnet"] = predictor.shear_measure()
    if "ngmix" in estimators:

        def ngmix_measure(galaxy, psf):
            e = np.full((len(galaxy), 2), np.nan)
            for start, obs in _ngmix_batches(renderer, galaxy, psf):
                e[start : start + len(obs)] = fit_shapes(
                    obs, seed=seed, psf_model=psf_model, gal_model=gal_model, nproc=nproc
                )[0]
                del obs
            return e

        measures["ngmix"] = ngmix_measure
    if "anacal" in estimators:

        def anacal_measure(galaxy, psf):
            return measure_gauss_fit(galaxy, psf, **anacal_kw).e

        measures["anacal"] = anacal_measure
    return measures


def _shared_shear_responses(renderer, measures, *, samples, seed, base_shear, step):
    """``de/dgamma`` for every estimator, rendering each offset exactly ONCE.

    The four re-renders that make up a central difference are identical across
    estimators -- only the measurement differs -- so rendering them per
    estimator multiplies the dominant cost by the number of columns in the
    table. At SuperBIT's ~40 ms/stamp that is the difference between 36 and 142
    core-hours for a 400k run, which is the difference between running the
    comparison and not.

    Returns ``{name: (N, 2, 2)}`` with ``[:, a, b] = de_a / dgamma_b``.
    """
    columns = {name: [None, None] for name in measures}
    for axis in (0, 1):
        measured = {name: {} for name in measures}
        for sign in (1.0, -1.0):
            offset = list(base_shear)
            offset[axis] += sign * step
            stamps = renderer.render(
                samples, seed=seed, base_shear_g1=offset[0], base_shear_g2=offset[1]
            )
            for name, measure in measures.items():
                value = np.asarray(measure(stamps.galaxy_images, stamps.psf_images), dtype=float)
                measured[name][sign] = value[:, :2]
        for name in measures:
            columns[name][axis] = (measured[name][1.0] - measured[name][-1.0]) / (2.0 * step)
    return {name: np.stack(cols, axis=-1) for name, cols in columns.items()}


def _metacal_pass(renderer, galaxy, psf, *, seed, psf_model, gal_model, step, nproc, predictor):
    """One metacal pass serving BOTH estimators.

    metacal shears the *image*: it deconvolves by the PSF, applies +/- step,
    and reconvolves with a dilated PSF. ngmix reads its response off its own
    fits to those five images. A network gets a metacal response the same way --
    by measuring the same five images -- which is why the reconvolved stamps
    come back from the worker. Note this is not deconvolution from the network's
    point of view: what it sees is an ordinary galaxy convolved with a slightly
    wider PSF.

    Running one pass for both is not an optimisation detail. metacal costs
    ~156 ms/object; doing it twice would double the dominant cost of the whole
    evaluation to produce the identical images.

    Returns ``(ngmix_shape, ngmix_extra, shearnet_shape)`` -- the last is None
    when there is no predictor.
    """
    from shearnet.methods.ngmix import METACAL_TYPES, _get_priors, mp_fit_one_single

    n = len(galaxy)
    want_network = predictor is not None
    e = np.full((n, 2), np.nan)
    dedg = np.full((n, 2, 2), np.nan)
    flags = np.ones(n, dtype=bool)
    sn_e = np.full((n, 2), np.nan)
    sn_dedg = np.full((n, 2, 2), np.nan)
    sn_flags = np.ones(n, dtype=bool)
    # Per-object scalars the diagnostics plot against. Free here -- they are
    # already in the struct -- and unrecoverable afterwards, because the fit
    # results are not kept.
    extra = {key: np.full(n, np.nan) for key in ("s2n", "T", "flux", "Tpsf")}
    measure = predictor.shear_measure() if want_network else None
    index = {name: i for i, name in enumerate(METACAL_TYPES)}

    for start, obs in _ngmix_batches(renderer, galaxy, psf):
        results, _ = mp_fit_one_single(
            obs,
            _get_priors(seed),
            np.random.RandomState(seed),
            psf_model=psf_model,
            gal_model=gal_model,
            mcal_pars={"psf": "dilate", "mcal_shear": step},
            nproc=nproc,
            return_images=want_network,
        )
        del obs
        rows_only = [r[0] for r in results] if want_network else results

        for offset, rows in enumerate(rows_only):
            i = start + offset
            by_type = {str(row["shear_type"]): row for row in rows}
            if "noshear" in by_type:
                for key in extra:
                    extra[key][i] = by_type["noshear"][key]
            if not all(k in by_type and by_type[k]["flags"] == 0 for k in METACAL_TYPES):
                continue
            e[i] = by_type["noshear"]["g"]
            for b, (up, down) in enumerate((("1p", "1m"), ("2p", "2m"))):
                dedg[i, :, b] = (by_type[up]["g"] - by_type[down]["g"]) / (2.0 * step)
            flags[i] = False

        if want_network and results:
            # Five metacal images per object, measured in one batched forward:
            # stack them as (5N, npix, npix) so the network sees one large batch
            # rather than five small ones.
            stack = np.concatenate([r[1] for r in results], axis=0)
            psf_stack = np.repeat(
                np.stack([r[2] for r in results], axis=0), len(METACAL_TYPES), axis=0
            )
            measured = np.asarray(measure(stack, psf_stack), dtype=float)
            measured = measured.reshape(len(results), len(METACAL_TYPES), 2)
            block = slice(start, start + len(results))
            sn_e[block] = measured[:, index["noshear"], :]
            for b, (up, down) in enumerate((("1p", "1m"), ("2p", "2m"))):
                sn_dedg[block, :, b] = (
                    measured[:, index[up], :] - measured[:, index[down], :]
                ) / (2.0 * step)
            sn_flags[block] = ~np.isfinite(measured).all(axis=(1, 2))
            del stack, psf_stack, results

    ngmix_shape = ShapeMeasurement(e=e, dedg=dedg, flags=flags)
    network_shape = (
        ShapeMeasurement(e=sn_e, dedg=sn_dedg, flags=sn_flags) if want_network else None
    )
    return ngmix_shape, extra, network_shape


# ----------------------------------------------------------------------
# per-object catalogs (what research/shear_bias/plots_from_fits.ipynb reads)
# ----------------------------------------------------------------------
def _psf_moments(renderer, psf_images, chunk=NGMIX_CHUNK):
    """Adaptive-moment ``(gpsf, Tpsf)`` per object, for the leakage x-axis.

    The notebook bins <e> against e_PSF, so the PSF shape has to be measured,
    not assumed. Same measurement ``m/helpers.make_struct`` uses, so the column
    means what it has always meant.
    """
    import ngmix

    from shearnet.core.moments import get_admoms_ngmix_fit

    n = len(psf_images)
    gpsf = np.full((n, 2), np.nan)
    tpsf = np.full(n, np.nan)
    centre = (psf_images.shape[1] - 1.0) / 2.0
    jacobian = ngmix.DiagonalJacobian(row=centre, col=centre, scale=renderer.spec.scale)
    for start in range(0, n, chunk):
        for offset, image in enumerate(psf_images[start : start + chunk]):
            obs = ngmix.Observation(
                np.ascontiguousarray(image, dtype=np.float64), jacobian=jacobian
            )
            fit = get_admoms_ngmix_fit(obs, reduced=True)
            if fit["flags"] == 0:
                gpsf[start + offset] = (fit["e1"], fit["e2"])
                tpsf[start + offset] = fit["T"]
    return gpsf, tpsf


def _catalog_table(columns):
    """``astropy.table.Table`` from ``{name: array}``, dropping empty columns."""
    from astropy.table import Table

    return Table({k: np.asarray(v) for k, v in columns.items() if v is not None})


def _write_catalog(config: Config, task: str, relative, hdus) -> Optional[Path]:
    """Write per-object tables as a multi-extension FITS, or nothing.

    The ``.npz`` this harness has always written holds the ENSEMBLE answer -- m,
    c, the response, the binned tables -- which is what the benchmark is for.
    ``plots_from_fits.ipynb`` plots per-object diagnostics and needs rows, so it
    reads a different product with a different granularity, not the same product
    in a different format. This writes that product, in the schema and under the
    filenames the notebook already expects.
    """
    if not relative:
        return None
    from astropy.io import fits

    root = Path(config.get("paths.root") or config.get("output.plot_path") or ".")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu_list = [fits.PrimaryHDU()]
    for name, table in hdus:
        hdu_list.append(fits.BinTableHDU(table, name=name))
    fits.HDUList(hdu_list).writeto(path, overwrite=True)
    logger.info("Saved per-object %s catalog to %s", task, path)
    return path


# ----------------------------------------------------------------------
# preflight
# ----------------------------------------------------------------------
#: Measured on a 53x53 stamp: resident bytes per ngmix Observation (362 kB, of
#: which only 88 kB is the four float64 arrays), and seconds per fit for a plain
#: shape fit and for a metacal fit. Used only to print a budget before the run
#: commits to it -- a wrong constant misprints a log line, it does not change a
#: result. Rescaled by stamp area, which is how all three actually vary.
_OBS_BYTES_PER_PIXEL = 362e3 / (53 * 53)
_FIT_SECONDS_PER_PIXEL = 6.9e-3 / (53 * 53)
_METACAL_SECONDS_PER_PIXEL = 156e-3 / (53 * 53)
#: AnaCal's Gaussian model fit, measured at 12.6 ms on a 53x53 stamp.
_ANACAL_SECONDS_PER_PIXEL = 12.6e-3 / (53 * 53)


def _log_cost_estimate(samples, renderer, estimators, section) -> None:
    """Print the memory and CPU budget this configuration is about to ask for.

    The bias task is the expensive one and it fails late: rendering happens
    first, so an allocation that cannot hold the measurement stage finds out
    hours in, with an OOM kill and no output. Estimating it up front costs
    nothing and turns that into a number you can read before the queue does.
    """
    npix = int(renderer.spec.npix)
    area = npix * npix
    gib = 1024**3
    stamps_gib = samples * area * 4 * 2 / gib  # gal + psf, float32
    # 2 populations resident + at most one response render at a time
    lines = [f"stamps {3 * stamps_gib:.1f} GiB (3 populations x {stamps_gib:.1f})"]
    seconds = 0.0

    workers = resolve_nproc(section.get("ngmix_nproc"), n_tasks=samples)
    # metacal runs whenever the network does, because that is where ShearNet's
    # metacal response comes from.
    if "ngmix" in estimators or "shearnet" in estimators:
        batch_gib = min(samples, NGMIX_CHUNK) * area * _OBS_BYTES_PER_PIXEL / gib
        lines.append(f"ngmix observations {batch_gib:.2f} GiB (batched at {NGMIX_CHUNK})")
        lines.append(f"metacal on {workers} worker(s)")
        seconds += 2 * samples * area * _METACAL_SECONDS_PER_PIXEL / workers
    if "ngmix" in estimators:
        # 2 populations + 2 axes x 2 signs, each fit once per population
        seconds += 2 * (1 + 4) * samples * area * _FIT_SECONDS_PER_PIXEL / workers
    if "anacal" in estimators:
        # 2 analytic passes + 2 populations x 4 sim re-renders. Serial: see
        # shearnet/methods/anacal_fit.py for why it is not parallelised.
        # 2 populations + 2 populations x 4 sim re-renders; the analytic pass
        # is not extra, because one fit returns the value and the response.
        anacal_seconds = 2 * (1 + 4) * samples * area * _ANACAL_SECONDS_PER_PIXEL
        seconds += anacal_seconds
        lines.append(f"AnaCal fit SERIAL ({anacal_seconds / 3600.0:.1f} h of the total)")

    logger.info("preflight: %s", "; ".join(lines))
    if seconds:
        logger.info(
            "preflight: measurement ~%.1f h; raise eval.evaluate.ngmix_nproc, "
            "drop to --baseline ngmix, or lower eval.n_obs if that does not fit "
            "the wall clock",
            seconds / 3600.0,
        )


# ----------------------------------------------------------------------
# the evaluation
# ----------------------------------------------------------------------
#: Correction schemes. Every estimator is measured under every correction it
#: admits, at every evaluation -- there is no option to produce half a result.
#:
#:   metacal  shear the IMAGE: deconvolve, apply +/- step, reconvolve with a
#:            dilated PSF, re-measure. ngmix's MetacalBootstrapper; ShearNet
#:            measures the same five reconvolved images.
#:   anacal   AnaCal's ANALYTIC response, differentiated symbolically through
#:            the fit with quintuple numbers. Exact, not a finite difference --
#:            but available only for an estimator AnaCal can differentiate,
#:            which here is its own Gaussian model fit.
#:   sim      shear the SCENE: re-render at gamma +/- step with the galaxies,
#:            PSFs and noise realisation all held fixed. Works for any
#:            estimator, black box or differentiable, so it is the one column
#:            every row has -- and the `anacal` estimator, which carries both,
#:            is what says whether `sim` agrees with the exact answer.
#:
#: A pair the harness cannot measure is simply absent from SUMMARY, rather than
#: filled in from a different derivative wearing the same name.
CORRECTIONS = ("metacal", "anacal", "sim")


def _run_evaluation(benchmark: Config, training: Config, estimators) -> dict:
    """Measure every estimator under every correction, and write one FITS.

    One pass produces: the plus/minus shear populations with per-object
    ellipticities and both response matrices, the PSF-leakage table, the
    ensemble m/c for each (estimator, correction) pair, and the timing. There
    is deliberately no way to ask for a subset -- a benchmark that sometimes
    writes half its columns is a benchmark whose outputs cannot be compared.
    """
    renderer = _renderer(benchmark, training)
    section = _section(benchmark, "evaluate")
    seed = int(_eval(benchmark, "seed"))
    samples = int(_eval(benchmark, "n_obs", benchmark.get("evaluation.test_samples")))
    shear = float(section.get("shear_true", 0.01))
    component = int(section.get("component", 0))
    step = float(section.get("response_step", section.get("metacal_step", 0.01)))
    njac = int(section.get("n_jackknife", 20))
    psf_model = section.get("psf_model", "gauss")
    gal_model = _eval(benchmark, "gal_model", "gauss")
    c_convention = section.get("c_convention", "shearnet")
    resample = section.get("resample", "jackknife")
    nproc = section.get("ngmix_nproc")
    batch = int(section.get("shearnet_batch_size", 4096))
    noise_sd = _noise_sd(training)
    anacal_kw = dict(
        pixel_scale=renderer.spec.scale,
        noise_variance=max(noise_sd, 1e-12) ** 2,
        sigma_arcsec=float(section.get("anacal_sigma_arcsec", 0.52)),
        stamp_size=int(section.get("anacal_stamp_size", 32)),
        num_epochs=int(section.get("anacal_epochs", 35)),
    )

    if renderer.spec.backend != "jax-galsim":
        # Raised before anything expensive runs. R^PSF is a finite difference on
        # the PSF shear, which needs the jax-galsim backend's explicit
        # psf_g1/psf_g2 per-object parameters -- and an evaluation that silently
        # dropped it would be exactly the half-result this entry point exists to
        # rule out.
        raise ValueError(
            f"the evaluation needs dataset.backend: jax-galsim, got "
            f"{renderer.spec.backend!r}. R^PSF is measured by finite-differencing "
            "the PSF shear, which the galsim backend has no handle for, so this "
            "run could only produce part of the table. Retrain or re-render with "
            "the jax-galsim backend."
        )

    logger.info(
        "evaluation: %d objects, seed %d, shear %+.4f on g%d, backend %s, "
        "generation %s, exp %s, estimators %s, corrections %s",
        samples, seed, shear, component + 1, renderer.spec.backend,
        renderer.spec.generation, renderer.spec.exp, list(estimators), list(CORRECTIONS),
    )
    _log_cost_estimate(samples, renderer, estimators, section)

    predictor = (
        SavedModelPredictor(_model_name(benchmark), config=training)
        if "shearnet" in estimators
        else None
    )
    measures = _measure_callables(
        renderer, estimators, seed=seed, psf_model=psf_model, gal_model=gal_model,
        anacal_kw=anacal_kw, predictor=predictor, nproc=nproc,
    )

    plus, minus = renderer.shear_pair(samples, seed=seed, shear=shear, component=component)
    if plus.psf_images is None:
        raise RuntimeError("Training-matched benchmarks require the PSF channel")

    result = {
        "backend": renderer.spec.backend,
        "generation": renderer.spec.generation,
        "exp": renderer.spec.exp,
        "seed": seed,
        "samples": samples,
        "shear_true": shear,
        "component": component,
        "response_step": step,
        "c_convention": c_convention,
        "resample": resample,
        "estimators": np.asarray(list(estimators), dtype="U16"),
        "corrections": np.asarray(list(CORRECTIONS), dtype="U16"),
        "ngmix_nproc": resolve_nproc(nproc, n_tasks=samples),
    }
    common = dict(
        component=component, njac=njac, c_convention=c_convention, resample=resample,
        bin_values=np.sum(plus.galaxy_images, axis=(1, 2)),
    )

    # shapes[estimator][correction][population] -> ShapeMeasurement
    shapes = {name: {c: {} for c in CORRECTIONS} for name in estimators}
    columns = {"plus": {}, "minus": {}}

    for label, stamps, sign in (("plus", plus, 1.0), ("minus", minus, -1.0)):
        col = columns[label]
        base = [0.0, 0.0]
        base[component] = sign * shear

        # --- truth and PSF, once per population --------------------------
        labels = np.asarray(stamps.labels, dtype=float)
        col["g_th"] = labels[:, :2]
        if labels.shape[1] > 2:
            col["hlr_th"] = labels[:, 2]
        if labels.shape[1] > 3:
            col["flux_th"] = labels[:, 3]
        gpsf, tpsf = _psf_moments(renderer, stamps.psf_images)
        col["gpsf"], col["Tpsf"] = gpsf, tpsf
        gal = np.asarray(stamps.galaxy_images, dtype=float)
        col["s2n"] = np.sqrt(np.sum(gal**2, axis=(1, 2))) / max(noise_sd, 1e-12)
        del gal

        # --- the anacal (analytic) response -------------------------------
        # First, because one call returns the shape AND its exact de/dgamma:
        # the quintuple numbers carry the derivative out of the fit itself.
        # Measuring here and reusing the value below saves refitting the same
        # stamps -- at ~12.6 ms/object and no parallelism, that duplicate pass
        # was ~40 minutes of a 200k run.
        analytic = None
        if "anacal" in estimators:
            analytic = measure_gauss_fit(
                stamps.galaxy_images, stamps.psf_images, **anacal_kw
            )
            shapes["anacal"]["anacal"][label] = analytic
            col["R_anacal_anacal"] = analytic.dedg
            col["flag_anacal"] = analytic.flags.astype(np.int32)

        # --- the sim (scene) response, shared across every estimator ------
        raw = {}
        for name, measure in measures.items():
            if name == "anacal" and analytic is not None:
                raw[name] = analytic.e  # the same fit, already run
            else:
                raw[name] = np.asarray(
                    measure(stamps.galaxy_images, stamps.psf_images), dtype=float
                )
        responses = _shared_shear_responses(
            renderer, measures, samples=samples, seed=seed, base_shear=base, step=step
        )
        for name in measures:
            e = raw[name][:, :2]
            shapes[name]["sim"][label] = ShapeMeasurement(
                e=e, dedg=responses[name], flags=~np.isfinite(e).all(axis=1)
            )
            col[f"e_{name}"] = e
            col[f"R_{name}_sim"] = responses[name]
        if predictor is not None:
            # shear_measure() slices to (g1, g2) because that is all the
            # response protocol needs, so the size and flux the network also
            # predicts have to be read from a full forward pass. One extra pass
            # over the population, ~0.1 ms/stamp, for two columns that are a
            # deliverable in their own right.
            full = np.asarray(
                predictor(stamps.galaxy_images, stamps.psf_images, batch_size=batch), dtype=float
            )
            for i, key in enumerate(predictor.output_keys):
                if key in ("hlr", "flux") and i < full.shape[1]:
                    col[f"{key}_shearnet"] = full[:, i]

        # --- the metacal (image) response, one pass for both estimators ---
        # Gated on the network, not on ngmix. metacal is a *correction*, and
        # ShearNet's metacal response comes from the same bootstrapper, so
        # dropping ngmix as a reported baseline must not silently cost ShearNet
        # a column. The ngmix columns below are written only when ngmix is
        # actually in the table.
        if predictor is not None or "ngmix" in estimators:
            ngmix_shape, ngmix_extra, network_shape = _metacal_pass(
                renderer, stamps.galaxy_images, stamps.psf_images,
                seed=seed + (1 if label == "plus" else 2),
                psf_model=psf_model, gal_model=gal_model, step=step, nproc=nproc,
                predictor=predictor,
            )
            if "ngmix" in estimators:
                shapes["ngmix"]["metacal"][label] = ngmix_shape
                col["R_ngmix_metacal"] = ngmix_shape.dedg
                col["flag_ngmix"] = ngmix_shape.flags.astype(np.int32)
                col["s2n_ngmix"] = ngmix_extra["s2n"]
                col["T_ngmix"] = ngmix_extra["T"]
                col["flux_ngmix"] = ngmix_extra["flux"]
            if network_shape is not None:
                shapes["shearnet"]["metacal"][label] = network_shape
                col["R_shearnet_metacal"] = network_shape.dedg
                col["flag_shearnet"] = network_shape.flags.astype(np.int32)


    # --- ensemble m and c for every (estimator, correction) ---------------
    for name, by_correction in shapes.items():
        for correction, populations in by_correction.items():
            if set(populations) != {"plus", "minus"}:
                continue
            prefix = f"{name}_{correction}"
            result.update(
                _flatten(
                    prefix,
                    paired_bias(populations["plus"], populations["minus"], shear, **common),
                )
            )

    leakage_columns = _leakage_pass(
        renderer, measures, predictor, samples=samples, seed=seed, step=step,
        njac=njac, noise_sd=noise_sd, result=result,
    )
    result.update(_timing_pass(renderer, predictor, samples=samples, seed=seed, batch=batch))
    _write_evaluation_fits(benchmark, section, columns, leakage_columns, result)
    return result


def _leakage_pass(renderer, measures, predictor, *, samples, seed, step, njac, noise_sd, result):
    """``R^PSF`` per estimator on an unsheared population.

    Measured by the scene protocol -- offset every object's PSF shear by
    +/- step with the galaxies and noise pinned -- so the column is ``sim`` for
    every estimator. metacal has no counterpart here unless the bootstrapper is
    configured with the ``*_psf`` shear types, which it is not; neither does the
    analytic route, whose quintuple numbers carry ``d/dgamma`` and ``d/dx`` but
    not ``d/de^PSF``.
    """
    stamps = renderer.render(samples, seed=seed)
    columns = {}
    gpsf, tpsf = _psf_moments(renderer, stamps.psf_images)
    columns["gpsf"], columns["Tpsf"] = gpsf, tpsf
    gal = np.asarray(stamps.galaxy_images, dtype=float)
    columns["s2n"] = np.sqrt(np.sum(gal**2, axis=(1, 2))) / max(noise_sd, 1e-12)
    del gal

    for name, measure in measures.items():
        psf_response = renderer_shear_response(
            renderer.response_renderer(samples, seed=seed, psf=True), measure, step=step, psf=True
        )
        e = np.asarray(measure(stamps.galaxy_images, stamps.psf_images), dtype=float)[:, :2]
        stats = leakage(e, psf_response, njac=njac)
        result.update({f"{name}_leakage_{k}": v for k, v in stats.items()})
        columns[f"e_{name}"] = e
        columns[f"Rpsf_{name}_sim"] = psf_response
        columns[f"Rbar_psf_{name}"] = np.full(samples, float(stats["psf_response"][0, 0]))
        logger.info(
            "%s R^PSF = [[%+.3e %+.3e] [%+.3e %+.3e]] +/- %.1e",
            name, *stats["psf_response"].ravel(), float(np.max(stats["psf_response_err"])),
        )
    return columns


def _timing_pass(renderer, predictor, *, samples, seed, batch):
    """Render and inference wall-clock, on the same population size.

    In-loop training has no separable render cost -- the render lives inside the
    step -- so this times the *benchmark* render, which is always up front. The
    reported generation records which mode trained the model, so the number is
    not misread as training throughput.
    """
    warm = renderer.render(2, seed=seed + 10_000)
    if predictor is not None:
        predictor(warm.galaxy_images, warm.psf_images, batch_size=batch)  # compile

    start = time.perf_counter()
    stamps = renderer.render(samples, seed=seed)
    render_seconds = time.perf_counter() - start
    inference_seconds = float("nan")
    if predictor is not None:
        start = time.perf_counter()
        predictor(stamps.galaxy_images, stamps.psf_images, batch_size=batch)
        inference_seconds = time.perf_counter() - start
    return {
        "render_seconds": render_seconds,
        "inference_seconds": inference_seconds,
        "timing_note": "benchmark rendering is always up-front; generation is the training mode",
    }


def _summary_table(result):
    """One row per (estimator, correction) -- the table you read first."""
    rows = {k: [] for k in ("estimator", "correction", "m", "m_err", "c", "c_err",
                            "R11", "R22", "n_used")}
    for key in sorted(result):
        if not key.endswith("_m") or f"{key}_err" not in result:
            continue
        prefix = key[: -len("_m")]
        correction = next((c for c in CORRECTIONS if prefix.endswith(f"_{c}")), None)
        if correction is None:
            continue
        response = np.asarray(result[f"{prefix}_response"], dtype=float)
        rows["estimator"].append(prefix[: -len(correction) - 1])
        rows["correction"].append(correction)
        rows["m"].append(float(result[f"{prefix}_m"]))
        rows["m_err"].append(float(result[f"{prefix}_m_err"]))
        rows["c"].append(float(result[f"{prefix}_c"]))
        rows["c_err"].append(float(result[f"{prefix}_c_err"]))
        rows["R11"].append(response[0, 0])
        rows["R22"].append(response[1, 1])
        rows["n_used"].append(int(result[f"{prefix}_n_used"]))
    return _catalog_table(rows)


def _write_evaluation_fits(benchmark, section, columns, leakage_columns, result):
    """One FITS holding everything the evaluation measured.

    ``TAB_P`` / ``TAB_M`` are the +gamma / -gamma populations, one row per
    object; ``LEAKAGE`` is the unsheared population with ``R^PSF``; ``SUMMARY``
    is one row per (estimator, correction). Written unconditionally: there is
    no config switch that produces a partial file.
    """
    from astropy.io import fits

    relative = section.get("output", "benchmarking/evaluation.fits")
    root = Path(benchmark.get("paths.root") or benchmark.get("output.plot_path") or ".")
    path = root / relative
    if path.suffix.lower() not in (".fits", ".fit"):
        path = path.with_suffix(".fits")
    path.parent.mkdir(parents=True, exist_ok=True)

    primary = fits.PrimaryHDU()
    for key in ("backend", "generation", "exp", "seed", "samples", "shear_true",
                "component", "response_step", "c_convention", "resample",
                "ngmix_nproc", "render_seconds", "inference_seconds"):
        if key in result:
            value = result[key]
            primary.header[key[:8].upper()] = (
                value.item() if hasattr(value, "item") else value, key
            )

    hdus = [primary,
            fits.BinTableHDU(_catalog_table(columns["plus"]), name="TAB_P"),
            fits.BinTableHDU(_catalog_table(columns["minus"]), name="TAB_M"),
            fits.BinTableHDU(_catalog_table(leakage_columns), name="LEAKAGE"),
            fits.BinTableHDU(_summary_table(result), name="SUMMARY")]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    logger.info("Saved evaluation to %s", path)
    return path


def main() -> None:
    args = _parser().parse_args()
    benchmark = Config(args.config)
    training = load_training_config(_model_name(benchmark), args.training_config)
    estimators = _estimators(_section(benchmark, "evaluate"), args.baseline)
    result = _run_evaluation(benchmark, training, estimators)
    for key, value in sorted(result.items()):
        if np.ndim(value) == 0:
            logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
