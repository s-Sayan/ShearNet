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
from shearnet.methods.anacal import (
    ShapeMeasurement,
    fpfs_shapes,
    leakage,
    measure_fpfs,
    paired_bias,
    renderer_shear_response,
)

logger = get_logger(__name__)

#: Estimators this harness knows how to run.
ESTIMATORS = ("shearnet", "ngmix", "fpfs")


def _parser():
    parser = argparse.ArgumentParser(description="Training-matched shear-bias benchmark")
    parser.add_argument("--task", choices=("m", "psf-leakage", "timing"), required=True)
    parser.add_argument("-c", "--config", required=True, help="Benchmark YAML configuration")
    parser.add_argument(
        "--training-config",
        default=None,
        help="Saved training_config.yaml; resolved from the model name when omitted.",
    )
    parser.add_argument(
        "--estimators",
        default=None,
        help=f"Comma-separated subset of {ESTIMATORS}; default is the config's list.",
    )
    return parser


# ----------------------------------------------------------------------
# config plumbing (the unit-test 'eval' schema and the package schema)
# ----------------------------------------------------------------------
def _eval(config: Config, key: str, default=None):
    value = config.get(f"eval.{key}")
    return config.get(f"evaluation.{key}", default) if value is None else value


def _section(config: Config, task: str) -> dict:
    key = {"m": "bias", "psf-leakage": "leakage", "timing": "timing"}[task]
    return config.get(f"eval.{key}", config.get(f"evaluation.{key}", {})) or {}


def _model_name(config: Config) -> str:
    name = config.get("meta.model_name") or config.get("output.model_name")
    if not name:
        raise ValueError("Benchmark config must set meta.model_name or output.model_name")
    return name


def _estimators(section: dict, override: Optional[str]) -> Sequence[str]:
    requested = (
        [s.strip() for s in override.split(",")]
        if override
        else list(section.get("estimators", ESTIMATORS))
    )
    unknown = sorted(set(requested) - set(ESTIMATORS))
    if unknown:
        raise ValueError(f"unknown estimators {unknown}; choose from {list(ESTIMATORS)}")
    return requested


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
def _measure_callables(renderer, estimators, *, seed, psf_model, gal_model, fpfs_kw, predictor):
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
            obs = renderer.observations_from_stamps(galaxy, psf)
            return fit_shapes(obs, seed=seed, psf_model=psf_model, gal_model=gal_model)[0]

        measures["ngmix"] = ngmix_measure
    if "fpfs" in estimators:

        def fpfs_measure(galaxy, psf):
            return fpfs_shapes(measure_fpfs(galaxy, psf, **fpfs_kw)).e

        measures["fpfs"] = fpfs_measure
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


def _ngmix_metacal_shapes(observations, *, seed, psf_model, gal_model, step):
    """ngmix ellipticity with metacal's image-shearing response, unchanged.

    Kept exactly as ``m/main.py`` runs it, so the number in the metacal column
    is the number that column has always meant.
    """
    from shearnet.methods.ngmix import _get_priors, mp_fit_one_single

    data_list, _ = mp_fit_one_single(
        observations,
        _get_priors(seed),
        np.random.RandomState(seed),
        psf_model=psf_model,
        gal_model=gal_model,
        mcal_pars={"psf": "dilate", "mcal_shear": step},
    )
    n = len(data_list)
    e = np.full((n, 2), np.nan)
    dedg = np.full((n, 2, 2), np.nan)
    flags = np.ones(n, dtype=bool)
    for i, rows in enumerate(data_list):
        by_type = {str(row["shear_type"]): row for row in rows}
        needed = ("noshear", "1p", "1m", "2p", "2m")
        if not all(name in by_type and by_type[name]["flags"] == 0 for name in needed):
            continue
        e[i] = by_type["noshear"]["g"]
        for b, (plus, minus) in enumerate((("1p", "1m"), ("2p", "2m"))):
            dedg[i, :, b] = (by_type[plus]["g"] - by_type[minus]["g"]) / (2.0 * step)
        flags[i] = False
    return ShapeMeasurement(e=e, dedg=dedg, flags=flags)


# ----------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------
def _run_bias(benchmark: Config, training: Config, estimators) -> dict:
    renderer = _renderer(benchmark, training)
    section = _section(benchmark, "m")
    seed = int(_eval(benchmark, "seed"))
    samples = int(_eval(benchmark, "n_obs", benchmark.get("evaluation.test_samples")))
    shear = float(section.get("shear_true", 0.01))
    component = int(section.get("component", 0))
    step = float(section.get("psf_response_step", section.get("metacal_step", 0.01)))
    njac = int(section.get("n_jackknife", 20))
    psf_model = section.get("psf_model", "gauss")
    gal_model = _eval(benchmark, "gal_model", "gauss")
    c_convention = section.get("c_convention", "shearnet")
    resample = section.get("resample", "jackknife")
    needs_obs = "ngmix" in estimators
    noise_sd = _noise_sd(training)
    fpfs_kw = dict(
        pixel_scale=renderer.spec.scale,
        noise_variance=max(noise_sd, 1e-12) ** 2,
        sigma_shapelets=float(section.get("fpfs_sigma_shapelets", 0.52)),
    )

    logger.info(
        "bias benchmark: %d objects, seed %d, shear %+.4f on g%d, backend %s, "
        "generation %s, exp %s, estimators %s",
        samples,
        seed,
        shear,
        component + 1,
        renderer.spec.backend,
        renderer.spec.generation,
        renderer.spec.exp,
        list(estimators),
    )
    plus, minus = renderer.shear_pair(
        samples, seed=seed, shear=shear, component=component, observations=needs_obs
    )
    if plus.psf_images is None:
        raise RuntimeError("Training-matched benchmarks require the PSF channel")

    predictor = (
        SavedModelPredictor(_model_name(benchmark), config=training)
        if "shearnet" in estimators
        else None
    )
    measures = _measure_callables(
        renderer,
        estimators,
        seed=seed,
        psf_model=psf_model,
        gal_model=gal_model,
        fpfs_kw=fpfs_kw,
        predictor=predictor,
    )

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
    }
    # Stratify on measured flux, which is what the paper's binned tables use.
    common = dict(
        component=component,
        njac=njac,
        c_convention=c_convention,
        resample=resample,
        bin_values=np.sum(plus.galaxy_images, axis=(1, 2)),
    )

    # One response pass per population, shared by every estimator.
    shapes = {name: {} for name in measures}
    for label, stamps, sign in (("plus", plus, 1.0), ("minus", minus, -1.0)):
        base = [0.0, 0.0]
        base[component] = sign * shear
        values = {
            name: np.asarray(measure(stamps.galaxy_images, stamps.psf_images), dtype=float)[:, :2]
            for name, measure in measures.items()
        }
        responses = _shared_shear_responses(
            renderer, measures, samples=samples, seed=seed, base_shear=base, step=step
        )
        for name in measures:
            flags = ~np.isfinite(values[name]).all(axis=1)
            shapes[name][label] = ShapeMeasurement(
                e=values[name], dedg=responses[name], flags=flags
            )

    for name in measures:
        result.update(
            _flatten(
                name, paired_bias(shapes[name]["plus"], shapes[name]["minus"], shear, **common)
            )
        )

    # FPFS additionally carries AnaCal's own analytic de/dg. Reporting it beside
    # the shared protocol is a free cross-check on the protocol itself: the two
    # must agree, and if they do not every other column is suspect.
    if "fpfs" in estimators and section.get("fpfs_cross_check", True):
        analytic = {}
        for label, stamps in (("plus", plus), ("minus", minus)):
            analytic[label] = fpfs_shapes(
                measure_fpfs(stamps.galaxy_images, stamps.psf_images, **fpfs_kw)
            )
        result.update(
            _flatten(
                "fpfs_analytic",
                paired_bias(analytic["plus"], analytic["minus"], shear, **common),
            )
        )

    if "ngmix" in estimators and section.get("metacal", True):
        result.update(
            _flatten(
                "ngmix_metacal",
                paired_bias(
                    _ngmix_metacal_shapes(
                        plus.observations,
                        seed=seed + 1,
                        psf_model=psf_model,
                        gal_model=gal_model,
                        step=step,
                    ),
                    _ngmix_metacal_shapes(
                        minus.observations,
                        seed=seed + 2,
                        psf_model=psf_model,
                        gal_model=gal_model,
                        step=step,
                    ),
                    shear,
                    **common,
                ),
            )
        )
    return result


def _run_leakage(benchmark: Config, training: Config, estimators) -> dict:
    renderer = _renderer(benchmark, training)
    section = _section(benchmark, "psf-leakage")
    seed = int(_eval(benchmark, "seed"))
    samples = int(_eval(benchmark, "n_obs", benchmark.get("evaluation.test_samples")))
    step = float(section.get("psf_response_step", 0.01))
    njac = int(section.get("n_jackknife", 20))
    if renderer.spec.backend != "jax-galsim":
        raise ValueError(
            "PSF leakage is measured by finite-differencing the PSF shear, which "
            "needs the jax-galsim backend's explicit psf_g1/psf_g2 parameters. "
            "Retrain or re-render with dataset.backend: jax-galsim."
        )

    stamps = renderer.render(samples, seed=seed)
    result = {
        "backend": renderer.spec.backend,
        "generation": renderer.spec.generation,
        "seed": seed,
        "samples": samples,
        "response_step": step,
    }
    if "shearnet" in estimators:
        predictor = SavedModelPredictor(_model_name(benchmark), config=training)
        measure = predictor.shear_measure()
        psf_response = renderer_shear_response(
            renderer.response_renderer(samples, seed=seed, psf=True), measure, step=step, psf=True
        )
        stats = leakage(measure(stamps.galaxy_images, stamps.psf_images), psf_response, njac=njac)
        result.update({f"shearnet_{k}": v for k, v in stats.items()})
        logger.info(
            "shearnet R^PSF = [[%+.3e %+.3e] [%+.3e %+.3e]] +/- %.1e",
            *stats["psf_response"].ravel(),
            float(np.max(stats["psf_response_err"])),
        )
    if "fpfs" in estimators:
        sigma = float(section.get("fpfs_sigma_shapelets", 0.52))
        noise_sd = _noise_sd(training)

        def measure(galaxy, psf):
            return fpfs_shapes(
                measure_fpfs(
                    galaxy,
                    psf,
                    pixel_scale=renderer.spec.scale,
                    noise_variance=max(noise_sd, 1e-12) ** 2,
                    sigma_shapelets=sigma,
                )
            ).e

        psf_response = renderer_shear_response(
            renderer.response_renderer(samples, seed=seed, psf=True), measure, step=step, psf=True
        )
        stats = leakage(measure(stamps.galaxy_images, stamps.psf_images), psf_response, njac=njac)
        result.update({f"fpfs_{k}": v for k, v in stats.items()})
    return result


def _run_timing(benchmark: Config, training: Config, estimators) -> dict:
    """Wall-clock for the benchmark path only.

    In-loop generation has no separable rendering cost -- the render lives inside
    the training step -- so this times the *benchmark* render (which is always
    up-front, whatever the run trained with) plus inference. The reported
    ``generation`` field records which mode the model trained under so the
    numbers are not mistaken for a training-throughput measurement.
    """
    renderer = _renderer(benchmark, training)
    section = _section(benchmark, "timing")
    seed = int(_eval(benchmark, "seed"))
    samples = int(_eval(benchmark, "n_obs", benchmark.get("evaluation.test_samples")))
    warmup = int(section.get("n_warmup", 1))
    batch = int(section.get("shearnet_batch_size", 4096))

    warm = renderer.render(max(warmup, 1), seed=seed + 10_000)
    predictor = SavedModelPredictor(_model_name(benchmark), config=training)
    predictor(warm.galaxy_images, warm.psf_images, batch_size=batch)  # compile

    start = time.perf_counter()
    stamps = renderer.render(samples, seed=seed)
    render_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictor(stamps.galaxy_images, stamps.psf_images, batch_size=batch)
    inference_seconds = time.perf_counter() - start
    return {
        "backend": renderer.spec.backend,
        "generation": renderer.spec.generation,
        "seed": seed,
        "samples": samples,
        "render_seconds": render_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": render_seconds + inference_seconds,
        "note": "benchmark rendering is always up-front; generation records the training mode",
    }


def _write(config: Config, task: str, result: dict) -> Path:
    section = _section(config, task)
    relative = section.get("output", f"benchmarking/{task}/training_matched.npz")
    root = Path(config.get("paths.root") or config.get("output.plot_path") or ".")
    path = root / relative
    if path.suffix != ".npz":
        path = path.with_name(f"{path.stem}_training_matched.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)
    return path


def main() -> None:
    args = _parser().parse_args()
    benchmark = Config(args.config)
    training = load_training_config(_model_name(benchmark), args.training_config)
    estimators = _estimators(_section(benchmark, args.task), args.estimators)
    runner = {"m": _run_bias, "psf-leakage": _run_leakage, "timing": _run_timing}[args.task]
    result = runner(benchmark, training, estimators)
    path = _write(benchmark, args.task, result)
    logger.info("Saved training-matched %s result to %s", args.task, path)
    for key, value in result.items():
        if np.ndim(value) == 0:
            logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
