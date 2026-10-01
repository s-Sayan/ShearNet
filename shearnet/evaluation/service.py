"""Measure a finished run and write one raw catalog.

``shearnet-eval`` renders every configured scene at every configured ring
station from the evaluation catalog, measures each stamp with every configured
estimator, and writes what they measured -- nothing more -- to one FITS file:

* ShearNet's prediction on the original stamp and on the nine metacal products,
* ngmix's fit of the original stamp and of the nine metacal products,
* the PSF moments and stamp-level observables,
* the truth each stamp was drawn from,
* the protocol, the configs and the provenance.

No response, bias, leakage slope, selection, weight or correction is computed
here; those are derived from this file, elsewhere. A failed fit is a flagged
row, not a missing one.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..artifacts import provenance
from ..artifacts.runs import EvaluationDir, RunDir, atomic_write
from ..config import Config
from ..io import catalog_schema as schema
from ..io.fits_catalog import CatalogWriter, key_value_hdu, numeric_hdu
from ..logging_utils import get_logger, run_log
from ..parallel import resolve_nproc
from .measurements import (
    METACAL_TYPES,
    NGMIX_CHUNK,
    fit_original,
    measure_psf,
    metacal,
    stamp_observables,
)
from .rendering import EvaluationRenderer, quarter_turns

logger = get_logger(__name__)

__all__ = ["evaluate", "metacal_seed", "plan"]


def metacal_seed(base_seed: int, scene: Dict, rotation_index: int) -> int:
    """The seed of the metacal fits of one scene at one station.

    Positive applied shear uses
    ``seed + 1``, negative ``seed + 2``, the zero-shear scene ``seed + 100 +
    station``. It only seeds ngmix's initial guesses.
    """
    total = scene["g1"] + scene["g2"]
    if scene["g1"] == 0.0 and scene["g2"] == 0.0:
        return base_seed + 100 + rotation_index
    return base_seed + (1 if total > 0 else 2)


def plan(config: Config) -> Dict:
    """How much work an evaluation is, before any of it is done."""
    n = config.get("evaluation.nobj")
    scenes = config.get("evaluation.scenes")
    rotations = config.get("evaluation.rotations_deg")
    estimators = config.get("evaluation.estimators")
    runs_metacal = "ngmix" in estimators or config.get("evaluation.metacal.shearnet")
    blocks = len(scenes) * len(rotations)
    return {
        "objects": n,
        "scenes": [s["name"] for s in scenes],
        "rotations_deg": rotations,
        "blocks": blocks,
        "records": n * blocks,
        "estimators": estimators,
        "metacal_fits": n * blocks if runs_metacal else 0,
        "ngmix_original_fits": n * blocks if "ngmix" in estimators else 0,
    }


def evaluate(run: RunDir, *, override: Optional[Path] = None, eval_name: str = "default",
             overwrite: bool = False) -> Path:
    """Evaluate ``run`` into ``<run>/evaluations/<eval_name>/``; returns the catalog path."""
    run.require_completed()
    config = Config.from_file(run.config_resolved)
    if override is not None:
        config = config.evaluation_override(override)
    edir = run.evaluation(eval_name)
    edir.create(overwrite=overwrite)
    if override is not None:
        atomic_write(edir.config_input, Path(override).read_bytes())
    config.save(edir.config_resolved)
    edir.write_status("running")
    with run_log(edir.log):
        try:
            path = _evaluate(config, run, edir)
        except BaseException as exc:
            edir.write_status("failed", error=f"{type(exc).__name__}: {exc}")
            logger.error("evaluation failed: %s", exc)
            raise
    return path


def _evaluate(config: Config, run: RunDir, edir: EvaluationDir) -> Path:
    from .predictor import RunPredictor

    start = time.time()
    run_name = config.get("run_options.run_name")
    work = plan(config)
    estimators = work["estimators"]
    want_shearnet = "shearnet" in estimators
    want_ngmix = "ngmix" in estimators
    shearnet_metacal = config.get("evaluation.metacal.shearnet")
    runs_metacal = want_ngmix or shearnet_metacal
    scenes = config.get("evaluation.scenes")
    rotations = config.get("evaluation.rotations_deg")
    n = work["objects"]
    seed = config.get("evaluation.seed")
    batch = config.get("evaluation.batch_size")
    nproc = config.get("run_options.ncores")
    ngmix_kw = dict(psf_model=config.get("evaluation.ngmix.psf_model"),
                    gal_model=config.get("evaluation.ngmix.gal_model"), nproc=nproc)
    step = config.get("evaluation.metacal.step")
    reconv = config.get("evaluation.metacal.psf")

    logger.info("evaluating %s (%s) -> %s", run_name, run.root, edir.root)
    logger.info("%d objects x %d scenes (%s) x %d stations (%s) = %d records; measuring %s%s",
                n, len(scenes), ", ".join(work["scenes"]), len(rotations),
                ", ".join(f"{r:g}" for r in rotations), work["records"], ", ".join(estimators),
                "; ngmix metacal on every record" if runs_metacal else "")
    if runs_metacal:
        logger.info("ngmix workers: %d", resolve_nproc(nproc, n_tasks=n))

    renderer = EvaluationRenderer(config)
    predictor = RunPredictor(run, config) if want_shearnet else None
    output_keys = tuple(config.get("model.output_keys"))

    tables = {"TRUTH": schema.TRUTH_COLUMNS + [
        schema.Column(f"label_{key}", "f8", description=f"the training target {key} for this "
                      "stamp, physical units") for key in output_keys],
        "STAMP": schema.STAMP_COLUMNS}
    if want_shearnet:
        tables["SHEARNET"] = schema.shearnet_columns(output_keys, metacal=shearnet_metacal)
    if want_ngmix:
        tables["NGMIX"] = schema.ngmix_columns(metacal=True)
    writer = CatalogWriter(edir.scratch, tables, work["records"])

    blocks: List[Dict] = []
    psf_reference = psf_moments = None
    for s, scene in enumerate(scenes):
        for k, rotation in enumerate(rotations):
            t0 = time.time()
            index = s * len(rotations) + k
            rows = slice(index * n, (index + 1) * n)
            keys = {
                "record_id": np.arange(rows.start, rows.stop, dtype=np.int64),
                "catalog_row": np.arange(n, dtype=np.int64),
                "scene_id": np.full(n, s, dtype=np.int16),
                "rotation_id": np.full(n, k, dtype=np.int16),
            }
            block = renderer.render(scene["g1"], scene["g2"], rotation)
            rendered = time.time() - t0

            truth = dict(keys, rotation_deg=np.full(n, float(rotation)), **block.truth)
            for i, key in enumerate(output_keys):
                truth[f"label_{key}"] = block.labels[:, i]
            writer.fill("TRUTH", rows, truth)

            # the PSF does not depend on the scene or the station; measure it once
            # and check that is still true rather than assume it
            if psf_reference is None or not np.array_equal(block.psf, psf_reference):
                psf_moments = measure_psf(block.psf, renderer.cfg.scale, seed=seed)
                psf_reference = block.psf
            writer.fill("STAMP", rows, dict(keys, **psf_moments,
                                            **stamp_observables(block.galaxy,
                                                                config.get("simulation.noise_sigma"))))

            if want_shearnet:
                writer.fill("SHEARNET", rows, dict(
                    keys, **_shearnet_values(predictor(block.galaxy, block.psf, batch),
                                             output_keys, "original")))
            if want_ngmix:
                writer.fill("NGMIX", rows, keys)

            mseed = metacal_seed(seed, scene, k)
            failures = {"ngmix_original": 0, "ngmix_metacal": 0}
            for start in range(0, n, NGMIX_CHUNK) if (want_ngmix or runs_metacal) else ():
                stop = min(start + NGMIX_CHUNK, n)
                chunk = slice(rows.start + start, rows.start + stop)
                if want_ngmix:
                    fit = fit_original(renderer.observations(block.galaxy[start:stop],
                                                             block.psf[start:stop]),
                                       seed=seed, **ngmix_kw)
                    failures["ngmix_original"] += int(np.count_nonzero(fit["flags"]))
                    writer.fill("NGMIX", chunk, _ngmix_values(fit, "original"))
                if runs_metacal:
                    images = want_shearnet and shearnet_metacal
                    fits, gal_stack, psf_stack = metacal(
                        renderer.observations(block.galaxy[start:stop], block.psf[start:stop]),
                        seed=mseed, step=step, psf=reconv, return_images=images, **ngmix_kw)
                    bad = np.zeros(stop - start, dtype=bool)
                    for t in METACAL_TYPES:
                        bad |= fits[t]["flags"] != 0
                        if want_ngmix:
                            writer.fill("NGMIX", chunk, _ngmix_values(fits[t], t))
                    failures["ngmix_metacal"] += int(bad.sum())
                    if images:
                        m, npix = gal_stack.shape[0], gal_stack.shape[-1]
                        preds = predictor(gal_stack.reshape(m * 9, npix, npix),
                                          psf_stack.reshape(m * 9, npix, npix), batch)
                        preds = preds.reshape(m, 9, len(output_keys))
                        values = {}
                        for j, t in enumerate(METACAL_TYPES):
                            values.update(_shearnet_values(preds[:, j, :], output_keys, t))
                        writer.fill("SHEARNET", chunk, values)
                        del gal_stack, psf_stack

            seconds = time.time() - t0
            blocks.append({"scene_id": s, "rotation_id": k, "first_record": rows.start,
                           "n_records": n, "metacal_seed": mseed if runs_metacal else -1,
                           "ngmix_seed": seed if want_ngmix else -1,
                           "noise_quarter_turns": quarter_turns(rotation),
                           "seconds": round(seconds, 1)})
            logger.info(
                "block %d/%d: scene %s, rotation %g: rendered in %.0f s, measured in %.0f s%s",
                index + 1, work["blocks"], scene["name"], rotation, rendered,
                seconds - rendered,
                "" if not runs_metacal else
                f"; ngmix flags: {failures['ngmix_original']} original, "
                f"{failures['ngmix_metacal']} metacal")

    path = edir.catalog_path(run_name)
    header = _primary_header(config, run, edir, work)
    extra = _metadata_hdus(config, run, renderer, blocks, scenes, rotations)
    writer.write(path, header, extra)
    writer.cleanup()
    seconds = round(time.time() - start, 1)
    edir.write_status("completed", catalog=path.name, records=work["records"], seconds=seconds)
    logger.info("")
    logger.info("EVALUATION COMPLETE")
    logger.info("Checkpoint: %s", run.read_manifest()["checkpoint"]["sha256"])
    logger.info("Records:    %d (failed fits are flagged, not dropped)", work["records"])
    logger.info("FITS:       %s", path)
    logger.info("Scientific analysis: performed separately")
    return path


def _shearnet_values(preds: np.ndarray, output_keys, variant: str) -> Dict[str, np.ndarray]:
    values = {}
    finite = np.isfinite(preds).all(axis=1)
    if "g1" in output_keys:
        i1, i2 = output_keys.index("g1"), output_keys.index("g2")
        values[f"g_{variant}"] = preds[:, [i1, i2]]
    for i, key in enumerate(output_keys):
        if key not in ("g1", "g2"):
            values[f"{key}_{variant}"] = preds[:, i]
    values[f"flags_{variant}"] = (~finite).astype(np.int32)
    return values


def _ngmix_values(fit: Dict[str, np.ndarray], variant: str) -> Dict[str, np.ndarray]:
    return {f"{name}_{variant}": value for name, value in fit.items()}


def _primary_header(config: Config, run: RunDir, edir: EvaluationDir, work: Dict) -> Dict:
    import jax

    manifest = run.read_manifest()
    return {
        "SCHEMA": (schema.SCHEMA_NAME, "catalog layout"),
        "SCHEMAV": (schema.SCHEMA_VERSION, "catalog layout version"),
        "RUNNAME": (config.get("run_options.run_name"), "training run"),
        "EVALNAME": (edir.name, "evaluation name"),
        "CKPTSHA": (manifest["checkpoint"]["sha256"], ""),
        "CKPTEPO": (manifest["checkpoint"]["epoch"], "epoch of the saved model"),
        "NOBJ": (work["objects"], "objects per scene and station"),
        "NSCENE": (len(work["scenes"]), "scenes (SCENES table)"),
        "NROT": (len(work["rotations_deg"]), "ring stations (ROTATIONS table)"),
        "NRECORD": (work["records"], "rows of every per-record table"),
        "BACKEND": (config.get("simulation.backend"), "renderer"),
        "PSFMODE": (config.get("simulation.psf.mode"), "ideal or superbit"),
        "X64": (bool(jax.config.jax_enable_x64), "JAX_ENABLE_X64 at render time"),
        "MCALSTEP": (config.get("evaluation.metacal.step"), "one-sided metacal step"),
        "MCALPSF": (config.get("evaluation.metacal.psf"), "metacal reconvolution PSF"),
        "DATE": (time.strftime("%Y-%m-%dT%H:%M:%S"), "written"),
    }


def _metadata_hdus(config, run, renderer, blocks, scenes, rotations) -> list:
    training = run.config_resolved.read_text()
    evaluation = config.to_yaml()
    return [
        numeric_hdu("SCENES", {
            "scene_id": np.arange(len(scenes)),
            "name": np.array([s["name"] for s in scenes]),
            "g1": np.array([s["g1"] for s in scenes]),
            "g2": np.array([s["g2"] for s in scenes]),
        }),
        numeric_hdu("ROTATIONS", {
            "rotation_id": np.arange(len(rotations)),
            "rotation_deg": np.asarray(rotations, dtype=float),
            "noise_quarter_turns": np.array([quarter_turns(r) for r in rotations]),
        }),
        numeric_hdu("BLOCKS", {key: np.array([b[key] for b in blocks]) for key in blocks[0]}),
        numeric_hdu("PSF_FILES", {
            "psf_file_id": np.arange(max(len(renderer.psf_files), 1)),
            "path": np.array(renderer.psf_files or ["<ideal Gaussian>"]),
        }),
        key_value_hdu("PROTOCOL", {
            "row_order": "scene-major, then ring station, then catalog row",
            "pairing": "every scene and station renders the same catalog rows with the same "
                       "offsets, PSFs and noise (noise turned with the station by quarter turns)",
            "ring": "source shape and offset rotated by rotation_deg; PSF not rotated; "
                    "labels recomposed",
            "applied_shear": "GalSim .shear(g) after the source shape: reduced shear, area "
                             "preserving, no convergence or magnification",
            "metacal": f"ngmix MetacalBootstrapper, psf={config.get('evaluation.metacal.psf')}, "
                       f"step={config.get('evaluation.metacal.step')} (products are "
                       f"2*step apart), types={list(METACAL_TYPES)}",
            "metacal_seeds": "BLOCKS.metacal_seed; zero-shear scene seed+100+station, "
                             "positive shear seed+1, negative seed+2",
            "ngmix_fit": f"Fitter(model={config.get('evaluation.ngmix.gal_model')}) with the "
                         "GPriorBA/CenPrior/flat priors of shearnet.methods.ngmix, PSF "
                         f"{config.get('evaluation.ngmix.psf_model')}",
            "ngmix_chunk": NGMIX_CHUNK,
            "noise_sigma": config.get("simulation.noise_sigma"),
            "pixel_scale_arcsec": config.get("simulation.pixel_scale"),
            "eval_catalog": config.get("simulation.catalogs.eval_file"),
            "eval_catalog_sha256": _sha256(config.get("simulation.catalogs.eval_file")),
            "evaluation_seed": config.get("evaluation.seed"),
            "training_seed": config.get("training.seed"),
        }),
        key_value_hdu("CONFIG", {"training": training, "evaluation": evaluation}),
        key_value_hdu("PROVENANCE", {**provenance.collect(),
                                     "training_manifest": run.read_manifest()}),
    ]


def _sha256(path) -> Optional[str]:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
