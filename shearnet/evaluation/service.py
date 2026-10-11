"""Run one estimator, committing completed observations throughout the job."""
from pathlib import Path
import time
import numpy as np

from ..artifacts import provenance
from ..artifacts.runs import atomic_write, write_json
from ..artifacts.recovery import Interrupted, StopRequest, exclusive
from ..io import catalog_schema as schema
from ..io.fits_catalog import CatalogWriter, numeric_hdu, key_value_hdu
from ..logging_utils import get_logger
from ..parallel import resolve_nproc
from .identity import digest, file_digest, implementation, observation_recipe
from .journal import Journal
from .rendering import EvaluationRenderer, quarter_turns
from .measurements import stamp_observables
from .workers import Workers

logger = get_logger(__name__)


#: Shard-header keys of the work spent on a shard's records (see TIMING).
TIMING_KEYS = ("RENDSEC", "INFSEC", "INFIMG", "CMPSEC", "PSFFITS", "PSFSEC", "FITERR")


def plan(config):
    n = config.get("evaluation.nobj")
    scenes = config.get("evaluation.scenes")
    rotations = config.get("evaluation.rotations_deg")
    return {"objects": n, "scenes": [s["name"] for s in scenes],
            "rotations_deg": rotations, "records": n * len(scenes) * len(rotations)}


def columns(config, estimator):
    outputs = config.get("model.output_keys")
    tables = {"TRUTH": schema.TRUTH_COLUMNS + [
        schema.Column(f"label_{k}", "f8", description=f"training target {k}") for k in outputs],
        "STAMP": schema.STAMP_COLUMNS}
    metacal = config.get("evaluation.metacal.shearnet")
    tables[estimator.upper()] = (schema.ngmix_columns() if estimator == "ngmix" else
                                 schema.shearnet_columns(outputs, metacal=metacal))
    return tables


def evaluate(config, output, *, estimator, run=None, resume=False, chunk_size=None,
             checkpoint_seconds=300, stop=None):
    """Measure a standalone estimator catalog; ``output`` owns its shards.

    ``chunk_size`` galaxies are rendered at a time, in every scene and station,
    and handed to the workers one galaxy at a time (default: the renderer's
    ``simulation.jax_batch_size``, so no render batch is padded).
    """
    if estimator not in ("shearnet", "ngmix"):
        raise ValueError("choose shearnet or ngmix")
    chunk_size = chunk_size or config.get("simulation.jax_batch_size")
    if chunk_size < 1 or checkpoint_seconds <= 0:
        raise ValueError("chunk size and checkpoint interval must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output / ".writer.lock"), StopRequest() as signals:
        stop = stop or signals
        status = output / "status.json"
        write_json(status, {"state": "running"})
        began = time.monotonic()
        try:
            path = _evaluate(config, output, estimator, run, resume, chunk_size,
                             checkpoint_seconds, stop)
        except BaseException as exc:
            write_json(status, {"state": "paused" if isinstance(exc, Interrupted) else "failed",
                                "error": str(exc)})
            raise
        write_json(status, {"state": "completed", "catalog": str(path),
                            "attempt_seconds": time.monotonic() - began})
        return path


def _evaluate(config, output, estimator, run, resume, chunk_size, interval, stop):
    import jax
    from .predictor import RunPredictor

    if not jax.config.jax_enable_x64:
        raise ValueError("evaluation requires JAX_ENABLE_X64=1 in both jobs")
    if estimator not in config.get("evaluation.estimators"):
        raise ValueError(f"{estimator} is not enabled in evaluation.estimators")
    predictor = None
    checkpoint = None
    if estimator == "shearnet":
        if run is None:
            raise ValueError("ShearNet evaluation requires --run")
        run.require_completed()
        checkpoint = file_digest(run.best_params)
        if checkpoint != run.read_manifest()["checkpoint"]["sha256"]:
            raise ValueError("model checkpoint differs from the training manifest")
        predictor = RunPredictor(run, config)
    renderer = EvaluationRenderer(config)
    identity = {"observation": renderer.recipe, "implementation": implementation(),
                "estimator": estimator,
                "measurement": config.get("evaluation.ngmix") if estimator == "ngmix" else
                {"checkpoint": checkpoint, "model": config.get("model"),
                 "normalizers": [file_digest(p) for p in
                                 (run.label_normalizer, run.image_normalizer) if p.exists()]}}
    tables = columns(config, estimator)
    work = plan(config)
    journal = Journal(output / "shards", identity, tables, work["records"], resume=resume)
    date_path = output / "created.txt"
    if not date_path.exists():
        atomic_write(date_path, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()).encode())
    config.save(output / "config.resolved.yaml")
    atomic_write(output / "identity.json", (str(journal.signature) + "\n").encode())
    n = work["objects"]
    scenes = config.get("evaluation.scenes")
    rotations = work["rotations_deg"]
    output_keys = tuple(config.get("model.output_keys"))
    shearnet_metacal = config.get("evaluation.metacal.shearnet")
    variants = ("original",) + (schema.METACAL_TYPES if shearnet_metacal else ())
    workers = resolve_nproc(config.get("run_options.ncores"), n_tasks=n)
    batch = config.get("evaluation.batch_size")
    settings = dict(scale=renderer.cfg.scale, step=config.get("evaluation.metacal.step"),
                    psf=config.get("evaluation.metacal.psf"), estimator=estimator,
                    psf_model=config.get("evaluation.ngmix.psf_model"),
                    gal_model=config.get("evaluation.ngmix.gal_model"))

    # work done since the last commit, written into the next shard's header so
    # the catalog's TIMING sums over exactly the committed records
    spent = dict.fromkeys(TIMING_KEYS, 0.0)
    if predictor is not None:
        tick = time.perf_counter()
        predictor.compile([1], batch, renderer.cfg.npix, renderer.cfg.npix)
        spent["CMPSEC"] = time.perf_counter() - tick
        logger.info("ShearNet compiled in %.0f s, kept out of the inference time",
                    spent["CMPSEC"])
    buffer, rows, predicted = [], {}, []
    last_commit = time.monotonic()

    def commit():
        nonlocal last_commit
        if buffer:
            journal.commit(buffer, timing=spent)
            buffer.clear()
            spent.update(dict.fromkeys(TIMING_KEYS, 0.0))
            done = int(journal.done.sum())
            write_json(output / "progress.json", {"completed": done, "total": work["records"],
                                                   "checkpoint_unix": time.time()})
            logger.info("%s: saved %d/%d observations", estimator, done, work["records"])
        last_commit = time.monotonic()

    def tasks():
        """One task per galaxy: every observation of it not yet committed."""
        for first in range(0, n, chunk_size):
            catalog_rows = np.arange(first, min(n, first + chunk_size))
            pending = {int(r): [] for r in catalog_rows}
            for s, scene in enumerate(scenes):
                for k, angle in enumerate(rotations):
                    stop.check()
                    offset = (s * len(rotations) + k) * n
                    indices = catalog_rows[~journal.done[offset + catalog_rows]]
                    if not len(indices):
                        continue
                    tick = time.perf_counter()
                    block = renderer.render(scene["g1"], scene["g2"], angle, indices)
                    spent["RENDSEC"] += time.perf_counter() - tick
                    observations = renderer.observations(block.galaxy, block.psf)
                    stamps = stamp_observables(block.galaxy,
                                               config.get("simulation.noise_sigma"))
                    for j, catalog_row in enumerate(indices):
                        record = offset + int(catalog_row)
                        object_uid = digest([renderer.identity, int(catalog_row)])
                        uid = digest([object_uid, scene["g1"], scene["g2"], angle])
                        keys = dict(record_id=record, catalog_row=catalog_row, scene_id=s,
                                    rotation_id=k, object_uid=object_uid,
                                    observation_uid=uid)
                        truth = dict(keys, rotation_deg=angle,
                                     **{name: values[j] for name, values in block.truth.items()
                                        if name not in keys})
                        truth.update({f"label_{name}": block.labels[j, i]
                                      for i, name in enumerate(output_keys)})
                        rows[record] = {
                            "TRUTH": truth,
                            "STAMP": dict(keys, **{name: values[j]
                                                   for name, values in stamps.items()}),
                            estimator.upper(): dict(keys)}
                        pending[int(catalog_row)].append((record, uid, observations[j]))
            for catalog_row, observations in pending.items():
                if observations:
                    object_uid = digest([renderer.identity, catalog_row])
                    yield catalog_row, dict(settings, object_uid=object_uid,
                                            observations=observations)

    def finish_predictions():
        if not predicted:
            return
        galaxies = np.concatenate([result["galaxy"][:len(variants)] for _, result in predicted])
        psfs = np.concatenate([result["psf"][:len(variants)] for _, result in predicted])
        tick = time.perf_counter()
        predictions = predictor(galaxies, psfs, batch)
        spent["INFSEC"] += time.perf_counter() - tick
        spent["INFIMG"] += len(galaxies)
        predictions = predictions.reshape(len(predicted), len(variants), -1)
        for (record, _), values in zip(predicted, predictions):
            row = rows.pop(record)
            for j, variant in enumerate(variants):
                row["SHEARNET"].update(_shearnet_values(values[j], output_keys, variant))
            buffer.append(row)
        predicted.clear()

    try:
        with Workers(workers) as pool:
            for completed in pool.map(tasks(), stop):
                if completed is not None:
                    _, result = completed
                    spent["PSFFITS"] += result["psf_fits"]
                    spent["PSFSEC"] += result["psf_seconds"]
                    spent["FITERR"] += len(result["errors"])
                    for record, message in result["errors"]:
                        logger.warning("record %d: flagged, %s", record, message)
                    for record, measured in result["records"].items():
                        row = rows[record]
                        row["STAMP"].update(
                            measured["moments"],
                            input_fingerprint=measured["input_fingerprint"],
                            metacal_fingerprint=measured["metacal_fingerprint"])
                        if predictor is None:
                            row["NGMIX"].update(measured["values"])
                            buffer.append(rows.pop(record))
                        else:
                            predicted.append((record, measured))
                due = time.monotonic() - last_commit >= interval
                if len(predicted) * len(variants) >= batch or due:
                    finish_predictions()
                if len(buffer) >= 2048 or due:
                    commit()
    finally:
        # save completed observations even when a worker or the allocation fails
        if predictor is not None:
            finish_predictions()
        commit()
    stop.check()
    if not journal.done.all():
        raise RuntimeError("evaluation ended with missing observations")
    if observation_recipe(config, renderer.psf_files) != renderer.recipe:
        raise ValueError("evaluation inputs changed during the job")
    if checkpoint and file_digest(run.best_params) != checkpoint:
        raise ValueError("model checkpoint changed during evaluation")
    writer = CatalogWriter(output / ".assembly", tables, work["records"])
    for path in journal.paths:
        with journal.open(path) as hdul:
            ids = hdul["TRUTH"].data["record_id"]
            for name in tables:
                writer.fill(name, ids, {c.name: hdul[name].data[c.name] for c in tables[name]})
    header = {"SCHEMA": (schema.SCHEMA_NAME, "catalog layout"),
              "SCHEMAV": (schema.SCHEMA_VERSION, "catalog layout version"),
              "RUNNAME": (config.get("run_options.run_name"), "run"),
              "EVALNAME": (estimator, "evaluation"),
              "DATE": (date_path.read_text(), "evaluation created, UTC"),
              "BACKEND": (f"jax-galsim/{jax.devices()[0].platform}", "renderer and device"),
              "PSFMODE": (config.get("simulation.psf.mode"), "PSF source"),
              "X64": (True, "rendering precision"),
              "ESTIMATR": (estimator, "estimator"), "OBSSHA": (renderer.identity, ""),
              "IMPLSHA": (digest(identity["implementation"]), ""),
              "NOBJ": (n, "objects per scene and rotation"),
              "NRECORD": (work["records"], "complete observations"),
              "NSCENE": (len(scenes), "scenes"), "NROT": (len(rotations), "rotations"),
              "MCALSTEP": (config.get("evaluation.metacal.step"), "one-sided shear"),
              "MCALPSF": (config.get("evaluation.metacal.psf"), "reconvolution PSF")}
    if checkpoint:
        header["CKPTSHA"] = (checkpoint, "")
        header["CKPTEPO"] = (run.read_manifest()["checkpoint"]["epoch"], "model epoch")
    totals = journal.timing(TIMING_KEYS)
    extra = metadata(config, renderer, run, workers, identity) + [key_value_hdu("TIMING", {
        "device": jax.devices()[0].platform,
        "render_seconds": totals["RENDSEC"],
        "shearnet_inference_seconds": totals["INFSEC"],
        "shearnet_images": int(totals["INFIMG"]),
        "shearnet_compile_seconds": totals["CMPSEC"],
        "ngmix_workers": workers,
        "ngmix_psf_fits": int(totals["PSFFITS"]),
        "failures_raised": int(totals["FITERR"]),
        "ngmix_psf_cpu_seconds": totals["PSFSEC"],
        "note": "seconds summed over every attempt's committed work; compilation is kept "
                "out of the inference time. ngmix fitted each distinct PSF stamp once per "
                "galaxy: ngmix_psf_fits fits took ngmix_psf_cpu_seconds, which the NGMIX "
                "*_psf_cpu_seconds columns sum to; *_psf_uncached_cpu_seconds estimate a "
                "refit everywhere. failures_raised counts measurements that raised and were "
                "flagged (each logged with its error)"})]
    path = writer.write(output / f"{estimator}.fits", header, extra)
    writer.cleanup()
    return path


def _shearnet_values(prediction, output_keys, variant):
    values = {f"flags_{variant}": int(not np.isfinite(prediction).all())}
    if "g1" in output_keys:
        values[f"g_{variant}"] = prediction[[output_keys.index("g1"), output_keys.index("g2")]]
    for j, key in enumerate(output_keys):
        if key not in ("g1", "g2"):
            values[f"{key}_{variant}"] = prediction[j]
    return values


def metadata(config, renderer, run, workers, identity):
    scenes = config.get("evaluation.scenes")
    rotations = config.get("evaluation.rotations_deg")
    blocks = len(scenes) * len(rotations)
    return [
        numeric_hdu("SCENES", {"scene_id": np.arange(len(scenes)),
                               "name": np.array([s["name"] for s in scenes]),
                               "g1": np.array([s["g1"] for s in scenes]),
                               "g2": np.array([s["g2"] for s in scenes])}),
        numeric_hdu("ROTATIONS", {
            "rotation_id": np.arange(len(rotations)),
            "rotation_deg": np.array(rotations),
            "noise_quarter_turns": np.array([quarter_turns(a) for a in rotations])}),
        numeric_hdu("BLOCKS", {"scene_id": np.repeat(np.arange(len(scenes)), len(rotations)),
                               "rotation_id": np.tile(np.arange(len(rotations)), len(scenes)),
                               "first_record": np.arange(blocks) * renderer.nobj,
                               "n_records": np.full(blocks, renderer.nobj),
                               "ngmix_workers": np.full(blocks, workers)}),
        numeric_hdu("PSF_FILES", {"psf_file_id": np.arange(max(1, len(renderer.psf_files))),
                                  "path": np.array(renderer.psf_files or ["<ideal Gaussian>"])}),
        key_value_hdu("PROTOCOL", {
            "version": 2, "identity": identity,
            "row_order": "scene, rotation, catalog row",
            "pairing": "every scene and station of a galaxy shares its offset, PSF and noise "
                       "(the noise turned with the station by quarter turns) and its metacal "
                       "fixnoise field; schema 1 drew a separate fixnoise field per scene",
            "metacal": "ngmix.get_all_metacal; independent per-observation fitting streams",
            "psf_fits": "each distinct PSF stamp of a galaxy is fitted once, seeded from the "
                        "stamp, and that fit serves every observation that has it",
            "ngmix_workers": workers,
            "timing": "see the TIMING table"}),
        key_value_hdu("CONFIG", {"evaluation": config.to_yaml(),
                                "training": run.config_resolved.read_text() if run else ""}),
        key_value_hdu("PROVENANCE", {**provenance.collect(),
                                     "training_manifest": run.read_manifest() if run else None}),
    ]
