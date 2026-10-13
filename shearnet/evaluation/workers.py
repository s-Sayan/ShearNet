"""CPU work on one object at a time, with no random state shared between objects.

A task is one catalog galaxy and every observation of it still to measure (all
scenes and ring stations). Grouping by galaxy is what makes the ngmix job
affordable: the PSF is not rotated and does not depend on the applied shear, so
the 10 PSF fits per observation (original + 9 metacal products) are fits to only
6 distinct images per galaxy -- the original PSF, the dilated reconvolution PSF
the five galaxy-shear products share, and the four sheared ones. Each distinct
image is fitted once per galaxy, seeded from the image itself, so the answer is
the one a refit would give and does not depend on which observation asked
first. With em5 the PSF fits are ~94% of ngmix's time, and 5 scenes x 4
stations x 10 fits = 200 of them reduce to 6.
"""
import hashlib
import multiprocessing as mp
import queue
import time
import traceback

import numpy as np

from ..io.catalog_schema import METACAL_TYPES
from ..parallel import cpu_only_children
from .identity import fingerprint, rng
from .measurements import FIT_FIELDS, empty_fit, fit_galaxy, measure_psf


def _psf_key(psf_obs):
    h = hashlib.sha256()
    for array in (psf_obs.image, psf_obs.weight, psf_obs.jacobian.get_data()):
        a = np.ascontiguousarray(array)
        h.update(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def _failure(exc):
    return f"{type(exc).__name__}: {exc}"[:300]


class _ObjectPSFs:
    """PSF moments and PSF fits of one galaxy, each done once per distinct stamp."""

    def __init__(self, object_uid, scale, psf_model, gal_model):
        self.object_uid, self.scale = object_uid, scale
        self.psf_model, self.gal_model = psf_model, gal_model
        self.moments, self.fits = {}, {}
        self.fit_seconds = 0.0

    def measure(self, psf_obs):
        key = _psf_key(psf_obs)
        if key not in self.moments:
            seed = int(rng(self.object_uid, "moments").randint(0, 2**31))
            moments = measure_psf(np.asarray([psf_obs.image]), self.scale, seed=seed)
            self.moments[key] = {name: values[0] for name, values in moments.items()}
        return self.moments[key]

    def fit(self, psf_obs):
        """``(result, cpu_seconds, ran_now, error)`` of the one fit to this stamp.

        ``ran_now`` is True only for the call that ran the fit; ``result`` is
        None and ``error`` says why when the fit raised.
        """
        from ..methods.ngmix import build_runners

        key = _psf_key(psf_obs)
        if key in self.fits:
            result, seconds, error = self.fits[key]
            return result, seconds, False, error
        start = time.process_time()
        try:
            _, psf_runner = build_runners(rng(self.object_uid, "psf:" + key),
                                          psf_model=self.psf_model, gal_model=self.gal_model)
            result, error = psf_runner.go(obs=psf_obs), None
        except Exception as exc:  # ngmix and numba raise a zoo of errors on bad stamps
            result, error = None, "PSF fit: " + _failure(exc)
        seconds = time.process_time() - start
        self.fits[key] = (result, seconds, error)
        self.fit_seconds += seconds
        return result, seconds, True, error


def _measure(variant, target, obs, uid, psfs):
    """One galaxy fit of one product, never raising.

    Returns ``(row, psf_seconds, psf_ran_now, error)``. Every step -- the PSF
    fit, building the runner, the size guess, attaching the PSF, the fit -- is
    inside the one failure boundary: seeds are deterministic, so an error
    escaping here would recur on every resume and stop the stage for good.
    """
    from ..methods.ngmix import build_runners, set_tguess

    psf_result, psf_seconds, ran_now, error = psfs.fit(target.psf)
    start = time.process_time()
    row = empty_fit()
    if error is None and psf_result["flags"] == 0:
        try:
            runner, _ = build_runners(rng(uid, "fit:" + variant),
                                      psf_model=psfs.psf_model, gal_model=psfs.gal_model)
            # every product is guessed from the original stamp's size, as SuperBIT does
            set_tguess(runner, obs)
            row = fit_galaxy(runner, target, psf_result)
        except Exception as exc:
            row, error = empty_fit(), _failure(exc)
    row["cpu_seconds"] = time.process_time() - start
    return row, psf_seconds, ran_now, error


def _metacal_products(obs, object_uid, step, psf):
    import ngmix

    start = time.process_time()
    try:
        products, error = ngmix.metacal.get_all_metacal(
            obs, rng=rng(object_uid, "fixnoise"), step=step, psf=psf,
            types=list(METACAL_TYPES)), None
    except Exception as exc:  # a hopeless stamp is nine flagged products, not a crash
        products, error = {}, "metacal: " + _failure(exc)
    return products, time.process_time() - start, error


def _ngmix_values(obs, products, uid, psfs, preparation):
    values, errors = {}, []
    spent = {"fit": 0.0, "metacal": preparation}
    psf_spent = {"fit": 0.0, "metacal": 0.0}
    psf_uncached = {"fit": 0.0, "metacal": 0.0}
    for variant in ("original",) + METACAL_TYPES:
        stage = "fit" if variant == "original" else "metacal"
        target = obs if variant == "original" else products.get(variant)
        if target is None:
            row = empty_fit()
        else:
            row, psf_seconds, ran_now, error = _measure(variant, target, obs, uid, psfs)
            if error:
                errors.append(f"{variant}: {error}")
            spent[stage] += row.pop("cpu_seconds") + (psf_seconds if ran_now else 0.0)
            psf_spent[stage] += psf_seconds if ran_now else 0.0
            psf_uncached[stage] += psf_seconds
        for name in FIT_FIELDS:
            values[f"{name}_{variant}"] = row[name]
    for stage in ("fit", "metacal"):
        values[f"{stage}_cpu_seconds"] = spent[stage]
        values[f"{stage}_psf_cpu_seconds"] = psf_spent[stage]
        values[f"{stage}_psf_uncached_cpu_seconds"] = psf_uncached[stage]
    values["metacal_image_cpu_seconds"] = preparation
    return values, errors


def perform(task):
    """Measure every observation in ``task``: one galaxy, many scenes/stations.

    Fit failures are flagged rows; ``errors`` lists them as
    ``(record, message)`` so the job can report what went wrong.
    """
    psfs = _ObjectPSFs(task["object_uid"], task["scale"], task["psf_model"],
                       task["gal_model"])
    records, errors = {}, []
    nan = np.full(4, np.nan, dtype=np.float32)
    for record, uid, obs in task["observations"]:
        products, preparation, error = _metacal_products(obs, task["object_uid"],
                                                         task["step"], task["psf"])
        if error:
            errors.append((record, error))
        result = {
            "moments": psfs.measure(obs.psf),
            "input_fingerprint": np.stack([fingerprint(obs.image), fingerprint(obs.psf.image)]),
            "metacal_fingerprint": np.stack([
                np.stack([fingerprint(products[t].image), fingerprint(products[t].psf.image)])
                if t in products else np.stack([nan, nan]) for t in METACAL_TYPES]),
        }
        if task["estimator"] == "shearnet":
            variants = [obs] + [products.get(t) for t in METACAL_TYPES]
            result["galaxy"] = np.stack([
                v.image if v is not None else np.full_like(obs.image, np.nan)
                for v in variants]).astype(np.float32)
            result["psf"] = np.stack([
                v.psf.image if v is not None else np.full_like(obs.psf.image, np.nan)
                for v in variants]).astype(np.float32)
        else:
            result["values"], fit_errors = _ngmix_values(obs, products, uid, psfs, preparation)
            errors += [(record, message) for message in fit_errors]
        records[record] = result
    return {"records": records, "errors": errors, "psf_fits": len(psfs.fits),
            "psf_seconds": psfs.fit_seconds}


def _worker(inputs, outputs):
    while True:
        item = inputs.get()
        if item is None:
            return
        key, task = item
        try:
            outputs.put((key, perform(task), None))
        except BaseException:
            outputs.put((key, None, traceback.format_exc()))
            return


class Workers:
    """A bounded process pool that reports dead workers and can stop promptly."""

    def __init__(self, count):
        self.count = count

    def __enter__(self):
        ctx = mp.get_context("spawn")
        self.inputs = ctx.Queue(maxsize=2 * self.count)
        self.outputs = ctx.Queue(maxsize=2 * self.count)
        self.processes = [ctx.Process(target=_worker, args=(self.inputs, self.outputs))
                          for _ in range(self.count)]
        with cpu_only_children():
            for process in self.processes:
                process.start()
        return self

    def map(self, tasks, stop):
        tasks = iter(tasks)
        pending = 0
        exhausted = False
        while pending or not exhausted:
            stop.check()
            while pending < 2 * self.count and not exhausted:
                try:
                    item = next(tasks)
                except StopIteration:
                    exhausted = True
                else:
                    self.inputs.put(item)
                    pending += 1
            if not pending:
                break
            try:
                key, result, error = self.outputs.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode is not None for p in self.processes):
                    raise RuntimeError("evaluation worker exited before returning its result")
                yield None
                continue
            pending -= 1
            if error:
                raise RuntimeError(f"evaluation worker failed on object {key}:\n{error}")
            yield key, result

    def __exit__(self, *args):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for q in (self.inputs, self.outputs):
            q.cancel_join_thread()
            q.close()
