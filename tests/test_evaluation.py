"""Split evaluation, identical observations, and recovery after interruption."""
import json
import numpy as np
import pytest
from astropy.io import fits
from shearnet.config import Config
from shearnet.artifacts import RunDir
from shearnet.artifacts.recovery import Interrupted
from shearnet.training.service import train
from shearnet.evaluation.service import evaluate
from shearnet.evaluation.rendering import EvaluationRenderer
from shearnet.io.catalog_schema import METACAL_TYPES
from shearnet.evaluation.workers import perform
from shearnet.evaluation.identity import digest
from shearnet.io.merge import merge

TINY = {
    "schema_version": 1, "run_options": {"run_name": "tiny", "ncores": 1},
    "simulation": {"backend": "jax-galsim", "noise_sigma": 1., "stamp_size": 21,
                   "jax_fft_size": 64, "jax_batch_size": 2},
    "model": {"type": "mlp", "output_keys": ["g1", "g2"]},
    "training": {"nobj": 40, "epochs": 2, "batch_size": 8},
    "evaluation": {"seed": 7, "nobj": 2, "batch_size": 8,
                   "scenes": [{"name": "plus", "g1": .01, "g2": 0.},
                              {"name": "minus", "g1": -.01, "g2": 0.}],
                   "rotations_deg": [0., 90.]}}


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("split")
    config = Config.from_dict(TINY)
    run = RunDir(root / "training")
    train(config, run)
    sn = evaluate(config, root / "shearnet", estimator="shearnet", run=run, chunk_size=2)
    ng = evaluate(config, root / "ngmix", estimator="ngmix", chunk_size=1)
    return root, config, run, sn, ng


def test_independent_inputs_and_exact_merge(experiment):
    root, _, _, sn, ng = experiment
    output = merge(sn, ng, root / "merged.fits")
    repeat = merge(sn, ng, root / "merged-repeat.fits")
    assert output.read_bytes() == repeat.read_bytes()
    with fits.open(sn) as a, fits.open(ng) as b, fits.open(output) as c:
        assert a["TRUTH"].data.tobytes() == b["TRUTH"].data.tobytes()
        assert a["STAMP"].data.tobytes() == b["STAMP"].data.tobytes()
        assert c["SHEARNET"].data.tobytes() == a["SHEARNET"].data.tobytes()
        assert c["NGMIX"].data.tobytes() == b["NGMIX"].data.tobytes()
        assert c[0].header["SCHEMAV"] == 2
        assert len(np.unique(c["TRUTH"].data["observation_uid"])) == 8
        uids = c["TRUTH"].data["object_uid"].reshape(4, 2)
        assert np.all(uids == uids[0])
        assert "NGMIX" not in a and "SHEARNET" not in b


def _perturbed(source, target, table, column, factor):
    with fits.open(source) as hdul:
        hdul[table].data[column][0] *= factor
        hdul.writeto(target, checksum=True)
    return target


def test_merge_rejects_changed_inputs(experiment):
    root, _, _, sn, ng = experiment
    bad = _perturbed(ng, root / "wrong.fits", "STAMP", "input_fingerprint", 1.01)
    with pytest.raises(ValueError, match="STAMP.input_fingerprint"):
        merge(sn, bad, root / "must-not-exist.fits")
    assert not (root / "must-not-exist.fits").exists()


def test_merge_accepts_roundoff_between_devices(experiment):
    """The ShearNet job renders on the GPU, ngmix on CPUs: the same stamps
    agree to roundoff, and that must not stop the merge."""
    root, _, _, sn, ng = experiment
    near = _perturbed(ng, root / "roundoff.fits", "STAMP", "metacal_fingerprint", 1 + 1e-6)
    near = _perturbed(near, root / "roundoff2.fits", "STAMP", "psf_g", 1 + 1e-9)
    output = merge(sn, near, root / "merged-roundoff.fits")
    with fits.open(output) as c, fits.open(near) as b:
        assert c["STAMP"].data.tobytes() == b["STAMP"].data.tobytes()


def test_worker_count_and_chunk_size_do_not_change_measurements(experiment):
    root, config, _, _, ng = experiment
    parallel = config.with_overrides({"run_options": {"ncores": 2}})
    other = evaluate(parallel, root / "parallel", estimator="ngmix", chunk_size=2)
    with fits.open(ng) as a, fits.open(other) as b:
        for table in ("TRUTH", "STAMP", "NGMIX"):
            for name in a[table].columns.names:
                if name.endswith("cpu_seconds"):
                    continue
                x, y = a[table].data[name], b[table].data[name]
                if x.dtype.kind in "SU":
                    np.testing.assert_array_equal(x, y)
                else:
                    np.testing.assert_array_equal(x, y)


def test_resume_keeps_committed_results(experiment):
    root, config, _, _, ng = experiment
    output = root / "interrupted"

    class StopAfterCommit:
        def check(self):
            if (output / "progress.json").exists():
                raise Interrupted("test interruption")

    with pytest.raises(Interrupted):
        evaluate(config, output, estimator="ngmix", chunk_size=1,
                 checkpoint_seconds=.001, stop=StopAfterCommit())
    assert json.loads((output / "status.json").read_text())["state"] == "paused"
    committed = {p.name: p.read_bytes() for p in (output / "shards").glob("*.fits")}
    assert committed
    resumed = evaluate(config, output, estimator="ngmix", resume=True, chunk_size=2)
    for name, content in committed.items():
        assert (output / "shards" / name).read_bytes() == content
    with fits.open(ng) as a, fits.open(resumed) as b:
        for name in a["NGMIX"].columns.names:
            if not name.endswith("cpu_seconds"):
                np.testing.assert_array_equal(a["NGMIX"].data[name], b["NGMIX"].data[name])


def test_shearnet_preparation_never_fits_em5(monkeypatch):
    import shearnet.methods.ngmix as methods
    config = Config.from_dict(TINY)
    renderer = EvaluationRenderer(config)
    block = renderer.render(.01, 0, indices=np.array([0]))
    observation = renderer.observations(block.galaxy, block.psf)[0]

    def forbidden(*args, **kwargs):
        pytest.fail("ShearNet preparation called a model fitter")

    monkeypatch.setattr(methods, "build_runners", forbidden)
    result = perform(_task([(0, digest("observation"), observation)], "shearnet"))
    assert result["psf_fits"] == 0
    images = result["records"][0]["galaxy"]
    assert images.shape == (10, 21, 21) and np.isfinite(images).all()


def _task(observations, estimator, object_uid=None):
    return dict(object_uid=object_uid or digest("object"), observations=observations,
                scale=.141, step=.01, psf="dilate", estimator=estimator,
                psf_model="em3", gal_model="gauss")


def _observations(config, scenes):
    renderer = EvaluationRenderer(config)
    out = []
    for i, (g1, angle) in enumerate(scenes):
        block = renderer.render(g1, 0, angle, indices=np.array([0]))
        out.append((i, digest(["observation", i]), renderer.observations(block.galaxy,
                                                                            block.psf)[0]))
    return out


def test_each_distinct_psf_is_fitted_once_per_galaxy():
    """The PSF is the same in every scene and station, and metacal makes only 6
    distinct PSF stamps from it, so 3 observations x 10 fits are 6 PSF fits."""
    observations = _observations(Config.from_dict(TINY), [(.01, 0.), (-.01, 0.), (.01, 90.)])
    result = perform(_task(observations, "ngmix"))
    assert result["psf_fits"] == 6 and not result["errors"]
    measured = [m["values"] for m in result["records"].values()]
    for values in measured:
        assert values["flags_original"] == 0 and values["flags_noshear"] == 0
        # the estimate counts every PSF fit where it is used, as a refit would
        assert values["metacal_psf_uncached_cpu_seconds"] > 0
    # the measured columns charge each PSF fit once, to the observation that ran it
    spent = sum(v["fit_psf_cpu_seconds"] + v["metacal_psf_cpu_seconds"] for v in measured)
    np.testing.assert_allclose(spent, result["psf_seconds"])
    assert measured[1]["metacal_psf_cpu_seconds"] == 0


def test_a_shared_psf_fit_does_not_depend_on_which_observation_came_first():
    observations = _observations(Config.from_dict(TINY), [(.01, 0.), (-.01, 0.)])
    alone = perform(_task(observations[1:], "ngmix"))["records"][1]["values"]
    shared = perform(_task(observations, "ngmix"))["records"][1]["values"]
    for name, value in alone.items():
        if not name.endswith("cpu_seconds"):
            np.testing.assert_array_equal(value, shared[name], err_msg=name)


def _assert_all_flagged(result, message):
    from shearnet.evaluation.measurements import FLAG_MISSING

    values = result["records"][0]["values"]
    for variant in ("original",) + METACAL_TYPES:
        assert values[f"flags_{variant}"] == FLAG_MISSING, variant
        assert np.isnan(values[f"g_{variant}"]).all()
    assert len(result["errors"]) == 10
    assert all(message in error for _, error in result["errors"])


def test_a_failing_size_guess_is_a_flagged_row(monkeypatch):
    """Seeds are deterministic, so an error escaping a fit would recur on every
    resume: everything from building the runner on is inside the boundary."""
    import shearnet.methods.ngmix as methods

    def broken(runner, obs):
        raise np.linalg.LinAlgError("singular moments")

    monkeypatch.setattr(methods, "set_tguess", broken)
    observations = _observations(Config.from_dict(TINY), [(.01, 0.)])
    _assert_all_flagged(perform(_task(observations, "ngmix")), "LinAlgError: singular")


def test_a_psf_mixture_that_cannot_be_attached_is_a_flagged_row(monkeypatch):
    import shearnet.methods.ngmix as methods

    class Result(dict):
        def get_gmix(self):
            raise ValueError("bad mixture")

    class Runner:
        guesser = type("Guesser", (), {"T": None})()

        def go(self, obs):
            return Result(flags=0)

    monkeypatch.setattr(methods, "build_runners", lambda *a, **k: (Runner(), Runner()))
    observations = _observations(Config.from_dict(TINY), [(.01, 0.)])
    _assert_all_flagged(perform(_task(observations, "ngmix")), "ValueError: bad mixture")


def test_fingerprints_see_where_the_pixels_are():
    """Sums alone cannot tell an image from its rotation; the projections can."""
    from shearnet.evaluation.identity import fingerprint
    from shearnet.io.merge import agree

    image = _observations(Config.from_dict(TINY), [(.01, 0.)])[0][2].image
    same = fingerprint(image * (1 + 1e-7))
    assert agree("input_fingerprint", fingerprint(image), same)
    for moved in (np.rot90(image), np.fliplr(image), np.roll(image, 1, axis=0)):
        assert not agree("input_fingerprint", fingerprint(image), fingerprint(moved))


def test_merge_tolerance_does_not_depend_on_the_other_rows():
    from shearnet.io.merge import agree

    assert not agree("psf_T_hsm", np.array([1e6, 1.0]), np.array([1e6, 2.0]))
    assert not agree("psf_T_hsm", np.array([1.0]), np.array([2.0]))
    assert agree("psf_T_hsm", np.array([1e6, 1.0]), np.array([1e6, 1.0 + 1e-9]))
    # a stamp sum near zero is judged against its stamp, not its own size
    assert agree("flux_stamp", np.array([0.0]), np.array([1e-3]), np.array([5e4]))
    assert not agree("flux_stamp", np.array([0.0]), np.array([50.0]), np.array([5e4]))


def test_the_catalogs_say_what_the_work_cost(experiment):
    _, _, _, sn, ng = experiment
    with fits.open(sn) as a, fits.open(ng) as b:
        shearnet = {r["key"]: r["value"] for r in a["TIMING"].data}
        ngmix = {r["key"]: r["value"] for r in b["TIMING"].data}
        assert int(shearnet["shearnet_images"]) == 8 * 10
        assert float(shearnet["shearnet_compile_seconds"]) > 0
        assert float(shearnet["shearnet_inference_seconds"]) > 0
        # two galaxies, six distinct PSF stamps each
        assert int(ngmix["ngmix_psf_fits"]) == 12
        assert int(ngmix["failures_raised"]) == 0
        table = b["NGMIX"].data
        assert np.all(table["metacal_psf_cpu_seconds"] < table["metacal_cpu_seconds"])
        assert np.all(table["metacal_psf_uncached_cpu_seconds"]
                      >= table["metacal_psf_cpu_seconds"])
        spent = table["fit_psf_cpu_seconds"].sum() + table["metacal_psf_cpu_seconds"].sum()
        np.testing.assert_allclose(spent, float(ngmix["ngmix_psf_cpu_seconds"]), rtol=1e-6)


def test_resume_rejects_a_changed_seed(experiment):
    root, config, _, _, _ = experiment
    wrong = config.with_overrides({"evaluation": {"seed": 11}})
    with pytest.raises(ValueError, match="resume inputs"):
        evaluate(wrong, root / "ngmix", estimator="ngmix", resume=True)


def test_allocation_kill_preserves_committed_observations(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    config = Config.from_dict(TINY).with_overrides({"evaluation": {"nobj": 8}})
    config_path = tmp_path / "config.yaml"
    config.save(config_path)
    output = tmp_path / "killed"
    args = [sys.executable, "-m", "shearnet.cli.evaluate", "--config", str(config_path),
            "--estimator", "ngmix", "--output", str(output), "--chunk-size", "1",
            "--checkpoint-seconds", ".001", "--quiet"]
    with (tmp_path / "eval.log").open("w") as log:
        process = subprocess.Popen(args, stdout=log, stderr=log, start_new_session=True)
        try:
            deadline = time.monotonic() + 90
            while not (output / "progress.json").exists():
                assert process.poll() is None, (tmp_path / "eval.log").read_text()
                if time.monotonic() >= deadline:
                    pytest.fail("evaluation did not commit a checkpoint within 90 seconds")
                time.sleep(.05)
            os.killpg(process.pid, signal.SIGKILL)
            assert process.wait(timeout=10) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    committed = {p.name: p.read_bytes() for p in (output / "shards").glob("*.fits")}
    assert committed
    result = subprocess.run(args + ["--resume"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    for name, data in committed.items():
        assert (output / "shards" / name).read_bytes() == data
    with fits.open(output / "ngmix.fits") as hdul:
        assert len(hdul["NGMIX"].data) == 32
        assert np.array_equal(hdul["TRUTH"].data["record_id"], np.arange(32))
