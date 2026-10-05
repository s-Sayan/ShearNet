"""``shearnet-eval``: one raw catalog per evaluation, nothing derived in it."""

import json

import numpy as np
import pytest

pytest.importorskip("jax_galsim")
pytest.importorskip("ngmix")

from astropy.io import fits  # noqa: E402

from shearnet.artifacts import RunDir  # noqa: E402
from shearnet.cli.evaluate import main as eval_main  # noqa: E402
from shearnet.cli.train import main as train_main  # noqa: E402
from shearnet.evaluation.measurements import METACAL_TYPES  # noqa: E402
from shearnet.evaluation.service import metacal_seed  # noqa: E402

TINY = """\
schema_version: 1
run_options: {run_name: tiny, ncores: 1}
simulation: {backend: jax-galsim, noise_sigma: 0.05, stamp_size: 21, jax_fft_size: 64,
             jax_batch_size: 8}
model: {type: fork-like, galaxy_branch: cnn, psf_branch: cnn, output_keys: [g1, g2, hlr]}
training: {nobj: 40, epochs: 1, batch_size: 8}
evaluation:
  seed: 7
  nobj: 3
  batch_size: 16
  rotations_deg: [0, 90]
"""

N, SCENES, STATIONS = 3, 5, 2


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    config = root / "tiny.yaml"
    config.write_text(TINY)
    assert train_main(["--config", str(config), "--run", str(root / "run"), "-q"]) == 0
    return RunDir(root / "run")


@pytest.fixture(scope="module")
def catalog(run):
    assert eval_main(["--run", str(run.root), "-q"]) == 0
    return run.evaluation("default").catalog_path("tiny")


def test_one_catalog_with_every_extension(catalog, run):
    with fits.open(catalog) as hdul:
        names = [h.name for h in hdul]
        assert names[:5] == ["PRIMARY", "TRUTH", "STAMP", "SHEARNET", "NGMIX"]
        for extra in ("SCHEMA", "SCENES", "ROTATIONS", "BLOCKS", "PSF_FILES", "PROTOCOL",
                      "CONFIG", "PROVENANCE"):
            assert extra in names
        header = hdul[0].header
        assert header["NRECORD"] == N * SCENES * STATIONS
        assert header["CKPTSHA"] == run.read_manifest()["checkpoint"]["sha256"]
        for name in ("TRUTH", "STAMP", "SHEARNET", "NGMIX"):
            assert len(hdul[name].data) == N * SCENES * STATIONS
    assert run.evaluation("default").state == "completed"


def test_the_status_records_how_long_the_evaluation_took(catalog, run):
    """A chunk loop's offset once shadowed the start time, so the status said
    the evaluation had taken ~56 years."""
    status = run.evaluation("default").read_status()
    assert 0 < status["seconds"] < 3600


def test_rows_are_scene_major_then_station_then_object(catalog):
    truth = fits.getdata(catalog, "TRUTH")
    np.testing.assert_array_equal(truth["record_id"], np.arange(len(truth)))
    np.testing.assert_array_equal(truth["catalog_row"], np.tile(np.arange(N), SCENES * STATIONS))
    np.testing.assert_array_equal(truth["scene_id"], np.repeat(np.arange(SCENES), N * STATIONS))
    np.testing.assert_array_equal(truth["rotation_id"],
                                  np.tile(np.repeat(np.arange(STATIONS), N), SCENES))
    for name in ("STAMP", "SHEARNET", "NGMIX"):
        np.testing.assert_array_equal(fits.getdata(catalog, name)["record_id"],
                                      truth["record_id"])


def test_truth_separates_source_shape_applied_shear_and_label(catalog):
    truth = fits.getdata(catalog, "TRUTH")
    scenes = fits.getdata(catalog, "SCENES")
    for s, scene in enumerate(scenes):
        rows = truth[truth["scene_id"] == s]
        np.testing.assert_allclose(rows["g_applied"], [[scene["g1"], scene["g2"]]] * len(rows))
    zero = truth[(truth["scene_id"] == 0)]
    # no applied shear: the pre-PSF shape IS the source shape
    np.testing.assert_allclose(zero["e_prepsf"], zero["e_source"], atol=1e-12)
    # a 90-degree station negates the spin-2 source shape, object by object
    r0, r90 = zero[zero["rotation_id"] == 0], zero[zero["rotation_id"] == 1]
    np.testing.assert_allclose(r90["e_source"], -r0["e_source"], atol=1e-12)
    np.testing.assert_allclose(truth["label_g1"], truth["e_prepsf"][:, 0], rtol=1e-6, atol=1e-7)


def test_the_psf_is_the_same_in_every_scene_and_station(catalog):
    stamp = fits.getdata(catalog, "STAMP")
    first = stamp[:N]
    for block in range(SCENES * STATIONS):
        np.testing.assert_array_equal(stamp["psf_g"][block * N:(block + 1) * N], first["psf_g"])


def test_every_variant_of_every_estimator_is_there(catalog):
    shearnet = fits.getdata(catalog, "SHEARNET").columns.names
    ngmix = fits.getdata(catalog, "NGMIX").columns.names
    for variant in ("original",) + METACAL_TYPES:
        assert {f"g_{variant}", f"hlr_{variant}", f"flags_{variant}"} <= set(shearnet)
        assert {f"g_{variant}", f"g_cov_{variant}", f"T_{variant}", f"Tpsf_{variant}",
                f"flux_{variant}", f"s2n_{variant}", f"flags_{variant}"} <= set(ngmix)
    assert fits.getdata(catalog, "NGMIX")["g_cov_noshear"].shape[1:] == (2, 2)


def test_nothing_derived_is_in_the_catalog(catalog):
    """No responses, biases, leakage fits, corrections or summary tables."""
    import re

    forbidden = re.compile(r"(^|_)(R|Rgamma|Rpsf|Rbarpsf|m|c|alpha|beta|weight|corrected|"
                           r"ring|selected|mask)(_|$)", re.IGNORECASE)
    with fits.open(catalog) as hdul:
        names = [h.name for h in hdul]
        for table in ("SUMMARY", "BINNED", "LEAKSUM", "LEAKAGE", "TAB_P", "TAB_M"):
            assert table not in names
        for name in ("TRUTH", "STAMP", "SHEARNET", "NGMIX"):
            for column in hdul[name].columns.names:
                assert not forbidden.search(column), (name, column)


def test_the_catalog_carries_its_own_provenance(catalog, run):
    config = {row["key"]: row["value"] for row in fits.getdata(catalog, "CONFIG")}
    assert "run_name: tiny" in config["training"]
    assert "rotations_deg: [0.0, 90.0]" in config["evaluation"]
    provenance = {row["key"]: row["value"] for row in fits.getdata(catalog, "PROVENANCE")}
    assert json.loads(provenance["training_manifest"])["checkpoint"]["sha256"] == \
        run.read_manifest()["checkpoint"]["sha256"]
    blocks = fits.getdata(catalog, "BLOCKS")
    assert len(blocks) == SCENES * STATIONS
    assert list(blocks["metacal_seed"][:2]) == [7 + 100, 7 + 101]


def test_a_second_evaluation_leaves_the_model_and_the_first_alone(catalog, run, tmp_path):
    before = run.best_params.read_bytes(), catalog.read_bytes()
    override = tmp_path / "small.yaml"
    override.write_text("evaluation:\n  nobj: 2\n  rotations_deg: [0]\n"
                        "  estimators: [shearnet]\n  metacal: {shearnet: false}\n")
    assert eval_main(["--run", str(run.root), "--config", str(override),
                      "--eval-name", "small", "-q"]) == 0
    assert (run.best_params.read_bytes(), catalog.read_bytes()) == before
    small = run.evaluation("small").catalog_path("tiny")
    with fits.open(small) as hdul:
        assert "NGMIX" not in [h.name for h in hdul]
        assert len(hdul["SHEARNET"].data) == 2 * SCENES
        assert "g_noshear" not in hdul["SHEARNET"].columns.names
    # the same name again is refused before any work
    assert eval_main(["--run", str(run.root), "--config", str(override),
                      "--eval-name", "small", "-q"]) == 2


def test_an_override_cannot_change_the_model(run, tmp_path):
    override = tmp_path / "bad.yaml"
    override.write_text("model:\n  type: cnn\n")
    assert eval_main(["--run", str(run.root), "--config", str(override), "-q"]) == 2


def test_an_unfinished_run_is_refused(tmp_path):
    run = RunDir(tmp_path / "run")
    run.create()
    run.write_status("running")
    assert eval_main(["--run", str(run.root), "-q"]) == 2


def test_dry_run_writes_nothing(run):
    assert eval_main(["--run", str(run.root), "--eval-name", "dry", "--dry-run"]) == 0
    assert not run.evaluation("dry").root.exists()


def test_metacal_seeds_reproduce_the_old_harness():
    zero, plus, minus = ({"g1": 0.0, "g2": 0.0}, {"g1": 0.01, "g2": 0.0},
                         {"g1": 0.0, "g2": -0.01})
    assert [metacal_seed(150, zero, k) for k in range(4)] == [250, 251, 252, 253]
    assert metacal_seed(150, plus, 3) == 151 and metacal_seed(150, minus, 0) == 152


def test_psf_moments_are_reproducible():
    import galsim

    from shearnet.evaluation.measurements import measure_psf

    psf = galsim.Gaussian(fwhm=0.5).shear(g1=0.05, g2=-0.02)
    stamps = np.stack([psf.drawImage(nx=21, ny=21, scale=0.141).array] * 3)
    a, b = measure_psf(stamps, 0.141, seed=3), measure_psf(stamps, 0.141, seed=3)
    np.testing.assert_array_equal(a["psf_g"], b["psf_g"])
    assert a["psf_T_admom"][0] > a["psf_T_hsm"][0]   # trace > determinant size when elliptical


def test_a_failed_fit_is_a_flagged_row_not_a_missing_one():
    import ngmix

    from shearnet.evaluation.measurements import FLAG_MISSING, fit_original

    jac = ngmix.DiagonalJacobian(row=10, col=10, scale=0.141)
    psf = ngmix.Observation(np.zeros((21, 21)), weight=np.ones((21, 21)), jacobian=jac)
    obs = ngmix.Observation(np.zeros((21, 21)), weight=np.ones((21, 21)), jacobian=jac, psf=psf)
    fit = fit_original([obs], seed=1, psf_model="gauss", gal_model="gauss", nproc=1)
    assert fit["flags"][0] != 0
    assert np.isnan(fit["g"][0]).all()
    assert FLAG_MISSING > 0
