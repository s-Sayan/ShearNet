"""End-to-end smoke test of the training-matched benchmark entry point.

Every other test here checks a piece. This one trains a tiny model, writes the
artifacts where a real run writes them, and drives ``research/shear_bias/run.py``
the way the SLURM scripts do -- which is the only way to catch the wiring
between the saved config, the renderer, the estimators and the output file.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("galsim")
pytest.importorskip("anacal")
pytest.importorskip("ngmix")
jax = pytest.importorskip("jax")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "shear_bias"))

import run as harness  # noqa: E402

from shearnet.config.config_handler import Config  # noqa: E402

NPIX, SCALE, FWHM, MODEL = 53, 0.141, 0.5, "smoke_model"


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """A real saved run: checkpoint, normalizer and training_config.yaml."""
    import yaml

    from shearnet.config.config_handler import load_default_config
    from shearnet.core.dataset import generate_dataset
    from shearnet.core.train import save_checkpoint, train_model
    from shearnet.utils.normalization import fit_normalizer, save_normalizer

    tmp_path = tmp_path_factory.mktemp("run")
    cfg = load_default_config()
    cfg["dataset"].update(
        {
            "samples": 32,
            "seed": 3,
            "stamp_size": NPIX,
            "pixel_size": SCALE,
            "psf_fwhm": FWHM,
            "nse_sd": 5.0,
            "exp": "ideal",
            # the evaluation needs jax-galsim for R^PSF; the model itself is
            # backend-agnostic, so this only changes which renderer draws the
            # benchmark stamps
            "backend": "jax-galsim",
        }
    )
    cfg["model"]["output_keys"] = ["g1", "g2", "hlr", "flux"]
    cfg["output"]["model_name"] = MODEL
    cfg["output"]["plot_path"] = str(tmp_path / "plots")
    cfg["output"]["save_path"] = str(tmp_path / "ckpt")
    model_dir = tmp_path / "plots" / MODEL
    model_dir.mkdir(parents=True)
    (tmp_path / "ckpt").mkdir()
    with open(model_dir / "training_config.yaml", "w") as handle:
        yaml.dump(cfg, handle)

    images, labels = generate_dataset(
        32, psf_fwhm=FWHM, npix=NPIX, scale=SCALE, seed=3, nse_sd=5.0,
        output_keys=("g1", "g2", "hlr", "flux"),
    )
    norm = fit_normalizer(labels)
    save_normalizer(norm, str(model_dir / "label_normalizer.npz"))
    state, *_ = train_model(
        images,
        (labels - norm["mean"]) / norm["std"],
        jax.random.PRNGKey(0),
        epochs=1,
        batch_size=8,
        nn="cnn",
        output_keys=("g1", "g2", "hlr", "flux"),
    )
    save_checkpoint(
        state, step=1, checkpoint_dir=str(tmp_path / "ckpt"), model_name=MODEL, overwrite=True
    )

    benchmark = {
        "meta": {"model_name": MODEL},
        "paths": {"root": str(tmp_path / "bench")},
        "eval": {
            "seed": 99,
            "n_obs": 16,
            "gal_model": "gauss",
            "evaluate": {
                "baseline": "both",
                "shear_true": 0.02,
                "n_jackknife": 4,
                "psf_model": "gauss",
                "response_step": 0.01,
                "shearnet_batch_size": 64,
                "output": "evaluation.fits",
            },
        },
    }
    benchmark_path = tmp_path / "benchmark.yaml"
    with open(benchmark_path, "w") as handle:
        yaml.dump(benchmark, handle)
    return Config(str(benchmark_path)), Config(str(model_dir / "training_config.yaml"))


def test_evaluation_measures_every_estimator_under_every_correction(trained_run):
    """One pass, every estimator, every correction it admits, one file.

    The contract: an evaluation always carries ngmix, AnaCal's model fit and
    ShearNet, each under every correction that is defined for it. There is no
    configuration that produces a subset, so this asserts on the whole set
    rather than on whatever happened to be enabled.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, harness.ESTIMATORS)

    assert result["backend"] == "jax-galsim"
    assert result["seed"] == 99 and result["seed"] != training.get("dataset.seed")
    # the pairs the harness can actually measure: metacal shears images so it
    # needs ngmix's bootstrapper, and anacal is analytic so only AnaCal's own
    # fit has it. sim is universal.
    expected = {("ngmix", "metacal"), ("ngmix", "sim"),
                ("anacal", "anacal"), ("anacal", "sim"),
                ("shearnet", "metacal"), ("shearnet", "sim")}
    for estimator, correction in sorted(expected):
        prefix = f"{estimator}_{correction}"
        for field in ("m", "m_err", "c", "c_err"):
            assert np.isfinite(result[f"{prefix}_{field}"]), f"{prefix}_{field}"
        assert result[f"{prefix}_m_err"] > 0.0, prefix
        assert result[f"{prefix}_response"].shape == (2, 2)
        assert result[f"{prefix}_n_used"] > 0
    for estimator in harness.ESTIMATORS:
        assert result[f"{estimator}_leakage_psf_response"].shape == (2, 2)
    assert result["render_seconds"] > 0.0
    assert result["inference_seconds"] > 0.0


def test_the_baseline_switch_always_keeps_shearnet():
    """ShearNet is not optional; the switch chooses what it is measured against.

    Turning off the AnaCal fit is the point of the switch -- its fit is serial
    and costs ~7 h at production N -- so the run must still be a comparison
    after you do.
    """
    assert harness._estimators({}, None) == ["ngmix", "shearnet"]      # default
    assert harness._estimators({}, "ngmix") == ["ngmix", "shearnet"]
    assert harness._estimators({}, "anacal") == ["anacal", "shearnet"]
    assert harness._estimators({}, "both") == ["ngmix", "anacal", "shearnet"]
    assert harness._estimators({"baseline": "anacal"}, None) == ["anacal", "shearnet"]
    # the command line beats the config
    assert harness._estimators({"baseline": "anacal"}, "ngmix") == ["ngmix", "shearnet"]
    for bad in ("fpfs", "shearnet", ""):
        with pytest.raises(ValueError, match="baseline"):
            harness._estimators({}, bad)


def test_shearnet_keeps_metacal_when_ngmix_is_switched_off(trained_run):
    """metacal is a correction, not an estimator.

    ShearNet's metacal response comes from ngmix's bootstrapper, so dropping
    ngmix as a reported baseline must not silently cost ShearNet a column --
    that is exactly the half-result this harness refuses to produce.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, ("anacal", "shearnet"))
    for correction in ("metacal", "sim"):
        assert np.isfinite(result[f"shearnet_{correction}_m"]), correction
    assert np.isfinite(result["anacal_anacal_m"])
    # ngmix contributes no estimator columns. (`ngmix_nproc` stays: it is the
    # worker count metacal ran on, which is metadata about the run either way.)
    for correction in harness.CORRECTIONS:
        assert f"ngmix_{correction}_m" not in result, correction
    assert "ngmix_leakage_psf_response" not in result


def test_fpfs_is_out_of_the_evaluation_but_still_in_the_library():
    """FPFS is not an estimator any more; its measurement code still is."""
    assert "fpfs" not in harness.ESTIMATORS
    with pytest.raises(ValueError, match="unknown baseline"):
        harness._estimators({}, "fpfs")
    from shearnet.methods.anacal import fpfs_shapes, measure_fpfs, resmooth  # noqa: F401


def test_anacal_analytic_response_is_not_a_finite_difference(trained_run):
    """The analytic and simulation responses must be genuinely different code.

    Both estimate de/dgamma for the same fit on the same stamps -- anacal by
    differentiating through it with quintuple numbers, sim by re-rendering the
    scene. They should agree in the mean, which is the point of carrying both;
    they cannot be bit-identical, which would mean one is secretly the other.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, ("ngmix", "anacal", "shearnet"))
    analytic = result["anacal_anacal_response"]
    simulated = result["anacal_sim_response"]
    assert np.isfinite(analytic).all() and np.isfinite(simulated).all()
    assert not np.allclose(analytic, simulated), (analytic, simulated)
    # a well-behaved response on an isotropic population is near-diagonal
    assert abs(analytic[0, 0] - analytic[1, 1]) / abs(analytic[0, 0]) < 0.35, analytic


def test_evaluation_writes_one_fits_with_every_hdu(trained_run):
    from astropy.io import fits

    benchmark, training = trained_run
    harness._run_evaluation(benchmark, training, harness.ESTIMATORS)
    path = Path(benchmark.get("paths.root")) / benchmark.get("eval.evaluate.output")
    assert path.exists(), path

    with fits.open(path) as hdul:
        names = [h.name for h in hdul]
        assert names == ["PRIMARY", "TAB_P", "TAB_M", "LEAKAGE", "SUMMARY"], names
        n = benchmark.get("eval.n_obs")
        for hdu in ("TAB_P", "TAB_M", "LEAKAGE"):
            assert len(hdul[hdu].data) == n, hdu
        for column in ("g_th", "hlr_th", "flux_th", "gpsf", "Tpsf", "s2n",
                       "e_ngmix", "e_anacal", "e_shearnet",
                       "R_ngmix_metacal", "R_ngmix_sim",
                       "R_anacal_anacal", "R_anacal_sim",
                       "R_shearnet_metacal", "R_shearnet_sim",
                       # the network predicts size and flux too; shear_measure()
                       # slices those off, so they need their own forward pass
                       "hlr_shearnet", "flux_shearnet"):
            assert column in hdul["TAB_P"].data.names, (column, hdul["TAB_P"].data.names)
        summary = hdul["SUMMARY"].data
        pairs = set(zip(summary["estimator"], summary["correction"]))
        assert {("ngmix", "metacal"), ("ngmix", "sim"),
                ("anacal", "anacal"), ("anacal", "sim"),
                ("shearnet", "metacal"), ("shearnet", "sim")} <= pairs, pairs


def test_shearnet_metacal_measures_the_reconvolved_images(trained_run):
    """ShearNet's metacal response must come from metacal's own images.

    Not from the scene protocol wearing a different name: the two are different
    derivatives and the whole point of reporting both is that they can disagree.
    The check is that the network's metacal response is NOT identical to its
    anacal one, and that both are finite.
    """
    benchmark, training = trained_run
    result = harness._run_evaluation(benchmark, training, harness.ESTIMATORS)
    metacal = result["shearnet_metacal_response"]
    anacal = result["shearnet_sim_response"]
    assert np.isfinite(metacal).all() and np.isfinite(anacal).all()
    assert not np.allclose(metacal, anacal), (metacal, anacal)


def test_the_backend_requirement_fails_before_the_expensive_part(trained_run, monkeypatch):
    """A galsim-backend run cannot measure R^PSF, so it must not start at all."""
    import yaml

    from shearnet.config.config_handler import Config

    benchmark, training = trained_run
    path = Path(training.config["output"]["plot_path"]) / MODEL / "galsim_config.yaml"
    with open(path, "w") as handle:
        yaml.dump({**training.config, "dataset": {**training.get("dataset"), "backend": "galsim"}}, handle)

    import shearnet.benchmarking as bm

    calls = {"n": 0}
    real = bm.TrainingMatchedRenderer.render
    monkeypatch.setattr(
        bm.TrainingMatchedRenderer, "render",
        lambda self, *a, **kw: (calls.__setitem__("n", calls["n"] + 1), real(self, *a, **kw))[1],
    )
    with pytest.raises(ValueError, match="jax-galsim"):
        harness._run_evaluation(benchmark, Config(str(path)), harness.ESTIMATORS)
    assert calls["n"] == 0, "it rendered before deciding it could not finish"


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        harness._estimators({}, "metacal")
    assert harness._estimators({"baseline": "ngmix"}, None) == ["ngmix", "shearnet"]
