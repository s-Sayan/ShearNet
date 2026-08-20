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
        }
    )
    cfg["model"]["output_keys"] = ["g1", "g2"]
    cfg["output"]["model_name"] = MODEL
    cfg["output"]["plot_path"] = str(tmp_path / "plots")
    cfg["output"]["save_path"] = str(tmp_path / "ckpt")
    model_dir = tmp_path / "plots" / MODEL
    model_dir.mkdir(parents=True)
    (tmp_path / "ckpt").mkdir()
    with open(model_dir / "training_config.yaml", "w") as handle:
        yaml.dump(cfg, handle)

    images, labels = generate_dataset(
        32, psf_fwhm=FWHM, npix=NPIX, scale=SCALE, seed=3, nse_sd=5.0, output_keys=("g1", "g2")
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
            "bias": {
                "estimators": ["shearnet", "fpfs"],
                "fpfs_cross_check": False,
                "shear_true": 0.02,
                "n_jackknife": 4,
                "psf_model": "gauss",
                "metacal_step": 0.01,
                "output": "bias.npz",
            },
            "leakage": {"estimators": ["shearnet"], "n_jackknife": 4, "output": "leak.npz"},
            "timing": {"n_warmup": 2, "output": "timing.npz"},
        },
    }
    benchmark_path = tmp_path / "benchmark.yaml"
    with open(benchmark_path, "w") as handle:
        yaml.dump(benchmark, handle)
    return Config(str(benchmark_path)), Config(str(model_dir / "training_config.yaml"))


def test_bias_task_reports_every_estimator_with_errors(trained_run):
    benchmark, training = trained_run
    result = harness._run_bias(benchmark, training, ("shearnet", "fpfs"))

    assert result["backend"] == "galsim"
    assert result["generation"] == "upfront"
    assert result["seed"] == 99 and result["seed"] != training.get("dataset.seed")
    for prefix in ("shearnet", "fpfs"):
        for field in ("m", "m_err", "c", "c_err"):
            value = result[f"{prefix}_{field}"]
            assert np.isfinite(value), f"{prefix}_{field} is not finite"
        assert result[f"{prefix}_m_err"] > 0.0, f"{prefix} has no uncertainty"
        assert result[f"{prefix}_response"].shape == (2, 2)
        assert result[f"{prefix}_n_used"] > 0
        # binned outputs are first-class, not an afterthought
        assert len(result[f"{prefix}_bin_m"]) == len(result[f"{prefix}_bin_count"])

    # FPFS on a known applied shear should land near zero even at N=16, because
    # the pair-matched estimator cancels shape noise.
    assert abs(result["fpfs_m"]) < 0.05, result["fpfs_m"]
    assert result["exp"] == "ideal"


def test_every_estimator_shares_one_response_pass(trained_run, monkeypatch):
    """The +/- re-renders are identical across estimators; render them once.

    Rendering per estimator multiplies the dominant cost by the number of table
    columns. At SuperBIT's ~40 ms/stamp that is the difference between a 36 and
    a 142 core-hour run.
    """
    import shearnet.benchmarking as bm

    benchmark, training = trained_run
    calls = {"n": 0}
    real = bm.TrainingMatchedRenderer.render

    def counting(self, *a, **kw):
        calls["n"] += 1
        return real(self, *a, **kw)

    monkeypatch.setattr(bm.TrainingMatchedRenderer, "render", counting)
    harness._run_bias(benchmark, training, ("shearnet", "fpfs"))
    # 2 populations (via shear_pair) + 2 axes x 2 signs x 2 populations = 10,
    # independent of how many estimators are in the table.
    assert calls["n"] == 10, calls["n"]


def test_bias_task_writes_a_readable_result(trained_run, tmp_path):
    benchmark, training = trained_run
    result = harness._run_bias(benchmark, training, ("fpfs",))
    path = harness._write(benchmark, "m", result)
    assert path.exists()
    with np.load(path, allow_pickle=False) as saved:
        assert float(saved["fpfs_m"]) == pytest.approx(result["fpfs_m"])
        assert saved["fpfs_response"].shape == (2, 2)


def test_leakage_task_needs_the_jax_backend(trained_run):
    benchmark, training = trained_run
    with pytest.raises(ValueError, match="jax-galsim"):
        harness._run_leakage(benchmark, training, ("shearnet",))


def test_timing_task_separates_render_from_inference(trained_run):
    benchmark, training = trained_run
    result = harness._run_timing(benchmark, training, ("shearnet",))
    assert result["render_seconds"] > 0.0
    assert result["inference_seconds"] > 0.0
    assert result["total_seconds"] == pytest.approx(
        result["render_seconds"] + result["inference_seconds"]
    )
    # in-loop training has no separable render cost; the field records which
    # mode trained the model so the number is not misread as throughput
    assert result["generation"] == "upfront"


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown estimators"):
        harness._estimators({}, "shearnet,metacal")
    assert harness._estimators({"estimators": ["fpfs"]}, None) == ["fpfs"]
    assert harness._estimators({}, None) == list(harness.ESTIMATORS)


@pytest.mark.slow
def test_ngmix_paths_run_and_agree_between_the_two_responses(trained_run):
    """metacal shears the image, the protocol shears the scene: same derivative.

    They are estimating the same R, so a gross disagreement means one of the two
    is wired wrong. The tolerance is loose because this runs at N=16 -- it is a
    wiring check, not a calibration measurement.
    """
    benchmark, training = trained_run
    result = harness._run_bias(benchmark, training, ("ngmix",))
    assert np.isfinite(result["ngmix_m"]) and result["ngmix_m_err"] > 0
    assert np.isfinite(result["ngmix_metacal_m"])
    protocol = result["ngmix_response"][0, 0]
    metacal = result["ngmix_metacal_response"][0, 0]
    assert protocol > 0 and metacal > 0
    assert abs(protocol - metacal) / metacal < 0.5, (protocol, metacal)
