"""``shearnet-train`` end to end on a tiny run, as installed."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from shearnet.artifacts import RunDir
from shearnet.cli.train import main

TINY = """\
schema_version: 1
run_options:
  run_name: tiny
simulation:
  noise_sigma: 0.01
  stamp_size: 21
model:
  type: cnn
  output_keys: [g1, g2]
training:
  nobj: 40
  epochs: 2
  batch_size: 8
"""


@pytest.fixture
def tiny(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(TINY)
    return path


def test_a_run_directory_holds_everything(tiny, tmp_path):
    run = RunDir(tmp_path / "run")
    assert main(["--config", str(tiny), "--run", str(run.root)]) == 0

    assert run.state == "completed"
    assert run.config_input.read_text() == TINY
    manifest = json.loads(run.manifest_path.read_text())
    assert manifest["run_name"] == "tiny"
    assert manifest["checkpoint"]["epoch"] in (1, 2)
    from shearnet.artifacts.checkpoints import file_sha256

    assert manifest["checkpoint"]["sha256"] == file_sha256(run.best_params)
    assert manifest["normalizers"]["images"] is None
    labels = np.load(run.label_normalizer)
    assert list(labels["output_keys"]) == ["g1", "g2"]
    history = np.load(run.history_npz)
    assert list(history["epoch"]) == [1, 2]
    assert run.learning_curve.stat().st_size > 0
    assert "TRAINING COMPLETE" in run.train_log.read_text()


def test_the_saved_model_restores_and_predicts(tiny, tmp_path):
    import jax.numpy as jnp

    from shearnet.artifacts.checkpoints import build_model_from_config, init_variables, load_params
    from shearnet.config import Config

    run = RunDir(tmp_path / "run")
    main(["--config", str(tiny), "--run", str(run.root)])
    config = Config.from_file(run.config_resolved)
    model = build_model_from_config(config)
    params = load_params(run.best_params, init_variables(model, config))
    out = model.apply(params, jnp.zeros((3, 21, 21)), output_keys=("g1", "g2"),
                      deterministic=True)
    assert out.shape == (3, 2)


def test_a_finished_run_is_not_overwritten_by_accident(tiny, tmp_path):
    run = tmp_path / "run"
    assert main(["--config", str(tiny), "--run", str(run)]) == 0
    from shearnet.artifacts import RunError

    with pytest.raises(RunError, match="already holds a run"):
        main(["--config", str(tiny), "--run", str(run)])
    assert main(["--config", str(tiny), "--run", str(run), "--overwrite"]) == 0


def test_dry_run_writes_nothing(tiny, tmp_path):
    assert main(["--config", str(tiny), "--run", str(tmp_path / "run"), "--dry-run"]) == 0
    assert not (tmp_path / "run").exists()


def test_bad_inputs_fail_before_any_work(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(TINY + "  epoch: 3\n")
    assert main(["--config", str(path), "--run", str(tmp_path / "run")]) == 2
    path.write_text(TINY.replace("simulation:\n", "simulation:\n  catalogs:\n"
                                 "    train_file: nope.fits\n    eval_file: nope2.fits\n"))
    assert main(["--config", str(path), "--run", str(tmp_path / "run")]) == 2
    path.write_text(TINY.replace("run_options:\n  run_name: tiny\n", ""))
    assert main(["--config", str(path), "--run", str(tmp_path / "run")]) == 2
    assert not (tmp_path / "run").exists()


def test_the_installed_command_runs_from_anywhere(tiny, tmp_path):
    """The console script, from a directory that is not the checkout."""
    env = dict(os.environ, MPLBACKEND="Agg")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "shearnet.cli.train", "--config", str(tiny), "--run",
         str(tmp_path / "run"), "-q"],
        cwd=tmp_path, capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert RunDir(tmp_path / "run").state == "completed"


@pytest.mark.slow
def test_an_inloop_run_records_its_response_terms(tmp_path):
    pytest.importorskip("jax_galsim")
    path = tmp_path / "inloop.yaml"
    path.write_text("""\
schema_version: 1
run_options: {run_name: inloop}
simulation: {backend: jax-galsim, noise_sigma: 0.01, stamp_size: 21, jax_fft_size: 64,
             jax_batch_size: 8}
model: {type: cnn, output_keys: [g1, g2]}
training:
  generation: inloop
  nobj: 48
  epochs: 2
  batch_size: 8
  response: {gamma_weight: 0.01, shift_weight: 0.01}
""")
    run = RunDir(tmp_path / "run")
    assert main(["--config", str(path), "--run", str(run.root)]) == 0
    history = np.load(run.history_npz)
    assert {"response_supervised", "response_gamma", "response_shift"} <= set(history.files)
