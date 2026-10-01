"""Run directories, exact checkpoints, and the training history."""

import json

import numpy as np
import pytest

from shearnet.artifacts import RunDir, RunError
from shearnet.artifacts.checkpoints import (
    build_model_from_config,
    init_variables,
    load_params,
    model_kwargs,
    save_params,
)
from shearnet.config import Config
from shearnet.training.history import History


def _config(**model):
    return Config.from_dict({"run_options": {"run_name": "t"}, "model": model})


# ----------------------------------------------------------------------
# the run directory
# ----------------------------------------------------------------------
def test_a_new_run_directory_has_the_layout(tmp_path):
    run = RunDir(tmp_path / "run")
    run.create()
    for path in (run.model_dir, run.normalizers_dir, run.training_dir, run.logs_dir):
        assert path.is_dir()


def test_an_occupied_run_directory_is_refused_unless_overwritten(tmp_path):
    run = RunDir(tmp_path / "run")
    run.create()
    run.write_status("completed")
    (run.evaluations_dir / "default").mkdir(parents=True)
    with pytest.raises(RunError, match="already holds a run \\(completed\\)"):
        RunDir(tmp_path / "run").create()
    RunDir(tmp_path / "run").create(overwrite=True)
    assert not run.evaluations_dir.exists()   # evaluations must not survive run replacement
    assert run.state is None


def test_status_records_its_history(tmp_path):
    run = RunDir(tmp_path)
    run.write_status("running")
    started = run.read_status()["started"]
    run.write_status("failed", error="boom")
    status = run.read_status()
    assert status["state"] == "failed" and status["error"] == "boom"
    assert status["started"] == started and "finished" in status
    with pytest.raises(ValueError):
        run.write_status("bogus")


def test_only_a_completed_run_can_be_evaluated(tmp_path):
    run = RunDir(tmp_path / "run")
    with pytest.raises(RunError, match="no run directory"):
        run.require_completed()
    run.create()
    run.write_status("running")
    with pytest.raises(RunError, match="status: running"):
        run.require_completed()
    run.write_status("completed")
    with pytest.raises(RunError, match="is missing"):
        run.require_completed()


def test_evaluation_names_are_directories_not_paths(tmp_path):
    run = RunDir(tmp_path)
    assert run.evaluation("cut").catalog_path("fid").name == "fid_cut.fits"
    with pytest.raises(ValueError):
        run.evaluation("../x")


# ----------------------------------------------------------------------
# checkpoints
# ----------------------------------------------------------------------
def test_params_round_trip_exactly(tmp_path):
    config = _config(type="cnn")
    model = build_model_from_config(config)
    variables = init_variables(model, config, seed=3)
    digest = save_params(variables, tmp_path / "best.msgpack")
    assert len(digest) == 64
    restored = load_params(tmp_path / "best.msgpack", init_variables(model, config, seed=9))
    import jax

    for a, b in zip(jax.tree_util.tree_leaves(variables), jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_another_architecture_does_not_load(tmp_path):
    """Loading rejects a checkpoint with a different multiscale-block setting."""
    common = dict(type="d4-fork-like", galaxy_branch="shearnet-d4", psf_branch="shearnet-d4",
                  d4_features=[4, 8, 8], d_model=8, num_heads=2, ffn_dim=8)
    with_block = _config(**common)
    without = _config(**common, d4_multiscale=False)
    model = build_model_from_config(with_block)
    save_params(init_variables(model, with_block), tmp_path / "best.msgpack")
    other = build_model_from_config(without)
    with pytest.raises(ValueError, match="does not hold this architecture"):
        load_params(tmp_path / "best.msgpack", init_variables(other, without))


def test_training_and_restoring_build_the_same_model():
    """model_kwargs is the one map from config to build_model; TrainConfig agrees."""
    from shearnet.core.specs import TrainConfig

    config = Config.from_file("configs/paper/fiducial.yaml")
    tc = TrainConfig.from_config(config)
    rename = {"nn": "nn", "galaxy_type": "galaxy_type", "psf_type": "psf_type"}
    for key, value in model_kwargs(config).items():
        assert getattr(tc, rename.get(key, key)) == value, key


# ----------------------------------------------------------------------
# history
# ----------------------------------------------------------------------
def test_history_keeps_sparse_validation_sparse(tmp_path):
    history = History(("g1", "g2"), tmp_path / "h.csv", tmp_path / "h.npz")
    history({"epoch": 1, "train_loss": 2.0, "val_loss": None, "val_per_key": None,
             "best": False, "seconds": 1.0})
    history({"epoch": 2, "train_loss": 1.0, "val_loss": 1.5, "val_per_key": [1.0, 2.0],
             "best": True, "seconds": 1.0, "response": {"psf": 0.1}})
    assert history.best_epoch == 2 and history.best_val_loss == 1.5
    saved = np.load(tmp_path / "h.npz")
    assert np.isnan(saved["val_loss"][0]) and saved["val_loss"][1] == 1.5
    assert list(saved["output_keys"]) == ["g1", "g2"]
    back = History.read_csv(tmp_path / "h.csv", ("g1", "g2"))
    assert back.records[0]["val_loss"] is None
    assert back.records[1]["response_psf"] == pytest.approx(0.1)
    assert back.best_epoch == 2


def test_learning_curve_is_a_readable_png(tmp_path):
    import matplotlib.image

    from shearnet.training.curves import plot_learning_curve

    history = History(("g1",))
    for epoch in range(1, 5):
        history({"epoch": epoch, "train_loss": 1.0 / epoch,
                 "val_loss": 1.2 / epoch if epoch % 2 == 0 else None,
                 "val_per_key": None, "best": epoch == 4, "seconds": 0.1})
    plot_learning_curve(history, tmp_path / "curve.png")
    image = matplotlib.image.imread(tmp_path / "curve.png")
    assert image.ndim == 3 and image.shape[0] > 100


def test_manifest_json_is_plain_json(tmp_path):
    run = RunDir(tmp_path)
    run.write_manifest({"a": 1, "path": tmp_path})
    assert json.loads(run.manifest_path.read_text())["path"] == str(tmp_path)
