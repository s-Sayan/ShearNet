"""research/hyperparam_search/sweep.py: write the grid, then rank the runs."""

import importlib.util
import json
from pathlib import Path

import pytest

from shearnet.config import Config, ConfigError

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "research" / "hyperparam_search"


def _module():
    spec = importlib.util.spec_from_file_location("sweep", SWEEP / "sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _module()


def _spec(tmp_path, grid):
    path = tmp_path / "spec.yaml"
    path.write_text(json.dumps({"base_config": str(REPO / "configs" / "smoke.yaml"),
                                "name_prefix": "t", "grid": grid}))
    return path


def test_combos_grid_and_random():
    grid = {"a": [1, 2, 3], "b": [4, 5]}
    assert len(sweep.combos(grid)) == 6
    assert sweep.combos(grid, "random", 4, seed=1) == sweep.combos(grid, "random", 4, seed=1)
    assert len({tuple(p.values()) for p in sweep.combos(grid, "random", 4)}) == 4
    with pytest.raises(ValueError):
        sweep.combos(grid, "bayes")


def test_write_and_collect(tmp_path):
    spec = _spec(tmp_path, {"training.learning_rate": [1e-3, 1e-4],
                            "training.batch_size": [16]})
    out = tmp_path / "out"
    assert len(sweep.write(spec, out)) == 2
    configs = sorted((out / "configs").glob("*.yaml"))
    assert len(configs) == 2
    lines = (out / "runs.txt").read_text().split("\n")[:-1]
    assert [line.split()[0] for line in lines] == ["--train-only"] * 2
    first = Config.from_file(configs[0])
    assert first.get("training.learning_rate") == 1e-3
    assert first.get("training.batch_size") == 16
    assert first.get("run_options.outdir") == str(out / "runs" / first.get("run_options.run_name"))

    run = Path(first.get("run_options.outdir"))
    run.mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"state": "completed"}))
    (run / "manifest.json").write_text(json.dumps({"best_val_loss": 0.5,
                                                   "checkpoint": {"epoch": 3}}))
    rows = sweep.collect(spec, out)
    assert rows[0]["best_val_loss"] == 0.5 and rows[0]["best_epoch"] == 3
    assert rows[1]["status"] == "missing" and rows[1]["best_val_loss"] is None
    assert (out / "results.csv").is_file()


def test_a_bad_key_stops_the_write(tmp_path):
    spec = _spec(tmp_path, {"training.learnig_rate": [1e-3]})
    with pytest.raises(ConfigError):
        sweep.write(spec, tmp_path / "out")


def test_example_sweep_is_valid(tmp_path):
    points = sweep.write(SWEEP / "example_sweep.yaml", tmp_path)
    assert len(points) == 36
