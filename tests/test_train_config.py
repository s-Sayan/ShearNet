"""The training CLI resolves its config from the YAML alone."""

import os

import pytest

from shearnet.cli.train import build_train_config, create_parser
from shearnet.config import ConfigError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_config_is_required():
    with pytest.raises(SystemExit):
        create_parser().parse_args([])


def test_a_current_schema_file_loads(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("schema_version: 1\nrun_options:\n  run_name: tiny\n"
                    "training:\n  epochs: 3\nmodel:\n  type: resnet\n")
    cfg = build_train_config(create_parser().parse_args(
        ["--config", str(path), "--run", str(tmp_path / "run")]))
    assert cfg.get("training.epochs") == 3
    assert cfg.get("model.type") == "resnet"
    assert cfg.get("run_options.run_name") == "tiny"
    assert cfg.get("run_options.outdir") == str(tmp_path / "run")


def test_a_run_needs_a_directory(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("schema_version: 1\nrun_options:\n  run_name: tiny\n")
    with pytest.raises(ConfigError, match="outdir or --run"):
        build_train_config(create_parser().parse_args(["--config", str(path)]))


def test_a_run_needs_a_name(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("schema_version: 1\ntraining:\n  epochs: 3\n")
    with pytest.raises(ConfigError, match="run_name"):
        build_train_config(create_parser().parse_args(
            ["--config", str(path), "--run", str(tmp_path / "run")]))


def test_the_architecture_is_taken_as_asked_for(tmp_path):
    """Nothing rewrites model.type behind the caller (the old process_psf did)."""
    path = tmp_path / "legacy.yaml"
    path.write_text("model:\n  type: d4-fork-like\n  process_psf: false\n"
                    "  output_keys: [g1, g2]\noutput:\n  model_name: m\n")
    cfg = build_train_config(create_parser().parse_args(
        ["--config", str(path), "--run", str(tmp_path / "run")]))
    assert cfg.get("model.type") == "d4-fork-like"
    assert any("process_psf" in note for note in cfg.notes)


def test_the_shipped_smoke_config_loads(tmp_path):
    cfg_path = os.path.join(REPO_ROOT, "configs", "smoke.yaml")
    cfg = build_train_config(create_parser().parse_args(
        ["--config", cfg_path, "--run", str(tmp_path / "run")]))
    assert cfg.get("model.type") == "fork-like"
    assert cfg.get("run_options.run_name") == "smoke"
