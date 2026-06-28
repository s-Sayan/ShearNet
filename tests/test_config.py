"""Smoke tests for the layered YAML configuration.

These import the ``shearnet`` package, which pulls in GalSim via the package
``__init__``; the ``importorskip`` keeps collection clean where that heavy
dependency is not installed.
"""
import os

import pytest

pytest.importorskip("galsim")

from shearnet.config.config_handler import Config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_default_config_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    cfg = Config()
    # The key training actually reads must be present and numeric.
    assert cfg.get("dataset.psf_fwhm") == 0.25
    assert cfg.get("model.type") == "cnn"
    # output_keys must be a real list, not a tuple-shaped string.
    assert list(cfg.get("model.output_keys")) == ["g1", "g2"]
    # No hardcoded personal catalog path leaks through.
    assert cfg.get("catalog.cosmos_cat_fname") is None


def test_user_config_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    cfg = Config(os.path.join(REPO_ROOT, "configs", "dry_run.yaml"))
    assert cfg.get("model.type") == "fork-like"
    assert cfg.get("dataset.psf_fwhm") == 0.25
    assert cfg.get("output.model_name") == "dry_run"
