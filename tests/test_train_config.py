"""Unit tests for the training CLI config resolver (``build_train_config``).

These exercise pure configuration resolution, so they do not need the heavy
GalSim/ngmix runtime. The
``importorskip`` guards keep collection clean where those import-time
dependencies of the package are unavailable.
"""

import os

import pytest

pytest.importorskip("galsim")
pytest.importorskip("ngmix")

import yaml  # noqa: E402

from shearnet.cli.train import (  # noqa: E402
    _CLI_DEFAULTS,
    build_train_config,
    create_parser,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_YAML = os.path.join(REPO_ROOT, "shearnet", "config", "default_config.yaml")

# _CLI_DEFAULTS key -> dotted path in default_config.yaml. ``plot`` is omitted
# on purpose: the bare CLI does not plot, while the YAML default enables it.
_SHARED_KEYS = {
    "epochs": "training.epochs",
    "seed": "dataset.seed",
    "batch_size": "training.batch_size",
    "samples": "dataset.samples",
    "patience": "training.patience",
    "psf_fwhm": "dataset.psf_fwhm",
    "nse_sd": "dataset.nse_sd",
    "exp": "dataset.exp",
    "nn": "model.type",
    "learning_rate": "training.learning_rate",
    "weight_decay": "training.weight_decay",
    "model_name": "output.model_name",
    "val_split": "training.val_split",
    "eval_interval": "training.eval_interval",
    "stamp_size": "dataset.stamp_size",
    "pixel_size": "dataset.pixel_size",
    "galaxy_type": "model.galaxy.type",
    "psf_type": "model.psf.type",
    "fusion": "model.fusion",
    "apply_psf_shear": "dataset.apply_psf_shear",
    "psf_shear_range": "dataset.psf_shear_range",
    "gap": "model.gap",
    "output_keys": "model.output_keys",
}


def _yaml_get(data, dotted):
    cur = data
    for key in dotted.split("."):
        cur = cur[key]
    return cur


def test_cli_defaults_match_yaml():
    """The argparse fallback defaults must not drift from default_config.yaml."""
    with open(DEFAULT_YAML) as f:
        data = yaml.safe_load(f)
    for cli_key, dotted in _SHARED_KEYS.items():
        expected = _yaml_get(data, dotted)
        actual = _CLI_DEFAULTS[cli_key]
        if cli_key == "output_keys":
            expected, actual = tuple(expected), tuple(actual)
        assert actual == expected, f"{cli_key} drifted: CLI {actual!r} vs YAML {expected!r}"


def test_no_config_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    cfg = build_train_config(create_parser().parse_args([]))
    assert cfg.get("dataset.samples") == 10000
    assert cfg.get("model.type") == "cnn"
    assert cfg.get("training.epochs") == 10
    assert cfg.get("dataset.hlr_type") == "constant"
    # No-config mode does not enable plotting unless asked.
    assert cfg.get("plotting.plot") is None


def test_cli_overrides_apply(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    args = create_parser().parse_args(["--epochs", "3", "--nn", "resnet", "--samples", "256"])
    cfg = build_train_config(args)
    assert cfg.get("training.epochs") == 3
    assert cfg.get("model.type") == "resnet"
    assert cfg.get("dataset.samples") == 256


def test_fork_like_is_honoured_as_asked_for(tmp_path, monkeypatch):
    """The architecture is the request; nothing rewrites it behind the caller.

    The old ``--process_psf`` flag asked the same question as ``--nn`` and the
    two could contradict each other, which the CLI resolved by silently
    replacing the model. A config could therefore train something it did not
    name.
    """
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    cfg = build_train_config(create_parser().parse_args(["--nn", "fork-like"]))
    assert cfg.get("model.type") == "fork-like"


def test_legacy_process_psf_key_still_loads(tmp_path, monkeypatch):
    """Every training_config.yaml written before the removal carries this key.

    Those runs have to stay re-evaluable, so the key is inert rather than
    rejected.
    """
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "model:\n  type: d4-fork-like\n  process_psf: true\n  output_keys: [g1, g2]\n"
    )
    cfg = build_train_config(create_parser().parse_args(["--config", str(path)]))
    assert cfg.get("model.type") == "d4-fork-like"


def test_disagreeing_legacy_key_warns_and_does_not_rewrite_the_model(
    tmp_path, monkeypatch, caplog
):
    """A contradiction is worth saying out loud, not worth acting on."""
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "model:\n  type: d4-fork-like\n  process_psf: false\n  output_keys: [g1, g2]\n"
    )
    with caplog.at_level("WARNING"):
        cfg = build_train_config(create_parser().parse_args(["--config", str(path)]))
    assert cfg.get("model.type") == "d4-fork-like"
    assert "process_psf" in caplog.text


def test_config_file_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    cfg_path = os.path.join(REPO_ROOT, "configs", "dry_run.yaml")
    cfg = build_train_config(create_parser().parse_args(["--config", cfg_path]))
    assert cfg.get("model.type") == "fork-like"
    assert cfg.get("output.model_name") == "dry_run"
