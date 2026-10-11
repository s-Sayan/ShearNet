"""The config schema: what it accepts, what it refuses, and dialect translation."""

import os
from pathlib import Path

import pytest

from shearnet.config import Config, ConfigError
from shearnet.config.loader import dump_yaml, load_yaml
from shearnet.config.schema import FIELDS, defaults, flatten

REPO = Path(__file__).resolve().parents[1]


def _write(tmp_path, text, name="config.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


# ----------------------------------------------------------------------
# yaml
# ----------------------------------------------------------------------
def test_scientific_notation_is_a_float():
    assert load_yaml("a: 1e-4\nb: 5E+3\nc: 1.0e-3") == {"a": 1e-4, "b": 5e3, "c": 1e-3}


def test_a_key_written_twice_is_an_error():
    with pytest.raises(Exception, match="duplicate key 'epochs'"):
        load_yaml("training:\n  epochs: 2\n  epochs: 3\n")


def test_dump_reads_back_to_the_same_values():
    data = defaults()
    assert load_yaml(dump_yaml(data)) == data


# ----------------------------------------------------------------------
# the schema
# ----------------------------------------------------------------------
def test_defaults_resolve():
    config = Config.from_dict({"run_options": {"run_name": "x"}})
    assert config.get("simulation.psf.gaussian_fwhm") == 0.5
    assert config.get("model.output_keys") == ["g1", "g2"]
    assert config.get("simulation.catalogs.train_file") is None
    assert config.get("training.response")["orbit_k"] == 2


def test_every_field_has_a_doc():
    assert all(field.doc for field in FIELDS.values())


def test_an_unknown_key_is_refused_with_a_suggestion():
    with pytest.raises(ConfigError, match="training.epoch .*did you mean 'training.epochs'"):
        Config.from_dict({"training": {"epoch": 3}})


def test_an_unknown_block_is_refused():
    with pytest.raises(ConfigError, match="unknown top-level blocks"):
        Config.from_dict({"schema_version": 1, "dataset": {"samples": 3}})


def test_types_are_checked():
    with pytest.raises(ConfigError, match="training.epochs must be an integer"):
        Config.from_dict({"training": {"epochs": "3"}})
    with pytest.raises(ConfigError, match="training.learning_rate must be a number"):
        Config.from_dict({"training": {"learning_rate": "fast"}})
    with pytest.raises(ConfigError, match="model.gap must be true or false"):
        Config.from_dict({"model": {"gap": 1}})
    with pytest.raises(ConfigError, match="must be one of"):
        Config.from_dict({"simulation": {"backend": "jax"}})
    with pytest.raises(ConfigError, match="must be > 0"):
        Config.from_dict({"simulation": {"pixel_scale": 0.0}})


def test_an_integral_float_is_accepted_for_an_int():
    assert Config.from_dict({"training": {"epochs": 3.0}}).get("training.epochs") == 3


def test_strict_get_refuses_a_misspelled_key():
    config = Config.from_dict({})
    with pytest.raises(KeyError):
        config.get("training.epoch")
    with pytest.raises(KeyError):
        config.get("dataset.seed")  # not a schema key


def test_relative_paths_resolve_against_the_file(tmp_path):
    path = _write(tmp_path, "schema_version: 1\nsimulation:\n  catalogs:\n"
                            "    train_file: cats/train.fits\n    eval_file: cats/eval.fits\n")
    config = Config.from_file(path)
    assert config.get("simulation.catalogs.train_file") == str(tmp_path / "cats/train.fits")


def test_reading_a_config_creates_nothing(tmp_path):
    path = _write(tmp_path, "run_options:\n  run_name: x\n  outdir: run\n")
    Config.from_file(path)
    assert sorted(os.listdir(tmp_path)) == ["config.yaml"]


def test_resolved_config_round_trips(tmp_path):
    config = Config.from_dict({"run_options": {"run_name": "x"},
                               "training": {"epochs": 3, "ema_decay": 0.99}})
    path = tmp_path / "resolved.yaml"
    config.save(path)
    assert Config.from_file(path) == config


# ----------------------------------------------------------------------
# cross-field rules
# ----------------------------------------------------------------------
@pytest.mark.parametrize("block,message", [
    ({"model": {"type": "fork-like", "d_model": 64}}, "model.d_model .*no effect"),
    ({"model": {"type": "d4-fork-like", "galaxy_branch": "shearnet-d4",
                "psf_branch": "shearnet-d4", "dropout": 0.1}}, "model.dropout .*no effect"),
    ({"model": {"type": "cnn", "fusion": "transformer"}}, "model.fusion .*no effect"),
    ({"model": {"type": "d4-fork-like", "gap": True}}, "refuses model.gap"),
    ({"training": {"loss_weights": [1.0]}}, "loss_weights has 1 entries for 2"),
    ({"training": {"generation": "inloop"}}, "needs simulation.backend jax-galsim"),
    ({"training": {"response": {"psf_weight": 0.1}}}, "set training.generation inloop"),
    ({"simulation": {"catalogs": {"train_file": "/a.fits"}}}, "eval_file is not"),
    ({"simulation": {"catalogs": {"train_file": "/a.fits", "eval_file": "/a.fits"}}},
     "eval_file is the training catalog"),
    ({"evaluation": {"seed": 42}}, "evaluation.seed equals training.seed"),
    ({"evaluation": {"estimators": ["ngmix"]}}, "metacal.shearnet needs shearnet"),
    ({"evaluation": {"rotations_deg": [0, 0]}}, "repeats an angle"),
    ({"evaluation": {"scenes": [{"name": "a", "g1": 0.9, "g2": 0.9}]}}, "must be < 1"),
    ({"evaluation": {"scenes": [{"name": "a", "g1": 0.0, "g2": 0.0},
                                {"name": "a", "g1": 0.1, "g2": 0.0}]}}, "appears twice"),
])
def test_cross_field_rules(block, message):
    with pytest.raises(ConfigError, match=message):
        Config.from_dict(block)


def test_the_d4_settings_a_shearnet_d4_branch_reads_are_allowed():
    Config.from_dict({"model": {
        "type": "d4-fork-like", "galaxy_branch": "shearnet-d4", "psf_branch": "shearnet-d4",
        "head": "attention", "num_pool_heads": 4, "d4_features": [16, 48, 64],
        "d4_multiscale": False, "fusion": "transformer", "fusion_pos": "rope2d"}})


# ----------------------------------------------------------------------
# evaluation overrides
# ----------------------------------------------------------------------
def test_an_evaluation_override_may_change_evaluation_settings(tmp_path):
    base = Config.from_dict({"run_options": {"run_name": "x"},
                             "simulation": {"catalogs": {"train_file": "/t.fits",
                                                         "eval_file": "/e.fits"}}})
    path = _write(tmp_path, "evaluation:\n  nobj: 30\n  rotations_deg: [0, 90]\n"
                            "simulation:\n  catalogs:\n    eval_file: cut.fits\n", "eval.yaml")
    config = base.evaluation_override(path)
    assert config.get("evaluation.nobj") == 30
    assert config.get("evaluation.rotations_deg") == [0.0, 90.0]
    assert config.get("simulation.catalogs.eval_file") == str(tmp_path / "cut.fits")
    assert config.get("simulation.catalogs.train_file") == "/t.fits"


def test_an_evaluation_override_cannot_touch_the_model(tmp_path):
    base = Config.from_dict({"run_options": {"run_name": "x"}})
    path = _write(tmp_path, "model:\n  type: resnet\n", "eval.yaml")
    with pytest.raises(ConfigError, match="may only change evaluation settings"):
        base.evaluation_override(path)


# ----------------------------------------------------------------------
# the package and unit-test layouts
# ----------------------------------------------------------------------
def test_obsolete_layout_is_rejected():
    with pytest.raises(ConfigError):
        Config.from_dict({"dataset": {"samples": 100}, "output": {"model_name": "old"}})


def test_every_config_in_the_repository_is_current_and_valid():
    """What the repo ships must load, and must be written in the current schema."""
    from shearnet.config.loader import read_yaml

    files = sorted(REPO.glob("configs/**/*.yaml"))
    assert len(files) > 40
    for path in files:
        assert read_yaml(path).get("schema_version") == 1, path
        Config.from_file(path)


def test_flatten_and_resolve_agree_on_every_field():
    assert set(flatten(defaults())) - {"schema_version"} == set(FIELDS)
