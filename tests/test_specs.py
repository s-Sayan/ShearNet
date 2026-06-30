"""Tests for the DatasetSpec / TrainConfig parameter-group dataclasses."""

import inspect

import pytest

pytest.importorskip("galsim")
pytest.importorskip("ngmix")

from shearnet.core.dataset import generate_dataset  # noqa: E402
from shearnet.core.specs import DatasetSpec, TrainConfig  # noqa: E402
from shearnet.core.train import train_model  # noqa: E402
from shearnet.config.config_handler import Config  # noqa: E402


def test_dataset_spec_kwargs_are_valid_generate_dataset_params():
    params = set(inspect.signature(generate_dataset).parameters)
    assert set(DatasetSpec(samples=1, psf_fwhm=0.25).as_kwargs()).issubset(params)


def test_train_config_kwargs_are_valid_train_model_params():
    params = set(inspect.signature(train_model).parameters)
    assert set(TrainConfig().as_kwargs()).issubset(params)


def test_dataset_spec_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    spec = DatasetSpec.from_config(Config())
    assert spec.samples == 10000
    assert spec.psf_fwhm == 0.25
    assert spec.npix == 53  # stamp_size
    assert spec.scale == 0.141  # pixel_size
    assert spec.output_keys == ("g1", "g2")
    assert spec.return_psf is False  # process_psf default


def test_train_config_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SHEARNET_DATA_PATH", str(tmp_path))
    tc = TrainConfig.from_config(Config(), save_path="/tmp/ckpt")
    assert tc.epochs == 10
    assert tc.batch_size == 32
    assert tc.nn == "cnn"
    assert tc.lr == 1e-3  # training.learning_rate
    assert tc.save_path == "/tmp/ckpt"
    assert tc.output_keys == ("g1", "g2")


def test_dataset_spec_build():
    spec = DatasetSpec(samples=8, psf_fwhm=0.25, npix=21, seed=0)
    images, labels = spec.build()
    assert images.shape == (8, 21, 21)
    assert labels.shape == (8, 2)


def test_train_config_run():
    import jax.random as random
    import numpy as np

    imgs = np.random.rand(16, 21, 21).astype("float32")
    labels = (np.random.rand(16, 2).astype("float32") - 0.5) * 0.1
    tc = TrainConfig(epochs=1, batch_size=8, nn="cnn")
    state, train_loss, val_loss, _ = tc.run(imgs, labels, random.PRNGKey(0))
    assert state is not None
    assert len(train_loss) == 1
