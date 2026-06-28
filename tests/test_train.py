"""Smoke tests for the training entry point, including optional PSF input."""
import pytest

pytest.importorskip("galsim")
pytest.importorskip("jax")

import jax.random as random

from shearnet.core.dataset import generate_dataset
from shearnet.core.train import train_model


def test_single_branch_train_without_psf():
    """A single-branch model trains from galaxy images alone (no psf_images)."""
    images, labels = generate_dataset(16, psf_fwhm=0.25, npix=33, seed=0)
    state, train_losses, val_losses, _ = train_model(
        images, labels, random.PRNGKey(0),
        epochs=1, batch_size=8, nn="cnn",
    )
    assert state is not None
    assert len(train_losses) == 1


def test_fork_like_requires_psf():
    """The two-branch model raises a clear error when PSF stamps are missing."""
    images, labels = generate_dataset(4, psf_fwhm=0.25, npix=33, seed=0)
    with pytest.raises(ValueError):
        train_model(images, labels, random.PRNGKey(0), nn="fork-like", epochs=1)
