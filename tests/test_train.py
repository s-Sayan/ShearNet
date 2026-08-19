"""Smoke tests for the training entry point, including optional PSF input."""

import pytest

pytest.importorskip("galsim")
pytest.importorskip("jax")

import jax.random as random  # noqa: E402

from shearnet.core.dataset import generate_dataset  # noqa: E402
from shearnet.core.train import train_model  # noqa: E402


def test_single_branch_train_without_psf():
    """A single-branch model trains from galaxy images alone (no psf_images)."""
    images, labels = generate_dataset(16, psf_fwhm=0.25, npix=33, seed=0)
    state, train_losses, val_losses, _ = train_model(
        images,
        labels,
        random.PRNGKey(0),
        epochs=1,
        batch_size=8,
        nn="cnn",
    )
    assert state is not None
    assert len(train_losses) == 1


def test_fork_like_requires_psf():
    """The two-branch model raises a clear error when PSF stamps are missing."""
    images, labels = generate_dataset(4, psf_fwhm=0.25, npix=33, seed=0)
    with pytest.raises(ValueError):
        train_model(images, labels, random.PRNGKey(0), nn="fork-like", epochs=1)


def test_single_branch_trains_with_four_output_keys():
    """output_keys sizes the head, so apply must receive it as well as init.

    Without it the head falls back to its two-key default at apply time and the
    parameter shapes disagree -- a latent failure for every single-branch run
    predicting more than g1/g2.
    """
    keys = ("g1", "g2", "hlr", "flux")
    images, labels = generate_dataset(16, psf_fwhm=0.5, npix=33, seed=0, output_keys=keys)
    state, train_losses, _, per_key = train_model(
        images,
        labels,
        random.PRNGKey(0),
        epochs=1,
        batch_size=8,
        nn="cnn",
        output_keys=keys,
    )
    assert state is not None
    assert len(train_losses) == 1
    assert len(per_key[0]) == 4
