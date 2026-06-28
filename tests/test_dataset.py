"""Smoke tests for galaxy-image simulation."""

import pytest

pytest.importorskip("galsim")
pytest.importorskip("ngmix")

import numpy as np  # noqa: E402

from shearnet.core.dataset import generate_dataset  # noqa: E402


def test_generate_dataset_shapes():
    images, labels = generate_dataset(8, psf_fwhm=0.25, npix=33, seed=0)
    assert images.shape == (8, 33, 33)
    assert labels.shape == (8, 2)
    assert np.isfinite(images).all()


def test_generate_dataset_with_psf_channel():
    images, labels = generate_dataset(4, psf_fwhm=0.25, npix=33, seed=0, return_psf=True)
    # galaxy + PSF stacked on a trailing channel axis
    assert images.shape == (4, 33, 33, 2)
