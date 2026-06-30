"""Unit tests for the pure FFT/sampling helpers in methods.ngmix."""

import numpy as np
import pytest

pytest.importorskip("ngmix")

from shearnet.methods.ngmix import (  # noqa: E402
    convolve2d,
    fft_ifft,
    fourier_transform,
    inverse_fourier_transform,
    sample_half_gaussian,
)


def test_fourier_roundtrip_recovers_image():
    rng = np.random.RandomState(0)
    img = np.abs(rng.rand(16, 16))
    recovered = inverse_fourier_transform(fourier_transform(img))
    assert recovered.shape == img.shape
    assert np.allclose(recovered, img, atol=1e-8)


def test_fft_ifft_roundtrip():
    rng = np.random.RandomState(1)
    img = np.abs(rng.rand(8, 8))
    assert np.allclose(fft_ifft(img), img, atol=1e-8)


def test_convolve2d_preserves_shape():
    img = np.zeros((21, 21))
    img[10, 10] = 1.0
    psf = np.ones((5, 5)) / 25.0
    out = convolve2d(img, psf)
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_sample_half_gaussian_count_and_floor():
    s = sample_half_gaussian(size=200, sigma=0.5, seed=0)
    assert len(s) == 200
    assert (s > 0.14).all()


def test_sample_half_gaussian_reproducible():
    a = sample_half_gaussian(size=50, sigma=0.5, seed=7)
    b = sample_half_gaussian(size=50, sigma=0.5, seed=7)
    assert np.allclose(a, b)
