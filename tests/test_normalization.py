"""Round-trip and persistence tests for label normalization (numpy only)."""

import numpy as np
import pytest

from shearnet.utils.normalization import (
    fit_normalizer,
    inverse_transform_labels,
    load_normalizer,
    save_normalizer,
    transform_labels,
)


def _labels(seed=0):
    rng = np.random.RandomState(seed)
    g = rng.normal(0.0, 0.2, (200, 2))
    extra = np.column_stack([np.full(200, 0.5), rng.normal(1e4, 1e3, 200)])
    return np.hstack([g, extra]).astype(np.float32)


def test_transform_is_zero_mean_unit_std():
    labels = _labels()
    params = fit_normalizer(labels)
    z = transform_labels(labels, params)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    # Columns with non-trivial spread should have ~unit std.
    std = z.std(axis=0)
    assert np.allclose(std[[0, 1, 3]], 1.0, atol=1e-4)


def test_inverse_is_left_inverse():
    labels = _labels()
    params = fit_normalizer(labels)
    recovered = inverse_transform_labels(transform_labels(labels, params), params)
    assert np.allclose(recovered, labels, atol=1e-3)


def test_zero_std_column_does_not_divide_by_zero():
    # Constant column -> std guarded to 1.0, transform stays finite.
    labels = np.column_stack([np.full(50, 0.5), np.linspace(-1, 1, 50)]).astype(np.float32)
    params = fit_normalizer(labels)
    z = transform_labels(labels, params)
    assert np.isfinite(z).all()
    assert params["std"][0] == 1.0


def test_save_and_load_roundtrip(tmp_path):
    labels = _labels()
    params = fit_normalizer(labels)
    path = tmp_path / "norm.npz"
    save_normalizer(params, str(path))
    loaded = load_normalizer(str(path))
    assert np.allclose(loaded["mean"], params["mean"])
    assert np.allclose(loaded["std"], params["std"])
