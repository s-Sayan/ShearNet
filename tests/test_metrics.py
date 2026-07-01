"""Unit tests for metric helper functions."""

import numpy as np
import pytest

pytest.importorskip("galsim")
pytest.importorskip("ngmix")
pytest.importorskip("optax")

from shearnet.metrics import (  # noqa: E402
    _per_label_metrics,
    remove_nan_preds,
    remove_nan_preds_multi,
)


def test_remove_nan_preds_drops_nan_rows():
    preds = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
    labels = np.array([[0.0, 0.0], [9.0, 9.0], [1.0, 1.0]])
    p, lab = remove_nan_preds(preds, labels)
    assert p.shape == (2, 2)
    assert lab.shape == (2, 2)
    assert not np.isnan(p).any()
    # the surviving labels are the non-NaN rows
    assert np.allclose(lab, np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_remove_nan_preds_multi_drops_union_of_nans():
    p1 = np.array([[1.0, 1.0], [np.nan, 1.0], [3.0, 3.0]])
    p2 = np.array([[1.0, 1.0], [1.0, 1.0], [np.nan, 3.0]])
    labels = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    a, b, lab = remove_nan_preds_multi(p1, p2, labels)
    # only the first row is NaN-free in both
    assert a.shape == (1, 2) and b.shape == (1, 2) and lab.shape == (1, 2)
    assert np.allclose(lab, np.array([[0.0, 0.0]]))


def test_per_label_metrics_zero_on_exact_match():
    arr = np.array([[0.1, -0.2], [0.0, 0.3]], dtype=np.float32)
    loss, bias = _per_label_metrics(arr, arr, ("g1", "g2"))
    assert loss["g1"] == pytest.approx(0.0, abs=1e-6)
    assert bias["g2"] == pytest.approx(0.0, abs=1e-6)
    assert "g1g2_combined" in loss


def test_per_label_metrics_bias_sign():
    labels = np.zeros((4, 2), dtype=np.float32)
    preds = np.full((4, 2), 0.5, dtype=np.float32)
    _, bias = _per_label_metrics(preds, labels, ("g1", "g2"))
    assert bias["g1"] == pytest.approx(0.5, abs=1e-6)
    assert bias["g2"] == pytest.approx(0.5, abs=1e-6)
