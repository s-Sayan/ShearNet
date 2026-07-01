"""Tests for the pluggable JAX loss API."""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from shearnet.core.losses import (
    LOSS_REGISTRY,
    huber_loss,
    mae_loss,
    mse_loss,
    register_loss,
    resolve_loss,
)


def _data():
    preds = jnp.array([[0.1, -0.2], [0.0, 0.3]])
    labels = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    weights = jnp.ones(2)
    return preds, labels, weights


def test_builtin_losses_are_scalars_and_nonnegative():
    preds, labels, weights = _data()
    for fn in (mse_loss, mae_loss, huber_loss):
        val = fn(preds, labels, weights)
        assert val.shape == ()
        assert float(val) >= 0.0


def test_resolve_loss_by_name_and_callable():
    assert resolve_loss("mse") is mse_loss
    custom = lambda p, lab, w: jnp.abs(p - lab).sum()  # noqa: E731
    assert resolve_loss(custom) is custom


def test_resolve_loss_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_loss("not-a-loss")


def test_resolve_loss_bad_type_raises():
    with pytest.raises(TypeError):
        resolve_loss(123)


def test_register_loss_then_resolve():
    def log_cosh(p, lab, w):
        return (jnp.log(jnp.cosh(p - lab)) * w[None, :]).mean()

    register_loss("log_cosh_test", log_cosh)
    assert "log_cosh_test" in LOSS_REGISTRY
    assert resolve_loss("log_cosh_test") is log_cosh


@pytest.mark.parametrize("loss", ["mse", "mae", "huber"])
def test_train_model_with_named_loss(loss):
    from shearnet.core.train import train_model

    imgs = np.random.rand(16, 21, 21).astype("float32")
    labels = (np.random.rand(16, 2).astype("float32") - 0.5) * 0.1
    state, tr, val, _ = train_model(
        imgs, labels, random.PRNGKey(0), epochs=1, batch_size=8, nn="cnn", loss=loss
    )
    assert len(tr) == 1
    assert np.isfinite(float(tr[0]))


def test_train_model_with_custom_callable_loss():
    from shearnet.core.train import train_model

    def weighted_mae(preds, labels, weights):
        return (jnp.abs(preds - labels) * weights[None, :]).mean()

    imgs = np.random.rand(16, 21, 21).astype("float32")
    labels = (np.random.rand(16, 2).astype("float32") - 0.5) * 0.1
    state, tr, val, _ = train_model(
        imgs, labels, random.PRNGKey(0), epochs=1, batch_size=8, nn="cnn", loss=weighted_mae
    )
    assert len(tr) == 1
    assert np.isfinite(float(tr[0]))
