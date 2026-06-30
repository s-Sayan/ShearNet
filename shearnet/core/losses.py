"""Pluggable training losses for ShearNet (JAX).

A loss is any callable ``loss(preds, labels, weights) -> scalar`` written in JAX,
where ``preds``/``labels`` have shape ``(batch, n_outputs)`` and ``weights`` is a
``(n_outputs,)`` per-output weight vector. Built-in losses are registered by name
in :data:`LOSS_REGISTRY`; research code can add its own::

    import jax.numpy as jnp
    from shearnet.core.losses import register_loss

    def log_cosh(preds, labels, weights):
        return (jnp.log(jnp.cosh(preds - labels)) * weights[None, :]).mean()

    register_loss("log_cosh", log_cosh)

and then select it by name (``train_model(..., loss="log_cosh")`` or
``training.loss: log_cosh`` in a config), or pass the callable directly
(``train_model(..., loss=log_cosh)``).
"""

import jax.numpy as jnp


def mse_loss(preds, labels, weights):
    """Weighted mean-squared error (the default loss)."""
    return (((preds - labels) ** 2) * weights[None, :]).mean()


def mae_loss(preds, labels, weights):
    """Weighted mean-absolute error."""
    return (jnp.abs(preds - labels) * weights[None, :]).mean()


def huber_loss(preds, labels, weights, delta=1.0):
    """Weighted Huber loss (quadratic near zero, linear in the tails)."""
    abs_err = jnp.abs(preds - labels)
    quad = jnp.minimum(abs_err, delta)
    lin = abs_err - quad
    huber = 0.5 * quad**2 + delta * lin
    return (huber * weights[None, :]).mean()


LOSS_REGISTRY = {
    "mse": mse_loss,
    "mae": mae_loss,
    "huber": huber_loss,
}


def register_loss(name, fn):
    """Register a named loss callable so it can be selected by ``name``."""
    if not callable(fn):
        raise TypeError(f"loss must be callable, got {type(fn)}")
    LOSS_REGISTRY[name] = fn


def resolve_loss(loss):
    """Resolve ``loss`` (a registry name or a callable) to a loss callable.

    Args:
        loss: Either a key in :data:`LOSS_REGISTRY` or a JAX callable
            ``(preds, labels, weights) -> scalar``.

    Returns:
        The loss callable.
    """
    if callable(loss):
        return loss
    if isinstance(loss, str):
        try:
            return LOSS_REGISTRY[loss]
        except KeyError:
            raise ValueError(
                f"Unknown loss '{loss}'. Available: {sorted(LOSS_REGISTRY)}, "
                "or pass a callable (preds, labels, weights) -> scalar."
            )
    raise TypeError(f"loss must be a registry name or a callable, got {type(loss)}")
