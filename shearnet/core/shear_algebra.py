"""Reduced-shear algebra shared by every backend.

One function, but it has to live somewhere both ``dataset`` (GalSim) and
``dataset_jax`` (JAX-GalSim) can import, and it must be written in plain
arithmetic so the same code runs on a Python float, a NumPy array and a traced
``jax`` array. That last property is what lets the in-loop trainer differentiate
the *label* with respect to the applied shear and use the result as the exact
per-object target for ``R^gamma`` (see
:class:`~shearnet.core.inloop.ResponseRegularization`).
"""

from __future__ import annotations

__all__ = ["compose_shear"]


def compose_shear(e1, e2, g1, g2):
    """Reduced ellipticity of a profile sheared by ``e`` and then by ``g``.

    Both GalSim and JAX-GalSim build the training stamp as
    ``profile.shear(g1=e1, g2=e2).shear(g1=g1, g2=g2)``, i.e. an intrinsic shape
    followed by an applied shear. For reduced shear the composition is the
    Moebius map of Seitz & Schneider (1997),

    .. math:: \\varepsilon = \\frac{\\varepsilon_s + g}{1 + g^* \\varepsilon_s}

    so the *observed* shape of that stamp is **not** ``(e1, e2)`` unless the
    applied shear is zero. GalSim's composition also carries a rotation, but for
    an elliptical isophote profile that rotation is unobservable, so the formula
    above is exactly the shape of the rendered object.

    This matters because ``(g1, g2)`` are the training labels: with a nonzero
    applied shear, labelling a stamp with its *intrinsic* shape asks the network
    to undo the very shear the response terms then ask it to be sensitive to.

    Args:
        e1, e2: intrinsic reduced ellipticity components.
        g1, g2: applied reduced shear components.

    Returns:
        ``(o1, o2)``, the observed reduced ellipticity. Reduces to ``(e1, e2)``
        exactly -- not approximately -- when ``g1 = g2 = 0``, so every run with
        the historical zero applied shear is bit-identical.

    Written with real arithmetic rather than Python complex so the same
    expression differentiates under ``jax.jvp`` without the holomorphic-vs-real
    ambiguity a complex intermediate would introduce.
    """
    num_r = e1 + g1
    num_i = e2 + g2
    # 1 + conj(g) * e
    den_r = 1.0 + g1 * e1 + g2 * e2
    den_i = g1 * e2 - g2 * e1
    den_sq = den_r * den_r + den_i * den_i
    o1 = (num_r * den_r + num_i * den_i) / den_sq
    o2 = (num_i * den_r - num_r * den_i) / den_sq
    return o1, o2
