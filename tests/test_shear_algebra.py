"""The reduced-shear composition, and that both backends label with it."""

import numpy as np
import pytest

from shearnet.core.shear_algebra import compose_shear

galsim = pytest.importorskip("galsim")


def test_zero_applied_shear_is_exactly_the_identity():
    """Every historical run has base shear zero; none of them may move."""
    for e1, e2 in [(0.0, 0.0), (0.3, -0.2), (-0.45, 0.4)]:
        assert compose_shear(e1, e2, 0.0, 0.0) == (e1, e2)


def test_matches_galsim_shear_composition():
    """The formula must agree with what the renderer actually draws."""
    rng = np.random.RandomState(0)
    for _ in range(20):
        e1, e2 = rng.uniform(-0.4, 0.4, size=2)
        g1, g2 = rng.uniform(-0.08, 0.08, size=2)
        got = compose_shear(e1, e2, g1, g2)
        want = galsim.Shear(g1=g1, g2=g2) + galsim.Shear(g1=e1, g2=e2)
        assert got[0] == pytest.approx(want.g1, abs=1e-12)
        assert got[1] == pytest.approx(want.g2, abs=1e-12)


def test_vectorises_over_numpy_arrays():
    e1 = np.array([0.1, -0.3, 0.0])
    e2 = np.array([0.0, 0.2, -0.4])
    o1, o2 = compose_shear(e1, e2, 0.02, -0.01)
    assert o1.shape == (3,) and o2.shape == (3,)
    for i in range(3):
        want = compose_shear(float(e1[i]), float(e2[i]), 0.02, -0.01)
        assert o1[i] == pytest.approx(want[0])
        assert o2[i] == pytest.approx(want[1])


def test_derivative_is_one_minus_eps_squared():
    """This is the analytic R^gamma target, so pin it down."""
    e1, e2, h = 0.3, 0.2, 1e-6
    d11 = (compose_shear(e1, e2, h, 0.0)[0] - compose_shear(e1, e2, -h, 0.0)[0]) / (2 * h)
    d21 = (compose_shear(e1, e2, h, 0.0)[1] - compose_shear(e1, e2, -h, 0.0)[1]) / (2 * h)
    d22 = (compose_shear(e1, e2, 0.0, h)[1] - compose_shear(e1, e2, 0.0, -h)[1]) / (2 * h)
    # eps^2 = (e1 + i e2)^2 -> Re = e1^2 - e2^2, Im = 2 e1 e2
    assert d11 == pytest.approx(1.0 - (e1**2 - e2**2), rel=1e-5)
    assert d22 == pytest.approx(1.0 + (e1**2 - e2**2), rel=1e-5)
    assert d21 == pytest.approx(-2 * e1 * e2, rel=1e-4)


def test_galsim_labels_track_the_applied_shear():
    """A sheared population must be labelled with the shape it was drawn with."""
    from shearnet.core.dataset import generate_dataset

    kw = dict(samples=8, psf_fwhm=0.5, npix=33, seed=0, output_keys=("g1", "g2"))
    _, plain = generate_dataset(**kw)
    _, sheared = generate_dataset(base_shear_g1=0.05, base_shear_g2=-0.03, **kw)
    assert not np.allclose(plain, sheared)
    want = np.stack(compose_shear(plain[:, 0], plain[:, 1], 0.05, -0.03), axis=1)
    assert np.allclose(sheared, want, atol=1e-6)


def test_jax_labels_match_the_galsim_labels_under_applied_shear():
    """The two backends must not disagree about what the label is."""
    pytest.importorskip("jax_galsim")
    from shearnet.core.dataset_jax import JaxRenderConfig, sample_truth

    cfg = JaxRenderConfig(npix=33, scale=0.141, psf_fwhm=0.5)
    truth = sample_truth(8, cfg, seed=0, base_shear_range=0.05, add_noise=False)
    want = np.stack(
        compose_shear(
            truth.params["g1"],
            truth.params["g2"],
            truth.params["base_g1"],
            truth.params["base_g2"],
        ),
        axis=1,
    )
    got = np.stack([truth.labels_raw["g1"], truth.labels_raw["g2"]], axis=1)
    assert np.allclose(got, want)
    # and a nonzero range really did draw a nonzero shear
    assert np.any(np.abs(truth.params["base_g1"]) > 0)
    assert not np.allclose(got, np.stack([truth.params["g1"], truth.params["g2"]], axis=1))


def test_base_shear_range_is_validated():
    pytest.importorskip("jax_galsim")
    from shearnet.core.dataset_jax import JaxRenderConfig, sample_truth

    cfg = JaxRenderConfig(npix=33, scale=0.141, psf_fwhm=0.5)
    with pytest.raises(ValueError, match="base_shear_range"):
        sample_truth(4, cfg, seed=0, base_shear_range=1.5, add_noise=False)
