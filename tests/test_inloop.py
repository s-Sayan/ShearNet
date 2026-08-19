"""Tests for in-loop (inside the jitted step) dataset generation."""

import numpy as np
import pytest

pytest.importorskip("jax_galsim")
jax = pytest.importorskip("jax")
import flax.linen as fnn  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from flax.training import train_state  # noqa: E402

from shearnet.core.dataset_jax import (  # noqa: E402
    JaxRenderConfig,
    render_truth,
    sample_truth,
)
from shearnet.core.inloop import (  # noqa: E402
    PROTECTED_PARAMS,
    RESPONSE_REPORT_FIELDS,
    RESPONSE_TERMS,
    InLoopGenerator,
    ResponseRegularization,
    _project_out,
    _protected_names,
    build_generator,
    compilation_report,
    format_response_report,
    make_batch_render,
    make_fused_eval_step,
    make_fused_train_step,
    make_response_report,
    normalize_state,
)

NPIX, SCALE, FWHM = 53, 0.141, 0.5
X64 = jax.config.jax_enable_x64


def _cfg(exp="ideal", batch=8):
    return JaxRenderConfig(
        npix=NPIX, scale=SCALE, psf_fwhm=FWHM, exp=exp, fft_size=256, batch_size=batch
    )


class _Tiny(fnn.Module):
    @fnn.compact
    def __call__(self, gal, psf):
        x = jnp.stack([gal, psf], axis=-1)
        x = fnn.relu(fnn.Conv(4, (3, 3))(x))
        return fnn.Dense(4)(x.reshape(x.shape[0], -1))


def _harness(exp="ideal", n=32, batch=8, apply_psf_shear=False):
    cfg = _cfg(exp, batch)
    gen = build_generator(n, cfg, batch, seed=0, apply_psf_shear=apply_psf_shear)
    model = _Tiny()
    params = model.init(
        jax.random.PRNGKey(0), jnp.zeros((1, NPIX, NPIX)), jnp.zeros((1, NPIX, NPIX))
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))

    def forward(params, gal, psf, dropout_key):
        del dropout_key
        return model.apply(params, gal, psf)

    def loss_fn(preds, lab):
        return jnp.mean((preds - lab) ** 2)

    labels = gen.labels(("g1", "g2", "hlr", "flux"))
    return gen, state, forward, loss_fn, labels


# ----------------------------------------------------------------------
# bookkeeping
# ----------------------------------------------------------------------
def test_steps_per_epoch_drops_only_the_remainder():
    gen = build_generator(50, _cfg(batch=8), 8, seed=0)
    assert gen.steps_per_epoch == 6  # 48 of 50
    idx = gen.epoch_indices(jax.random.PRNGKey(0))
    assert idx.shape == (6, 8)
    flat = np.asarray(idx).ravel()
    assert len(set(flat.tolist())) == len(flat)  # no object twice per epoch


def test_epoch_indices_reshuffle_between_epochs():
    gen = build_generator(50, _cfg(batch=8), 8, seed=0)
    a = np.asarray(gen.epoch_indices(jax.random.PRNGKey(0)))
    b = np.asarray(gen.epoch_indices(jax.random.PRNGKey(1)))
    assert not np.array_equal(a, b)  # remainder is not fixed


def test_rejects_batch_larger_than_sample():
    with pytest.raises(ValueError, match="fewer than one batch"):
        build_generator(4, _cfg(batch=8), 8, seed=0)


def test_labels_reject_psf_keys():
    gen = build_generator(16, _cfg(batch=8), 8, seed=0)
    with pytest.raises(ValueError, match="ngmix adaptive-moments"):
        gen.labels(("g1", "psf_T"))
    assert gen.labels(("g1", "g2")).shape == (16, 2)


# ----------------------------------------------------------------------
# precision plumbing
# ----------------------------------------------------------------------
def test_render_precision_follows_the_process():
    gen = build_generator(16, _cfg(batch=8), 8, seed=0)
    assert gen.cfg.x64 is bool(X64)


def test_network_input_is_float32_regardless_of_x64():
    """The cast is what keeps a global JAX_ENABLE_X64=1 from doubling training."""
    gen, state, _, loss_fn, labels = _harness()
    seen = {}

    def probe(p, gal, psf, dk):
        seen["gal"] = gal.dtype
        seen["psf"] = psf.dtype
        return jnp.zeros((gal.shape[0], 4), gal.dtype)

    step = make_fused_train_step(gen, probe, loss_fn, labels, 12.7)
    idx = gen.epoch_indices(jax.random.PRNGKey(0))[0]
    step(normalize_state(state), idx, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    assert seen["gal"] == jnp.float32
    assert seen["psf"] == jnp.float32


# ----------------------------------------------------------------------
# compilation -- the headline property
# ----------------------------------------------------------------------
@pytest.mark.parametrize("exp", ["ideal", "superbit"])
def test_fused_step_compiles_exactly_once(exp):
    """Many steps across many epochs must produce one executable, not many."""
    try:
        gen, state, forward, loss_fn, labels = _harness(exp=exp)
    except FileNotFoundError:
        pytest.skip("no PSFEx files available")
    rep = compilation_report(gen, forward, loss_fn, labels, 12.7, epochs=3, steps=3, state=state)
    assert rep["steps_run"] == 9
    assert rep["train_cache"] == 1
    assert rep["eval_cache"] == 1
    assert rep["train_traces"] == 1
    assert rep["eval_traces"] == 1


def test_normalize_state_prevents_a_second_executable():
    """TrainState.step starts as a Python int and becomes an array.

    That flips the leaf's abstract value on call two and costs a second XLA
    compile of the whole fused graph -- without a second Python retrace, which
    is why the cache size is the metric that matters.
    """
    gen, state, forward, loss_fn, labels = _harness()
    idx = gen.epoch_indices(jax.random.PRNGKey(0))

    raw = make_fused_train_step(gen, forward, loss_fn, labels, 12.7)
    s = state
    for i in range(3):
        s, _ = raw(s, idx[i], jax.random.PRNGKey(i), jax.random.PRNGKey(i))
    assert raw._cache_size() == 2  # the bug, reproduced

    fixed = make_fused_train_step(gen, forward, loss_fn, labels, 12.7)
    s = normalize_state(state)
    for i in range(3):
        s, _ = fixed(s, idx[i], jax.random.PRNGKey(i), jax.random.PRNGKey(i))
    assert fixed._cache_size() == 1  # and the fix


def test_normalize_state_is_a_noop_for_non_trainstate():
    sentinel = object()
    assert normalize_state(sentinel) is sentinel


# ----------------------------------------------------------------------
# fresh noise
# ----------------------------------------------------------------------
def test_noise_is_fresh_per_step_and_absent_without_it():
    gen, _, _, _, _ = _harness()
    idx = gen.epoch_indices(jax.random.PRNGKey(0))[0]

    noisy = make_batch_render(gen, nse_sd=12.7)
    a = np.asarray(noisy(idx, jax.random.PRNGKey(1))[0])
    b = np.asarray(noisy(idx, jax.random.PRNGKey(2))[0])
    assert not np.allclose(a, b), "same stamps twice: noise is not fresh"
    # same galaxies underneath, so the difference is pure zero-mean noise
    assert abs(float(np.mean(a - b))) < 5.0
    assert float(np.std(a - b)) > 1.0

    clean = make_batch_render(gen, nse_sd=0.0)
    c = np.asarray(clean(idx, jax.random.PRNGKey(1))[0])
    d = np.asarray(clean(idx, jax.random.PRNGKey(2))[0])
    assert np.allclose(c, d)
    # the PSF channel is never noised
    assert np.allclose(
        np.asarray(noisy(idx, jax.random.PRNGKey(1))[1]),
        np.asarray(clean(idx, jax.random.PRNGKey(9))[1]),
    )


def test_eval_noise_is_frozen_by_batch_index():
    """Validation must not wander with the noise draw, or patience reacts to it."""
    gen, state, forward, loss_fn, labels = _harness()
    step = make_fused_eval_step(gen, forward, loss_fn, labels, 12.7)
    idx = gen.epoch_indices(jax.random.PRNGKey(0))[0]
    st = normalize_state(state)
    val_key = jax.random.PRNGKey(7)
    a = step(st, idx, jax.random.fold_in(val_key, 0))
    b = step(st, idx, jax.random.fold_in(val_key, 0))
    assert float(a) == float(b)


# ----------------------------------------------------------------------
# the in-loop render must equal the up-front render
# ----------------------------------------------------------------------
@pytest.mark.parametrize("exp", ["ideal", "superbit"])
def test_inloop_render_matches_up_front_render(exp):
    """Both paths vmap the same make_render_one body; prove they agree."""
    n, batch = 16, 8
    cfg = _cfg(exp, batch)
    try:
        truth = sample_truth(n, cfg, seed=0, add_noise=False)
    except FileNotFoundError:
        pytest.skip("no PSFEx files available")
    gal_ref, psf_ref = render_truth(truth, cfg)

    gen = InLoopGenerator(truth, cfg, batch)
    render = make_batch_render(gen, nse_sd=0.0)
    got = np.asarray(render(jnp.arange(batch), jax.random.PRNGKey(0))[0], dtype=np.float64)
    want = gal_ref[:batch]
    scale_ = np.abs(want).sum(axis=(1, 2)).max()
    # float32 network cast sets the floor; render agreement is far tighter
    assert np.abs(got - want).max() / scale_ < 1e-6


def test_labels_line_up_with_the_gathered_batch():
    """The loss must see the labels of the objects that were rendered."""
    gen, state, _, _, labels = _harness()
    idx = jnp.array([3, 1, 7, 0, 5, 2, 6, 4])

    def zeros(params, gal, psf, dropout_key):
        del params, psf, dropout_key
        return jnp.zeros((gal.shape[0], 4), gal.dtype)

    def mean_sq(preds, lab):
        del preds
        return jnp.mean(lab**2)

    step = make_fused_train_step(gen, zeros, mean_sq, labels, 0.0)
    _, loss = step(normalize_state(state), idx, jax.random.PRNGKey(0), jax.random.PRNGKey(0))
    want = float(np.mean(np.asarray(labels)[np.asarray(idx)] ** 2))
    assert float(loss) == pytest.approx(want, rel=1e-6)


# ----------------------------------------------------------------------
# differentiability survives fusion
# ----------------------------------------------------------------------
def test_gradients_flow_through_the_fused_step():
    gen, state, forward, loss_fn, labels = _harness()
    step = make_fused_train_step(gen, forward, loss_fn, labels, 12.7)
    st = normalize_state(state)
    idx = gen.epoch_indices(jax.random.PRNGKey(0))[0]
    new, loss = step(st, idx, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    assert np.isfinite(float(loss))
    moved = jax.tree_util.tree_reduce(
        lambda acc, x: acc or bool(np.any(np.asarray(x) != 0)),
        jax.tree_util.tree_map(lambda a, b: np.asarray(a) - np.asarray(b), new.params, st.params),
        False,
    )
    assert moved, "parameters did not update"


# ----------------------------------------------------------------------
# response regularisation
# ----------------------------------------------------------------------
def test_response_regularisation_runs_all_terms_and_returns_metrics():
    """The combined JVP/orbit path is finite and updates the model."""
    gen, state, forward, loss_fn, labels = _harness(n=16, batch=8, apply_psf_shear=True)
    response = ResponseRegularization(
        gamma_weight=0.1,
        psf_weight=0.1,
        shift_weight=0.1,
        complement_weight=0.1,
        orbit_weight=0.1,
    )
    step = make_fused_train_step(
        gen,
        forward,
        loss_fn,
        labels,
        12.7,
        response=response,
        shear_indices=(0, 1),
        return_metrics=True,
    )
    idx = gen.epoch_indices(jax.random.PRNGKey(0))[0]
    old = normalize_state(state)
    new, loss, metrics = step(old, idx, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    assert np.isfinite(float(loss))
    # [supervised, *RESPONSE_TERMS, active]
    assert np.asarray(metrics).shape == (len(RESPONSE_TERMS) + 2,)
    assert np.all(np.isfinite(np.asarray(metrics)))
    assert float(metrics[0]) > 0.0
    assert float(metrics[-1]) == 1.0
    # every term must actually contribute; a silently-zero term is a term that
    # is measuring nothing (an untraced PSF shear does exactly that)
    for name, value in zip(RESPONSE_TERMS, np.asarray(metrics)[1:-1]):
        assert float(value) > 0.0, f"{name} term is identically zero"
    moved = jax.tree_util.tree_reduce(
        lambda acc, x: acc or bool(np.any(np.asarray(x) != 0)),
        jax.tree_util.tree_map(lambda a, b: np.asarray(a) - np.asarray(b), new.params, old.params),
        False,
    )
    assert moved


def test_response_regularisation_can_be_staggered():
    gen, state, forward, loss_fn, labels = _harness(n=16, batch=8, apply_psf_shear=True)
    response = ResponseRegularization(orbit_weight=1.0, every_n_steps=2)
    step = make_fused_train_step(
        gen,
        forward,
        loss_fn,
        labels,
        0.0,
        response=response,
        shear_indices=(0, 1),
        return_metrics=True,
    )
    idx = gen.epoch_indices(jax.random.PRNGKey(0))[0]
    state = normalize_state(state)
    state, _, first = step(state, idx, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    _, _, second = step(state, idx, jax.random.PRNGKey(3), jax.random.PRNGKey(4))
    assert float(first[-1]) == 1.0
    assert float(second[-1]) == 0.0
    assert float(second[1:-1].sum()) == 0.0


def test_response_config_validation():
    with pytest.raises(ValueError, match="orbit_k"):
        ResponseRegularization(orbit_weight=1.0, orbit_k=3)
    with pytest.raises(ValueError, match="gamma_target"):
        ResponseRegularization(gamma_weight=1.0, gamma_target="ensemble")
    with pytest.raises(ValueError, match="every_n_steps"):
        ResponseRegularization(gamma_weight=1.0, every_n_steps=0)
    assert ResponseRegularization().orbit_angles == (90.0,)
    assert ResponseRegularization(orbit_k=4).orbit_angles == (45.0, 90.0, 135.0)
    assert not ResponseRegularization().enabled
    assert ResponseRegularization(shift_weight=1e-3).enabled


@pytest.mark.parametrize("orbit_k", [2, 4])
def test_orbit_term_vanishes_only_for_an_orbit_invariant_estimator(orbit_k):
    """The term measures the spread of the prediction across the PSF orbit.

    A constant estimator is orbit-invariant by construction and must score
    exactly zero; note that a merely *PSF-channel-blind* model would not, because
    rotating the PSF also changes the convolved galaxy stamp.
    """
    gen, state, _, loss_fn, labels = _harness(n=8, batch=8, apply_psf_shear=True)

    def constant(params, gal, psf, dropout_key):
        del psf, dropout_key
        return jnp.zeros((gal.shape[0], 4), gal.dtype) + jnp.sum(
            jax.tree_util.tree_leaves(params)[0]
        )

    response = ResponseRegularization(orbit_weight=1.0, orbit_k=orbit_k)
    step = make_fused_train_step(
        gen, constant, loss_fn, labels, 0.0, response=response, return_metrics=True
    )
    _, _, metrics = step(
        normalize_state(state), jnp.arange(8), jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    )
    assert float(np.asarray(metrics)[1 + RESPONSE_TERMS.index("orbit")]) == 0.0


@pytest.mark.parametrize("orbit_k", [2, 4])
def test_orbit_actually_rotates_the_psf(orbit_k):
    """Guard the mechanism, not just the number: the orbit must re-render."""
    from shearnet.core.inloop import _make_render_batch

    gen = build_generator(4, _cfg(batch=4), 4, seed=0, apply_psf_shear=True)
    ref = _make_render_batch(gen.cfg, True, jnp.float32)
    p = {k: v[jnp.arange(4)] for k, v in gen.params.items()}
    base_gal, base_psf = ref(p)
    for angle in ResponseRegularization(orbit_k=orbit_k).orbit_angles:
        rot = _make_render_batch(gen.cfg, True, jnp.float32, angle)
        gal, psf = rot(p)
        # the PSF stamp moved, and so did the convolved galaxy
        assert not np.allclose(np.asarray(psf), np.asarray(base_psf), atol=1e-9)
        assert not np.allclose(np.asarray(gal), np.asarray(base_gal), atol=1e-9)
    # flux is conserved by a rotation
    assert float(jnp.sum(psf)) == pytest.approx(float(jnp.sum(base_psf)), rel=1e-4)


def test_orbit_term_is_nonzero_for_a_psf_sensitive_model():
    gen, state, forward, loss_fn, labels = _harness(n=8, batch=8, apply_psf_shear=True)
    step = make_fused_train_step(
        gen,
        forward,
        loss_fn,
        labels,
        0.0,
        response=ResponseRegularization(orbit_weight=1.0),
        return_metrics=True,
    )
    _, _, metrics = step(
        normalize_state(state), jnp.arange(8), jax.random.PRNGKey(0), jax.random.PRNGKey(1)
    )
    assert float(np.asarray(metrics)[1 + RESPONSE_TERMS.index("orbit")]) > 0.0


def test_protected_subspace_includes_the_intrinsic_shape():
    """The supervised signal must not sit in the penalised complement.

    ``g1``/``g2`` are simulator parameters, so their image tangents are physical.
    Leaving them out of the protected basis would have the complement penalty
    fight the very sensitivity the supervised loss is training for.
    """
    assert {"g1", "g2"}.issubset(PROTECTED_PARAMS)
    assert {"base_g1", "base_g2", "psf_g1", "psf_g2", "dx", "dy"}.issubset(PROTECTED_PARAMS)
    ideal = _protected_names(_cfg("ideal"))
    superbit = _protected_names(_cfg("superbit"))
    assert "psf_x" not in ideal  # a fixed analytic PSF does not move
    assert {"psf_x", "psf_y"}.issubset(superbit)


def test_projection_removes_the_protected_directions_exactly():
    """A rank-deficient basis must still give an exact projection."""
    key = jax.random.PRNGKey(0)
    b, s, d = 3, 5, 40
    basis = jax.random.normal(key, (b, s, d))
    # duplicate a row and zero another: exactly the degeneracies the physical
    # basis has (intrinsic vs applied shear; an untraced PSF shear)
    basis = basis.at[:, 1].set(basis[:, 0]).at[:, 4].set(0.0)
    z = jax.random.normal(jax.random.PRNGKey(1), (b, d))
    z_perp = _project_out(basis, z)
    overlap = jnp.einsum("bsd,bd->bs", basis, z_perp)
    assert float(jnp.max(jnp.abs(overlap))) < 1e-3
    # and it is a projection: applying it twice changes nothing
    assert np.allclose(np.asarray(_project_out(basis, z_perp)), np.asarray(z_perp), atol=1e-4)


def test_analytic_gamma_target_beats_identity_for_a_sheared_population():
    """The label response is 1 - eps^2, which is not the identity per object."""
    from shearnet.core.shear_algebra import compose_shear

    e1, e2 = 0.3, 0.2
    step = 1e-6
    d1 = (compose_shear(e1, e2, step, 0.0)[0] - compose_shear(e1, e2, -step, 0.0)[0]) / (2 * step)
    # d(o1)/d(gamma1) = 1 - Re(eps^2) = 1 - (e1^2 - e2^2)
    assert d1 == pytest.approx(1.0 - (e1**2 - e2**2), rel=1e-4)
    # and the composition is exactly the identity at zero applied shear
    assert compose_shear(e1, e2, 0.0, 0.0) == (e1, e2)


def test_response_report_measures_the_expected_derivatives():
    """A linear-in-image model has a response the report must recover."""
    gen, state, forward, _, _ = _harness(n=8, batch=8, apply_psf_shear=True)
    report = make_response_report(gen, forward, 0.0, shear_indices=(0, 1))
    values, cos = report(state.params, jnp.arange(8), jax.random.PRNGKey(0))
    values = np.asarray(values)
    assert values.shape == (len(RESPONSE_REPORT_FIELDS),)
    assert np.all(np.isfinite(values))
    fields = dict(zip(RESPONSE_REPORT_FIELDS, values))
    # the analytic target diagonal is 1 - Re(eps^2) ~ 1 for a shape catalogue
    assert 0.8 < fields["R_gamma_target_11"] < 1.2
    # R^PSF is only measurable because the report always traces the PSF shear
    assert abs(fields["R_psf_11"]) > 0.0
    assert np.asarray(cos).shape == (8,)
    assert np.all((np.asarray(cos) >= -1e-6) & (np.asarray(cos) <= 1.0 + 1e-6))
    assert "Rgamma" in format_response_report(values)


def test_orbit_term_refuses_a_circular_psf():
    """A round PSF makes the orbit a no-op; that must fail, not pass silently."""
    gen, state, forward, loss_fn, labels = _harness(n=8, batch=8)
    with pytest.raises(ValueError, match="identically zero"):
        make_fused_train_step(
            gen,
            forward,
            loss_fn,
            labels,
            0.0,
            response=ResponseRegularization(orbit_weight=1.0),
        )
