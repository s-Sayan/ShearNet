"""Tests for in-loop (inside the jitted step) dataset generation."""

import numpy as np
import pytest

pytest.importorskip("jax_galsim")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
import flax.linen as fnn  # noqa: E402
from flax.training import train_state  # noqa: E402

from shearnet.core.dataset_jax import (  # noqa: E402
    JaxRenderConfig,
    render_truth,
    sample_truth,
)
from shearnet.core.inloop import (  # noqa: E402
    InLoopGenerator,
    build_generator,
    compilation_report,
    make_batch_render,
    make_fused_eval_step,
    make_fused_train_step,
    normalize_state,
)

NPIX, SCALE, FWHM = 53, 0.141, 0.5
X64 = jax.config.jax_enable_x64


def _cfg(exp="ideal", batch=8):
    return JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM, exp=exp,
                           fft_size=256, batch_size=batch)


class _Tiny(fnn.Module):
    @fnn.compact
    def __call__(self, gal, psf):
        x = jnp.stack([gal, psf], axis=-1)
        x = fnn.relu(fnn.Conv(4, (3, 3))(x))
        return fnn.Dense(4)(x.reshape(x.shape[0], -1))


def _harness(exp="ideal", n=32, batch=8):
    cfg = _cfg(exp, batch)
    gen = build_generator(n, cfg, batch, seed=0)
    model = _Tiny()
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, NPIX, NPIX)), jnp.zeros((1, NPIX, NPIX)))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    forward = lambda p, gal, psf, dk: model.apply(p, gal, psf)
    loss_fn = lambda preds, lab: jnp.mean((preds - lab) ** 2)
    labels = gen.labels(("g1", "g2", "hlr", "flux"))
    return gen, state, forward, loss_fn, labels


# ----------------------------------------------------------------------
# bookkeeping
# ----------------------------------------------------------------------
def test_steps_per_epoch_drops_only_the_remainder():
    gen = build_generator(50, _cfg(batch=8), 8, seed=0)
    assert gen.steps_per_epoch == 6                    # 48 of 50
    idx = gen.epoch_indices(jax.random.PRNGKey(0))
    assert idx.shape == (6, 8)
    flat = np.asarray(idx).ravel()
    assert len(set(flat.tolist())) == len(flat)        # no object twice per epoch


def test_epoch_indices_reshuffle_between_epochs():
    gen = build_generator(50, _cfg(batch=8), 8, seed=0)
    a = np.asarray(gen.epoch_indices(jax.random.PRNGKey(0)))
    b = np.asarray(gen.epoch_indices(jax.random.PRNGKey(1)))
    assert not np.array_equal(a, b)                    # remainder is not fixed


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
    rep = compilation_report(gen, forward, loss_fn, labels, 12.7,
                             epochs=3, steps=3, state=state)
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
    assert raw._cache_size() == 2                 # the bug, reproduced

    fixed = make_fused_train_step(gen, forward, loss_fn, labels, 12.7)
    s = normalize_state(state)
    for i in range(3):
        s, _ = fixed(s, idx[i], jax.random.PRNGKey(i), jax.random.PRNGKey(i))
    assert fixed._cache_size() == 1               # and the fix


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
    assert np.allclose(np.asarray(noisy(idx, jax.random.PRNGKey(1))[1]),
                       np.asarray(clean(idx, jax.random.PRNGKey(9))[1]))


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
    got = np.asarray(render(jnp.arange(batch), jax.random.PRNGKey(0))[0],
                     dtype=np.float64)
    want = gal_ref[:batch]
    scale_ = np.abs(want).sum(axis=(1, 2)).max()
    # float32 network cast sets the floor; render agreement is far tighter
    assert np.abs(got - want).max() / scale_ < 1e-6


def test_labels_line_up_with_the_gathered_batch():
    """The loss must see the labels of the objects that were rendered."""
    gen, state, _, _, labels = _harness()
    idx = jnp.array([3, 1, 7, 0, 5, 2, 6, 4])

    zeros = lambda p, gal, psf, dk: jnp.zeros((gal.shape[0], 4), gal.dtype)
    mean_sq = lambda preds, lab: jnp.mean(lab ** 2)
    step = make_fused_train_step(gen, zeros, mean_sq, labels, 0.0)
    _, loss = step(normalize_state(state), idx,
                   jax.random.PRNGKey(0), jax.random.PRNGKey(0))
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
        jax.tree_util.tree_map(lambda a, b: np.asarray(a) - np.asarray(b),
                               new.params, st.params),
        False,
    )
    assert moved, "parameters did not update"
