"""Tests for the in-loop training driver and its config wiring."""

import numpy as np
import pytest

pytest.importorskip("jax_galsim")
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from shearnet.core import inloop as inloop_mod  # noqa: E402
from shearnet.core.dataset_jax import JaxRenderConfig  # noqa: E402
from shearnet.core.inloop import (  # noqa: E402
    InLoopGenerator,
    ResponseRegularization,
    noise_schedule_from_config,
    sample_truth,
)
from shearnet.core.specs import DatasetSpec  # noqa: E402
from shearnet.core.train_inloop import train_model_inloop  # noqa: E402

NPIX, SCALE, FWHM = 33, 0.141, 0.5


def _gen(n=128, batch=16, exp="ideal", **truth_kw):
    cfg = JaxRenderConfig(
        npix=NPIX, scale=SCALE, psf_fwhm=FWHM, exp=exp, fft_size=256, batch_size=batch
    )
    truth = sample_truth(n, cfg, seed=0, nse_sd=12.7, add_noise=False, **truth_kw)
    return InLoopGenerator(truth, cfg, batch)


def _train(gen, **kw):
    kw.setdefault("output_keys", ("g1", "g2", "hlr", "flux"))
    kw.setdefault("nse_sd", 12.7)
    kw.setdefault("epochs", 2)
    kw.setdefault("nn", "cnn")
    kw.setdefault("val_split", 0.25)
    return train_model_inloop(gen, jax.random.PRNGKey(0), **kw)


# ----------------------------------------------------------------------
# the loop runs and learns
# ----------------------------------------------------------------------
def test_training_runs_and_returns_the_train_model_tuple():
    state, train_losses, val_losses, per_key = _train(_gen(), epochs=3)
    assert len(train_losses) == 3 and len(val_losses) == 3
    assert np.all(np.isfinite(train_losses)) and np.all(np.isfinite(val_losses))
    assert np.asarray(per_key).shape == (3, 4)
    # a smoke-level sanity check that optimisation is happening at all
    assert train_losses[-1] < train_losses[0]


def test_fork_model_path():
    state, train_losses, _, _ = _train(
        _gen(), nn="fork-like", galaxy_type="research_backed", psf_type="forklens_psf", epochs=2
    )
    assert np.all(np.isfinite(train_losses))


def test_dropout_path_runs():
    _, train_losses, _, _ = _train(_gen(), nn="cnn", dropout=0.1, epochs=2)
    assert np.all(np.isfinite(train_losses))


def test_ema_path_runs_and_returns_averaged_weights():
    state_a, _, _, _ = _train(_gen(), epochs=2)
    state_b, _, _, _ = _train(_gen(), epochs=2, ema_decay=0.9)
    diff = jax.tree_util.tree_reduce(
        lambda acc, x: acc or bool(np.any(np.asarray(x) != 0)),
        jax.tree_util.tree_map(
            lambda a, b: np.asarray(a) - np.asarray(b), state_a.params, state_b.params
        ),
        False,
    )
    assert diff, "EMA weights should differ from the live weights"


def test_label_normalization_is_applied():
    """A normalizer with a huge std must shrink the loss it reports."""
    gen = _gen()
    raw = np.asarray(gen.labels(("g1", "g2", "hlr", "flux")))
    norm = {"mean": raw.mean(axis=0), "std": raw.std(axis=0) + 1e-12}
    _, plain, _, _ = _train(gen, epochs=1)
    _, normed, _, _ = _train(gen, epochs=1, label_norm=norm)
    assert not np.isclose(plain[0], normed[0])


def test_image_normalization_reaches_the_step():
    gen = _gen()
    big = {"gal_mean": 0.0, "gal_std": 1e6, "psf_mean": 0.0, "psf_std": 1e6}
    _, plain, _, _ = _train(gen, epochs=1)
    _, scaled, _, _ = _train(gen, epochs=1, img_norm=big)
    assert not np.isclose(plain[0], scaled[0])


# ----------------------------------------------------------------------
# compilation must survive the integration
# ----------------------------------------------------------------------
def test_driver_compiles_the_fused_steps_once(monkeypatch):
    """The epoch loop must not reintroduce a shape or leaf-type change."""
    captured = {}
    real_train = inloop_mod.make_fused_train_step
    real_eval = inloop_mod.make_fused_eval_step

    def spy_train(*a, **kw):
        captured["train"] = real_train(*a, **kw)
        return captured["train"]

    def spy_eval(*a, **kw):
        captured["eval"] = real_eval(*a, **kw)
        return captured["eval"]

    monkeypatch.setattr("shearnet.core.train_inloop.make_fused_train_step", spy_train)
    monkeypatch.setattr("shearnet.core.train_inloop.make_fused_eval_step", spy_eval)

    _train(_gen(n=128, batch=16), epochs=4)
    assert captured["train"]._cache_size() == 1
    assert captured["eval"]._cache_size() == 1


def test_validation_batches_are_in_fixed_order():
    """Frozen val noise depends on val batches not being reshuffled."""
    gen = _gen()
    ids = jnp.arange(96, 128)
    a = np.asarray(gen.batches(ids))
    b = np.asarray(gen.batches(ids))
    assert np.array_equal(a, b)
    shuffled = np.asarray(gen.batches(ids, key=jax.random.PRNGKey(3)))
    assert not np.array_equal(a, shuffled)


def test_batches_rejects_a_split_too_small_for_one_batch():
    gen = _gen(batch=16)
    with pytest.raises(ValueError, match="cannot fill a batch"):
        gen.batches(jnp.arange(4))


def test_rejects_split_leaving_no_validation_batch():
    gen = _gen(n=128, batch=16)
    with pytest.raises(ValueError, match="need at least one of each"):
        _train(gen, val_split=0.01)


def test_rejects_mismatched_loss_weights():
    with pytest.raises(ValueError, match="loss_weights length"):
        _train(_gen(), weights=[1.0, 1.0])


# ----------------------------------------------------------------------
# config wiring
# ----------------------------------------------------------------------
def test_spec_defaults_to_upfront():
    spec = DatasetSpec(samples=8, psf_fwhm=FWHM)
    assert spec.generation == "upfront"
    assert "generation" not in spec.as_kwargs()


def test_spec_inloop_requires_the_jax_backend():
    with pytest.raises(ValueError, match="requires dataset.backend: jax-galsim"):
        DatasetSpec(samples=8, psf_fwhm=FWHM, generation="inloop")
    DatasetSpec(
        samples=8, psf_fwhm=FWHM, generation="inloop", backend="jax-galsim"
    )  # valid combination


def test_spec_rejects_unknown_generation():
    with pytest.raises(ValueError, match="dataset.generation"):
        DatasetSpec(samples=8, psf_fwhm=FWHM, generation="lazy", backend="jax-galsim")


def test_spec_build_refuses_to_materialise_for_inloop():
    spec = DatasetSpec(samples=64, psf_fwhm=FWHM, generation="inloop", backend="jax-galsim")
    with pytest.raises(ValueError, match="never does"):
        spec.build()


def test_spec_build_inloop_generator():
    spec = DatasetSpec(
        samples=64,
        psf_fwhm=FWHM,
        npix=NPIX,
        seed=0,
        generation="inloop",
        backend="jax-galsim",
        jax_batch_size=16,
    )
    gen = spec.build_inloop_generator(16)
    assert gen.n == 64 and gen.batch_size == 16
    assert gen.steps_per_epoch == 4


def test_spec_build_inloop_generator_rejects_upfront():
    spec = DatasetSpec(samples=64, psf_fwhm=FWHM, backend="jax-galsim")
    with pytest.raises(ValueError, match="requires dataset.generation: inloop"):
        spec.build_inloop_generator(16)


def test_config_exposes_generation_default():
    from shearnet.config.config_handler import load_default_config

    cfg = load_default_config()
    assert cfg["dataset"]["generation"] == "upfront"
    assert cfg["training"]["response"]["orbit_weight"] == 0.0


# ----------------------------------------------------------------------
# response regularisation through the driver
# ----------------------------------------------------------------------
def test_response_training_runs_and_still_compiles_once(monkeypatch):
    """Enabling every response term must not cost a second executable."""
    captured = {}
    real_train = inloop_mod.make_fused_train_step

    def spy(*a, **kw):
        captured["train"] = real_train(*a, **kw)
        return captured["train"]

    monkeypatch.setattr("shearnet.core.train_inloop.make_fused_train_step", spy)
    gen = _gen(n=64, batch=16, apply_psf_shear=True)
    _, train_losses, _, _ = _train(
        gen,
        epochs=3,
        response=ResponseRegularization(
            gamma_weight=1e-3,
            psf_weight=1e-3,
            shift_weight=1e-3,
            complement_weight=1e-3,
            orbit_weight=1e-3,
        ),
    )
    assert np.all(np.isfinite(train_losses))
    assert captured["train"]._cache_size() == 1


def test_response_requires_the_shear_output_keys():
    with pytest.raises(ValueError, match="requires g1 and g2"):
        _train(
            _gen(), output_keys=("hlr", "flux"), response=ResponseRegularization(gamma_weight=1.0)
        )


def test_response_finds_the_shear_keys_wherever_they_sit():
    """shear_indices must follow output_keys, not assume positions 0 and 1."""
    seen = {}
    real = inloop_mod.make_fused_train_step

    def spy(*a, **kw):
        seen["shear_indices"] = kw["shear_indices"]
        return real(*a, **kw)

    gen = _gen(n=64, batch=16, apply_psf_shear=True)
    import unittest.mock as mock

    with mock.patch("shearnet.core.train_inloop.make_fused_train_step", spy):
        _train(
            gen,
            epochs=1,
            output_keys=("hlr", "flux", "g1", "g2"),
            response=ResponseRegularization(psf_weight=1e-3),
        )
    assert seen["shear_indices"] == (2, 3)


def test_noise_range_validation_is_evaluated_at_the_midpoint():
    seen = {}
    real = inloop_mod.make_fused_eval_step

    def spy(gen, forward, loss_fn, labels, nse_sd, **kw):
        seen["nse_sd"] = nse_sd
        return real(gen, forward, loss_fn, labels, nse_sd, **kw)

    import unittest.mock as mock

    with mock.patch("shearnet.core.train_inloop.make_fused_eval_step", spy):
        _train(_gen(), epochs=1, noise_range=(5.0, 15.0))
    assert seen["nse_sd"] == pytest.approx(10.0)


def test_noise_conditioning_rejects_image_normalization():
    with pytest.raises(ValueError, match="noise conditioning"):
        _train(_gen(), epochs=1, noise_condition=True, img_norm={"gal_mean": 0.0, "gal_std": 1.0})


# ----------------------------------------------------------------------
# config-block parsing
# ----------------------------------------------------------------------
def test_response_block_rejects_a_misspelled_weight():
    """An unrecognised key would leave the term at zero and run the control."""
    with pytest.raises(ValueError, match="unknown training.response keys"):
        ResponseRegularization.from_config({"psf_wieght": 0.1})
    parsed = ResponseRegularization.from_config({"psf_weight": 0.1, "orbit_k": 4, "report": True})
    assert parsed.psf_weight == 0.1 and parsed.orbit_k == 4
    assert ResponseRegularization.from_config(None) == ResponseRegularization()


def test_noise_block_parsing():
    assert noise_schedule_from_config({}) == (None, False)
    assert noise_schedule_from_config({"min_sd": 1, "max_sd": 3}) == ((1.0, 3.0), False)
    with pytest.raises(ValueError, match="both min_sd and max_sd"):
        noise_schedule_from_config({"min_sd": 1})
    with pytest.raises(ValueError, match="unknown training.noise keys"):
        noise_schedule_from_config({"minimum_sd": 1})
    with pytest.raises(ValueError, match="condition"):
        noise_schedule_from_config({"condition": True})


def test_default_config_response_block_parses():
    from shearnet.config.config_handler import load_default_config

    cfg = load_default_config()
    assert not ResponseRegularization.from_config(cfg["training"]["response"]).enabled
    assert noise_schedule_from_config(cfg["training"]["noise"]) == (None, False)
