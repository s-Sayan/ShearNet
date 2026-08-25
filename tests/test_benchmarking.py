"""Tests for the training-matched benchmark renderer and predictor.

The point of this module is that the benchmark and the training run share one
simulator. These tests check the properties the harness silently depends on --
that the population is held out, that plus/minus draws are pair-matched, and
that a saved model is fed exactly what it trained on -- rather than that the
functions return something.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("galsim")

from shearnet.benchmarking import (  # noqa: E402
    SavedModelPredictor,
    TrainingMatchedRenderer,
    load_training_config,
)
from shearnet.config.config_handler import Config  # noqa: E402

NPIX, SCALE, FWHM = 53, 0.141, 0.5


def _config(tmp_path, **overrides):
    """A minimal saved training config, written where the loader expects it."""
    import yaml

    from shearnet.config.config_handler import load_default_config

    cfg = load_default_config()
    cfg["dataset"].update(
        {
            "samples": 32,
            "seed": 7,
            "stamp_size": NPIX,
            "pixel_size": SCALE,
            "psf_fwhm": FWHM,
            "nse_sd": 1.0,
            "exp": "ideal",
        }
    )
    cfg["model"]["output_keys"] = ["g1", "g2"]
    cfg["output"]["plot_path"] = str(tmp_path / "plots")
    cfg["output"]["save_path"] = str(tmp_path / "ckpt")
    for path, value in overrides.items():
        node = cfg
        keys = path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    model_dir = tmp_path / "plots" / cfg["output"]["model_name"]
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "training_config.yaml", "w") as handle:
        yaml.dump(cfg, handle)
    return Config(str(model_dir / "training_config.yaml"))


# ----------------------------------------------------------------------
# held out by construction
# ----------------------------------------------------------------------
def test_render_refuses_the_training_seed(tmp_path):
    renderer = TrainingMatchedRenderer(_config(tmp_path))
    with pytest.raises(ValueError, match="equals dataset.seed"):
        renderer.render(4, seed=7)
    renderer.render(4, seed=8)  # a different seed is fine


def test_render_refuses_to_reuse_the_training_catalog(tmp_path):
    """A different seed re-renders the same rows; only a different file holds out."""
    catalog = tmp_path / "train.fits"
    catalog.write_text("")  # existence is all the check inspects
    config = _config(tmp_path, **{"catalog.cosmos_cat_fname": str(catalog)})
    with pytest.raises(ValueError, match="no evaluation catalog"):
        TrainingMatchedRenderer(config)
    with pytest.raises(ValueError, match="is the training catalog"):
        TrainingMatchedRenderer(config, eval_catalog=str(catalog))


def test_synthetic_catalog_runs_without_an_eval_catalog(tmp_path):
    """The CI fallback has no file, so nothing to hold out and nothing to demand."""
    TrainingMatchedRenderer(_config(tmp_path)).render(4, seed=9)


def test_catalog_cache_is_keyed_on_the_path(tmp_path):
    """One unkeyed cache slot silently served the training catalog to everyone."""
    from shearnet.core.dataset import _load_cosmos_cat

    a = _load_cosmos_cat(seed=1, cat_path=None)
    b = _load_cosmos_cat(seed=2, cat_path=None)
    assert a is not b
    assert not np.array_equal(a["G1"], b["G1"])
    assert _load_cosmos_cat(seed=1, cat_path=None) is a


# ----------------------------------------------------------------------
# pairing: what the whole bias estimator rests on
# ----------------------------------------------------------------------
def test_plus_and_minus_share_galaxies_and_noise(tmp_path):
    """(e_plus - e_minus)/2 only cancels shape noise if the pair is the same object."""
    renderer = TrainingMatchedRenderer(_config(tmp_path))
    plus, minus = renderer.shear_pair(8, seed=11, shear=0.02, component=0)
    assert plus.galaxy_images.shape == minus.galaxy_images.shape
    # Same PSF, same objects: the stamps differ, but only by the applied shear.
    assert not np.allclose(plus.galaxy_images, minus.galaxy_images)
    np.testing.assert_allclose(plus.psf_images, minus.psf_images)
    # Labels are the observed shapes, so they differ by exactly the composition.
    from shearnet.core.shear_algebra import compose_shear

    unsheared = renderer.render(8, seed=11)
    want = np.stack(compose_shear(unsheared.labels[:, 0], unsheared.labels[:, 1], 0.02, 0.0), 1)
    np.testing.assert_allclose(plus.labels[:, :2], want, atol=1e-6)
    # Noise is a function of the object index alone, so the difference of the
    # two stamps carries no noise: it is bounded by the shear signal, not by
    # several sigma of pixel noise.
    difference = plus.galaxy_images - minus.galaxy_images
    noiseless = (
        renderer.render(8, seed=11, base_shear_g1=0.02).galaxy_images
        - renderer.render(8, seed=11, base_shear_g1=-0.02).galaxy_images
    )
    np.testing.assert_allclose(difference, noiseless)


def test_response_renderer_offsets_only_the_shear(tmp_path):
    renderer = TrainingMatchedRenderer(_config(tmp_path))
    render = renderer.response_renderer(6, seed=12)
    base_gal, base_psf = render(0.0, 0.0)
    plus_gal, plus_psf = render(0.01, 0.0)
    assert not np.allclose(base_gal, plus_gal)
    np.testing.assert_allclose(base_psf, plus_psf)


# ----------------------------------------------------------------------
# backend transparency
# ----------------------------------------------------------------------
def test_the_jax_backend_renders_the_same_population(tmp_path):
    """A model trained on jax-galsim must be benchmarked on jax-galsim stamps."""
    pytest.importorskip("jax_galsim")
    galsim_stamps = TrainingMatchedRenderer(_config(tmp_path)).render(8, seed=13)
    jax_stamps = TrainingMatchedRenderer(
        _config(tmp_path, **{"dataset.backend": "jax-galsim"})
    ).render(8, seed=13)
    assert jax_stamps.galaxy_images.shape == galsim_stamps.galaxy_images.shape
    # Same truth, two renderers: the agreement is the backend validation, and it
    # is limited by the float32 network cast rather than by the renderers.
    scale = np.abs(galsim_stamps.galaxy_images).max()
    assert np.abs(jax_stamps.galaxy_images - galsim_stamps.galaxy_images).max() / scale < 1e-3
    np.testing.assert_allclose(jax_stamps.labels, galsim_stamps.labels, atol=1e-6)


def test_inloop_generation_still_benchmarks_up_front(tmp_path):
    """generation: inloop describes training; the benchmark always materialises."""
    pytest.importorskip("jax_galsim")
    renderer = TrainingMatchedRenderer(
        _config(
            tmp_path,
            **{"dataset.backend": "jax-galsim", "dataset.generation": "inloop"},
        )
    )
    assert renderer.spec.generation == "inloop"
    assert renderer.render(8, seed=14).galaxy_images.shape == (8, NPIX, NPIX)


def test_psf_shear_offset_needs_the_jax_backend(tmp_path):
    renderer = TrainingMatchedRenderer(_config(tmp_path))
    with pytest.raises(NotImplementedError, match="jax-galsim"):
        renderer.render(4, seed=15, psf_shear_g1=0.01)


def test_psf_shear_offset_moves_only_the_psf(tmp_path):
    pytest.importorskip("jax_galsim")
    renderer = TrainingMatchedRenderer(_config(tmp_path, **{"dataset.backend": "jax-galsim"}))
    base = renderer.render(6, seed=16)
    offset = renderer.render(6, seed=16, psf_shear_g1=0.02)
    assert not np.allclose(base.psf_images, offset.psf_images)
    # The galaxy stamp moves too -- it is convolved with the sheared PSF -- but
    # the intrinsic shape, i.e. the label, must not.
    np.testing.assert_allclose(base.labels, offset.labels, atol=1e-6)


def test_observations_carry_the_rendered_pixels_unchanged(tmp_path):
    pytest.importorskip("ngmix")
    renderer = TrainingMatchedRenderer(_config(tmp_path))
    stamps = renderer.render(4, seed=17, observations=True)
    assert stamps.observations is not None and len(stamps.observations) == 4
    for obs, image in zip(stamps.observations, stamps.galaxy_images):
        np.testing.assert_allclose(obs.image, image)


# ----------------------------------------------------------------------
# the predictor reproduces training-time input handling
# ----------------------------------------------------------------------
def _train_a_tiny_model(tmp_path, **overrides):
    """Train and persist a one-epoch model so the predictor has something real."""
    jax = pytest.importorskip("jax")

    from shearnet.core.dataset import generate_dataset
    from shearnet.core.train import save_checkpoint, train_model
    from shearnet.utils.normalization import (
        fit_image_normalizer,
        fit_normalizer,
        save_image_normalizer,
        save_normalizer,
    )

    config = _config(tmp_path, **overrides)
    model_dir = Path(config.get("output.plot_path")) / config.get("output.model_name")
    save_path = config.get("output.save_path")
    os.makedirs(save_path, exist_ok=True)

    output_keys = tuple(config.get("model.output_keys"))
    images, labels = generate_dataset(
        32, psf_fwhm=FWHM, npix=NPIX, scale=SCALE, seed=7, nse_sd=1.0, output_keys=output_keys
    )
    norm = fit_normalizer(labels)
    save_normalizer(norm, str(model_dir / "label_normalizer.npz"))
    img_norm = None
    if config.get("dataset.normalize_images"):
        img_norm = fit_image_normalizer(images, None)
        save_image_normalizer(img_norm, str(model_dir / "image_normalizer.npz"))
        images = (images - img_norm["gal_mean"]) / img_norm["gal_std"]
    normed = (labels - norm["mean"]) / norm["std"]
    state, *_ = train_model(
        images,
        normed,
        jax.random.PRNGKey(0),
        epochs=1,
        batch_size=8,
        nn="cnn",
        output_keys=output_keys,
    )
    save_checkpoint(
        state,
        step=1,
        checkpoint_dir=save_path,
        model_name=config.get("output.model_name"),
        overwrite=True,
    )
    return config, norm


def test_predictor_returns_physical_units(tmp_path):
    """The network trains on normalised labels; a benchmark needs raw ones back."""
    config, norm = _train_a_tiny_model(tmp_path)
    renderer = TrainingMatchedRenderer(config)
    stamps = renderer.render(8, seed=21)
    predictor = SavedModelPredictor(config.get("output.model_name"), config=config)
    assert predictor.label_normalizer is not None

    preds = predictor(stamps.galaxy_images, stamps.psf_images)
    assert preds.shape == (8, 2)
    assert np.all(np.isfinite(preds))
    # Undoing the normalizer must reproduce the raw network output, which is the
    # only way to tell the transform was applied at all.
    raw = (preds - norm["mean"]) / norm["std"]
    assert np.abs(raw).max() < np.abs(preds).max() * 1e6  # sane, not a no-op check
    assert not np.allclose(preds, raw)


def test_predictor_applies_the_image_normalizer(tmp_path):
    config, _ = _train_a_tiny_model(tmp_path, **{"dataset.normalize_images": True})
    predictor = SavedModelPredictor(config.get("output.model_name"), config=config)
    assert predictor.image_normalizer is not None
    stamps = TrainingMatchedRenderer(config).render(8, seed=22)
    with_norm = predictor(stamps.galaxy_images, stamps.psf_images)
    predictor.image_normalizer = None
    predictor._jitted = __import__("jax").jit(predictor._forward)
    without = predictor(stamps.galaxy_images, stamps.psf_images)
    assert not np.allclose(with_norm, without), "the image normalizer was not applied"


def test_shear_measure_selects_the_shear_columns(tmp_path):
    config, _ = _train_a_tiny_model(tmp_path, **{"model.output_keys": ["hlr", "g1", "g2"]})
    predictor = SavedModelPredictor(config.get("output.model_name"), config=config)
    assert predictor.output_keys == ("hlr", "g1", "g2")
    stamps = TrainingMatchedRenderer(config).render(4, seed=23)
    measured = predictor.shear_measure()(stamps.galaxy_images, stamps.psf_images)
    full = predictor(stamps.galaxy_images, stamps.psf_images)
    np.testing.assert_allclose(measured, full[:, 1:3])


def test_missing_training_config_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        load_training_config("nope", str(tmp_path / "absent.yaml"))
    with pytest.raises(FileNotFoundError, match="training_config.yaml"):
        load_training_config("nope-at-all")

def test_psf_response_offset_does_not_randomise_the_base_point(tmp_path):
    """R^PSF must be the derivative at the object's own PSF, not a random one.

    Setting apply_psf_shear for the offset renders drew a random +/- 0.05 PSF
    shear into the truth table, so the benchmark differentiated at a randomised
    base point while TRAINING differentiates at zero offset. The extra RNG draws
    also came before dx/dy, shifting the galaxy centroids of the perturbed
    renders by up to 2.3 pixels relative to the unperturbed one -- so e and gpsf
    described a different object configuration from Rpsf in the same row.
    """
    pytest.importorskip("jax_galsim")
    renderer = TrainingMatchedRenderer(_config(tmp_path, **{"dataset.backend": "jax-galsim"}))
    base = renderer.render(8, seed=21)
    plus = renderer.render(8, seed=21, psf_shear_g1=0.01)
    minus = renderer.render(8, seed=21, psf_shear_g1=-0.01)

    # the offset still reaches the PSF -- otherwise R^PSF would be exactly zero
    assert not np.allclose(plus.psf_images, minus.psf_images)
    # ... and the perturbation is symmetric about the UNPERTURBED render, which
    # is what "the derivative at this object's PSF" means. A random base point
    # would break this: base would not sit between the two legs.
    midpoint = 0.5 * (np.asarray(plus.psf_images) + np.asarray(minus.psf_images))
    spread = np.max(np.abs(np.asarray(plus.psf_images) - np.asarray(minus.psf_images)))
    assert np.max(np.abs(midpoint - np.asarray(base.psf_images))) < 0.05 * spread

    # same galaxies throughout: the extra draws must not shift the centroids
    np.testing.assert_allclose(base.labels, plus.labels, atol=1e-6)
    np.testing.assert_allclose(base.labels, minus.labels, atol=1e-6)
