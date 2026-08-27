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


# ----------------------------------------------------------------------
# shape-noise cancellation: the 90-degree rotated twin
# ----------------------------------------------------------------------
def test_rotating_the_intrinsic_shape_needs_the_jax_backend(tmp_path):
    renderer = TrainingMatchedRenderer(_config(tmp_path))
    with pytest.raises(NotImplementedError, match="jax-galsim"):
        renderer.render(4, seed=21, intrinsic_rotation=90.0)


def test_rotating_the_intrinsic_shape_flips_the_label_and_keeps_the_psf(tmp_path):
    """The twin is the same galaxy turned 90 degrees under the SAME PSF.

    Both halves matter. Flipping the label is what makes the intrinsic shape
    cancel in the pair mean; leaving the PSF alone is what keeps PSF leakage in
    the pair mean, where it can still be measured.
    """
    pytest.importorskip("jax_galsim")
    renderer = TrainingMatchedRenderer(_config(tmp_path, **{"dataset.backend": "jax-galsim"}))
    base = renderer.render(8, seed=22)
    twin = renderer.render(8, seed=22, intrinsic_rotation=90.0)

    # spin-2 doubles the angle, so a 90-degree rotation is a sign flip on both
    # components -- and with no applied shear the label IS the intrinsic shape
    np.testing.assert_allclose(twin.labels[:, :2], -base.labels[:, :2], atol=1e-6)
    # size and flux are rotation invariant
    np.testing.assert_allclose(twin.labels[:, 2:], base.labels[:, 2:], atol=1e-6)
    # the PSF is deliberately NOT rotated
    np.testing.assert_allclose(twin.psf_images, base.psf_images)
    # ...but the galaxy stamp is a different image
    assert not np.allclose(twin.galaxy_images, base.galaxy_images)


def test_the_rotated_twin_carries_its_own_noise(tmp_path):
    """rot90 of an i.i.d. field is an independent field, so noise averages down.

    Repeating the same noise realisation in the twin would make the pair mean
    keep all of it. Rotating the field about the stamp centre correlates only
    the single fixed pixel, so the twin is effectively an independent draw and
    the pair mean gets the sqrt(2) as well as the shape cancellation.
    """
    pytest.importorskip("jax_galsim")
    renderer = TrainingMatchedRenderer(
        _config(tmp_path, **{"dataset.backend": "jax-galsim"})
    )
    clean = renderer.render(4, seed=23, add_noise=False)
    clean_twin = renderer.render(4, seed=23, intrinsic_rotation=90.0, add_noise=False)
    noisy = renderer.render(4, seed=23)
    noisy_twin = renderer.render(4, seed=23, intrinsic_rotation=90.0)

    noise = noisy.galaxy_images - clean.galaxy_images
    noise_twin = noisy_twin.galaxy_images - clean_twin.galaxy_images
    np.testing.assert_allclose(noise_twin, np.rot90(noise, k=1, axes=(-2, -1)), atol=1e-5)
    # same amplitude, different realisation
    assert np.std(noise_twin) == pytest.approx(np.std(noise), rel=0.05)
    assert not np.allclose(noise_twin, noise)


def test_a_45_degree_rotation_sends_e_to_i_times_e(tmp_path):
    """Spin-2: 45 degrees on the sky is a quarter turn in shape space.

    That is the rotation that flips the sign of ``eps^2``, which is the term
    that biases the truth-referenced ``m`` -- 90 degrees leaves it alone,
    because ``(-eps)^2 = eps^2``.
    """
    pytest.importorskip("jax_galsim")
    renderer = TrainingMatchedRenderer(_config(tmp_path, **{"dataset.backend": "jax-galsim"}))
    base = renderer.render(8, seed=26)
    turned = renderer.render(8, seed=26, intrinsic_rotation=45.0)
    e1, e2 = base.labels[:, 0], base.labels[:, 1]
    np.testing.assert_allclose(turned.labels[:, 0], -e2, atol=1e-6)
    np.testing.assert_allclose(turned.labels[:, 1], e1, atol=1e-6)
    # eps^2 has flipped sign; |eps| has not moved
    np.testing.assert_allclose(
        turned.labels[:, 0] ** 2 - turned.labels[:, 1] ** 2, -(e1**2 - e2**2), atol=1e-6
    )


def test_the_ring_cancels_the_intrinsic_shape(tmp_path):
    """The mean over the ring is the applied shear, not the galaxy's own shape.

    Asserted on the labels -- exact and noise-free -- rather than on a
    measurement, because this is a property of the construction and not of any
    estimator. The 90-degree pair removes the intrinsic shape; what it leaves
    behind is the ``O(eps^2 gamma)`` term of the Moebius composition, and the
    full ring removes that too.
    """
    pytest.importorskip("jax_galsim")
    renderer = TrainingMatchedRenderer(_config(tmp_path, **{"dataset.backend": "jax-galsim"}))
    shear = 0.02
    stations = {
        degrees: renderer.render(
            64, seed=24, base_shear_g1=shear, intrinsic_rotation=degrees
        ).labels[:, :2]
        for degrees in (0.0, 45.0, 90.0, 135.0)
    }

    single = stations[0.0]
    pair = 0.5 * (stations[0.0] + stations[90.0])
    ring = np.mean(list(stations.values()), axis=0)

    # the intrinsic shape is gone from every object, not just from the ensemble
    assert np.std(pair[:, 0]) < 0.05 * np.std(single[:, 0])
    # ...and the residual the pair leaves is the eps^2 term, which the full ring
    # removes by another two orders of magnitude
    pair_residual = np.abs(pair[:, 0] - shear).max()
    ring_residual = np.abs(ring[:, 0] - shear).max()
    assert ring_residual < 0.02 * pair_residual, (ring_residual, pair_residual)
    assert ring_residual < 1e-4, ring_residual
    assert np.abs(ring[:, 1]).max() < 1e-4


def test_shear_pair_and_response_renderer_carry_the_rotation(tmp_path):
    pytest.importorskip("jax_galsim")
    renderer = TrainingMatchedRenderer(_config(tmp_path, **{"dataset.backend": "jax-galsim"}))
    plus, minus = renderer.shear_pair(6, seed=25, shear=0.02, intrinsic_rotation=90.0)
    straight, _ = renderer.shear_pair(6, seed=25, shear=0.02)
    assert not np.allclose(plus.galaxy_images, straight.galaxy_images)
    # the +/- pairing survives the rotation: same galaxies, opposite shear
    np.testing.assert_allclose(plus.psf_images, minus.psf_images)

    render = renderer.response_renderer(6, seed=25, intrinsic_rotation=90.0)
    rotated_gal, _ = render(0.0, 0.0)
    plain_gal, _ = renderer.response_renderer(6, seed=25)(0.0, 0.0)
    assert not np.allclose(rotated_gal, plain_gal)
