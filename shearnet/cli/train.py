"""Command-line interface for training ShearNet models."""

import argparse
import hashlib
import logging
import os

import jax.numpy as jnp
import jax.random as random
import numpy as np
import yaml

import shearnet.core.models

from .. import __version__
from ..config.config_handler import Config, ConfigError
from ..core.augment import d4_augment
from ..core.dataset import split_combined_images
from ..core.models import is_fork_model
from ..core.specs import DatasetSpec, TrainConfig
from ..logging_utils import get_logger
from ..plotting import plot_learning_curve
from ..utils.device import get_device
from ..utils.normalization import (
    fit_image_normalizer,
    fit_normalizer,
    identity_normalizer,
    save_image_normalizer,
    save_normalizer,
    transform_images,
    transform_labels,
)

logger = get_logger(__name__)

# Suppress noisy absl logging emitted by JAX (importing the package also does
# this, before JAX is imported, so import-time messages are already silenced).
logging.getLogger("absl").setLevel(logging.ERROR)


def create_parser():
    """Create argument parser for training."""
    data_path = os.getenv("SHEARNET_DATA_PATH", os.path.abspath("."))
    parser = argparse.ArgumentParser(
        description="Train a galaxy shear estimation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  shearnet-train --config configs/example.yaml

Every setting lives in the YAML; see shearnet/config/schema.py for the list.
        """,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config")
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join(data_path, "model_checkpoint"),
        help="Path to save the model parameters.",
    )
    parser.add_argument(
        "--plot_path",
        type=str,
        default=os.path.join(data_path, "plots"),
        help="Path to save the learning curve, config and normalizers.",
    )
    return parser


def build_train_config(args):
    """Load and validate ``--config``. Performs no simulation or training."""
    config = Config.from_file(args.config)
    if not config.get("run_options.run_name"):
        raise ConfigError(f"{args.config}: set run_options.run_name")
    logger.info(f"\nUsing config file: {args.config}")
    return config


def _prepare_training_data(config):
    """Simulate the training set and standardize its labels (and, optionally, images).

    Reads the dataset/model settings from ``config``, generates the stamps,
    splits off the PSF channel for the two-branch model, and (when
    ``dataset.normalize_labels`` is true, the default) fits the label normalizer
    on the training portion only; when false, an identity normalizer is used so
    the network trains on raw labels. Two further opt-in transforms may also be
    applied here, both fit on the training portion only:

    * ``dataset.normalize_images``: dataset-level input standardization
      (independent of the label normalizer -- see
      :mod:`shearnet.utils.normalization`).
    * ``dataset.d4_augment``: 8x D4 augmentation of the *training* portion only
      (ABLATION ONLY -- see :mod:`shearnet.core.augment`).

    Returns:
        ``(galaxy_images, psf_images, labels, norm_params, img_params,
        eff_val_split, resample_noise_sd)``. ``psf_images`` is ``None`` for
        single-branch models; ``labels`` is already normalized; ``img_params`` is
        ``None`` unless image normalization was enabled; ``eff_val_split`` is the
        validation fraction ``train_model`` must use so its internal split lands
        on the same train/val boundary after any augmentation (equal to
        ``training.val_split`` when augmentation is off); ``resample_noise_sd`` is
        the per-step noise std in model-input units for the fresh-noise path
        (``0.0`` unless ``training.resample_noise`` is set). With fresh noise the
        galaxy stamps are returned noise-free (noise is added each epoch in
        ``train_model``).
    """
    spec = DatasetSpec.from_config(config)
    val_split = config.get("training.val_split")
    normalize_labels = config.get("training.normalize_labels")
    normalize_images = config.get("training.normalize_images")
    do_d4_augment = config.get("training.d4_augment")
    output_keys = tuple(config.get("model.output_keys"))

    galaxy_images, labels = spec.build()
    # Split off the PSF channel only when the two-branch model needs it;
    # single-branch models leave psf_images as None.
    psf_images = None
    if spec.return_psf:
        galaxy_images, psf_images = split_combined_images(
            galaxy_images, has_psf=True, has_clean=False
        )
        logger.info(f"Shape of train PSF images: {psf_images.shape}")
    logger.info(f"Shape of train images: {galaxy_images.shape}")
    logger.info(f"Shape of train labels: {labels.shape}")

    # Everything below fits on the TRAIN portion only. Split first (on raw data)
    # so augmentation and the normalizers never see the validation stamps.
    split_idx = int(len(labels) * (1 - val_split))
    eff_val_split = val_split

    if do_d4_augment:
        # ABLATION ONLY: augment the train portion 8x, keep val untouched, and
        # reassemble as [aug_train, val]. eff_val_split is recomputed so
        # train_model's internal fractional split reproduces this exact boundary.
        gal_tr, psf_tr, lab_tr = (
            galaxy_images[:split_idx],
            (psf_images[:split_idx] if psf_images is not None else None),
            labels[:split_idx],
        )
        gal_val, psf_val, lab_val = (
            galaxy_images[split_idx:],
            (psf_images[split_idx:] if psf_images is not None else None),
            labels[split_idx:],
        )

        gal_tr, psf_tr, lab_tr = d4_augment(gal_tr, psf_tr, lab_tr, output_keys)
        logger.info(f"D4 augmentation (ABLATION) on: train {split_idx} -> {len(lab_tr)} stamps.")

        galaxy_images = np.concatenate([gal_tr, gal_val], axis=0)
        labels = np.concatenate([lab_tr, lab_val], axis=0)
        if psf_images is not None:
            psf_images = np.concatenate([psf_tr, psf_val], axis=0)
        split_idx = len(lab_tr)
        eff_val_split = len(lab_val) / len(labels)

    # Fit the label normalizer on the (possibly augmented) training portion.
    # When disabled (ablation), use an identity normalizer so the network sees
    # raw labels while the save/load/eval paths stay unchanged.
    if normalize_labels:
        norm_params = fit_normalizer(labels[:split_idx])
    else:
        norm_params = identity_normalizer(labels)
    labels = transform_labels(labels, norm_params)

    # Optional dataset-level image standardization, fit on the same training
    # portion and applied to galaxy (and PSF) stamps. Independent of the label
    # normalizer above.
    resample_noise = config.get("training.resample_noise")
    nse_sd = spec.nse_sd
    img_params = None
    if normalize_images:
        img_params = fit_image_normalizer(
            galaxy_images[:split_idx],
            psf_images[:split_idx] if psf_images is not None else None,
        )
        if resample_noise:
            # With fresh-noise training the stamps here are noise-free, so their
            # galaxy std understates what the network (and eval, on real noisy
            # stamps) actually sees. Inflate it to the noisy-equivalent
            # sqrt(var_clean + nse_sd^2) -- exact in expectation since the added
            # noise is independent and zero-mean -- so the input scale and the
            # saved normalizer match a standard baked-noise run. Mean is unchanged.
            img_params["gal_std"] = float(np.sqrt(img_params["gal_std"] ** 2 + nse_sd**2))
        galaxy_images = transform_images(galaxy_images, img_params, channel="gal")
        if psf_images is not None:
            psf_images = transform_images(psf_images, img_params, channel="psf")

    # Fresh-noise training re-noises the (now normalized) clean stamps every epoch
    # inside train_model; express the physical noise std in model-input units
    # (divide by the galaxy normalizer std, or 1.0 when image norm is off).
    gal_std = img_params["gal_std"] if img_params is not None else 1.0
    resample_noise_sd = (nse_sd / gal_std) if resample_noise else 0.0

    return (
        galaxy_images,
        psf_images,
        labels,
        norm_params,
        img_params,
        eff_val_split,
        resample_noise_sd,
    )


def _run_inloop_training(config, rng_key, model_dir, save_path):
    """Train with ``dataset.generation: inloop`` -- no dataset is materialised.

    Mirrors ``_prepare_training_data`` where it must: the label normalizer is
    fit on the same contiguous training portion, and image standardisation is
    fit on rendered stamps and then applied *inside* the step (there is nowhere
    else to apply it, since the stamps never exist outside the step).

    D4 augmentation is not applicable here -- it is an ablation that duplicates
    a materialised array, and in-loop generation has no array to duplicate.
    """
    import jax
    import jax.numpy as jnp

    from ..core.dataset_jax import render_dtype
    from ..core.inloop import (
        ResponseRegularization,
        make_batch_render,
        noise_schedule_from_config,
    )
    from ..core.train_inloop import train_model_inloop

    spec = DatasetSpec.from_config(config)
    batch_size = config.get("training.batch_size")
    output_keys = tuple(config.get("model.output_keys"))
    response = ResponseRegularization.from_config(config.get("training.response"))
    response_report = (config.get("training.response") or {}).get("report", None)
    noise_range, noise_condition = noise_schedule_from_config(config.get("training.noise"))
    if not jax.config.jax_enable_x64:
        # Not fatal -- float32 stamps are fine for training -- but a forgotten
        # JAX_ENABLE_X64=1 is invisible otherwise, and it is the difference
        # between a float64 and a float32 renderer for every response term.
        logger.warning(
            "JAX_ENABLE_X64 is not set: in-loop rendering will run in float32 "
            "(render accuracy ~1e-7). Export JAX_ENABLE_X64=1 before starting "
            "for float64; the stamps are cast to float32 before the network, "
            "so training memory and speed are unchanged either way."
        )
    logger.info("in-loop generation: render dtype %s", render_dtype().__name__)

    gen = spec.build_inloop_generator(batch_size)

    # Label normalizer, fit on the same contiguous train portion the trainer
    # will use, so the scale matches an up-front run with the same seed.
    val_split = config.get("training.val_split")
    split_idx = int(gen.n * (1 - val_split))
    raw_labels = np.asarray(gen.labels(output_keys))
    if config.get("training.normalize_labels"):
        norm_params = fit_normalizer(raw_labels[:split_idx])
    else:
        norm_params = identity_normalizer(raw_labels)

    # Image normalizer: render a few batches to fit it. Cheap relative to a run,
    # and it keeps the input scale identical to the up-front path.
    img_params = None
    if config.get("training.normalize_images"):
        probe = make_batch_render(gen, nse_sd=spec.nse_sd)
        n_probe = min(4, gen.steps_per_epoch)
        ids = gen.batches(jnp.arange(split_idx))[:n_probe]
        gals, psfs = [], []
        for s in range(n_probe):
            g, p = probe(ids[s], jax.random.fold_in(rng_key, s))
            gals.append(np.asarray(g))
            psfs.append(np.asarray(p))
        img_params = fit_image_normalizer(
            np.concatenate(gals),
            np.concatenate(psfs) if is_fork_model(config.get("model.type")) else None,
        )
        logger.info("image normalizer fit on %d rendered stamps", n_probe * batch_size)

    _save_run_artifacts(config, model_dir, norm_params, img_params)

    return train_model_inloop(
        gen,
        rng_key,
        output_keys=output_keys,
        label_norm=norm_params,
        img_norm=img_params,
        nse_sd=spec.nse_sd,
        epochs=config.get("training.epochs"),
        nn=config.get("model.type"),
        galaxy_type=config.get("model.galaxy_branch"),
        psf_type=config.get("model.psf_branch"),
        fusion=config.get("model.fusion"),
        head=config.get("model.head"),
        save_path=save_path,
        model_name=config.get("run_options.run_name"),
        val_split=val_split,
        eval_interval=config.get("training.eval_interval"),
        patience=config.get("training.patience"),
        lr=config.get("training.learning_rate"),
        weight_decay=config.get("training.weight_decay"),
        gap=config.get("model.gap"),
        weights=config.get("training.loss_weights"),
        loss=config.get("training.loss"),
        ema_decay=config.get("training.ema_decay"),
        dropout=config.get("model.dropout"),
        branch_features=config.get("model.branch_features"),
        d4_features=config.get("model.d4_features"),
        d4_depths_galaxy=config.get("model.d4_depths_galaxy"),
        d4_depths_psf=config.get("model.d4_depths_psf"),
        d4_multiscale=config.get("model.d4_multiscale"),
        orbit_scan=config.get("model.orbit_scan"),
        fusion_pos=config.get("model.fusion_pos"),
        design=config.get("model.design"),
        d_model=config.get("model.d_model"),
        num_heads=config.get("model.num_heads"),
        num_pool_heads=config.get("model.num_pool_heads"),
        num_self_attn_layers=config.get("model.num_self_attn_layers"),
        ffn_dim=config.get("model.ffn_dim"),
        response=response,
        noise_range=noise_range,
        noise_condition=noise_condition,
        response_report=response_report,
        # dataset.apply_psf_shear draws a per-object PSF shear into the truth
        # table; without this the renderer would never apply it, and the run
        # would silently train on round PSFs. (The response terms switch the
        # transform on for themselves regardless -- they need the tangent.)
        trace_psf_shear=config.get("simulation.apply_psf_shear"),
    )


def _save_run_artifacts(config, model_dir, norm_params, img_params=None):
    """Persist the resolved config (with provenance) and the normalizers.

    Provenance — the ShearNet version and a sha256 of the model-source file — is
    recorded into the saved config instead of copying ``models.py`` to
    ``architecture.py``, so a run can be tied back to the exact architecture code
    without snapshotting the source.
    """
    os.makedirs(model_dir, exist_ok=True)

    provenance = {"shearnet_version": __version__}
    try:
        with open(shearnet.core.models.__file__, "rb") as f:
            provenance["models_sha256"] = hashlib.sha256(f.read()).hexdigest()
    except OSError as e:
        logger.warning(f"WARNING: could not hash model source for provenance: {e}")
    with open(os.path.join(model_dir, "provenance.yaml"), "w") as f:
        yaml.safe_dump(provenance, f, sort_keys=False)

    config_path = os.path.join(model_dir, "training_config.yaml")
    config.save(config_path)
    logger.info(f"\nTraining configuration saved to: {config_path}")

    normalizer_path = os.path.join(model_dir, "label_normalizer.npz")
    save_normalizer(norm_params, normalizer_path)

    # Image normalizer travels next to the label normalizer; eval and the
    # research benchmarks pick it up via normalization.maybe_normalize_images.
    if img_params is not None:
        save_image_normalizer(img_params, os.path.join(model_dir, "image_normalizer.npz"))


def _save_losses(loss_path, train_loss, val_loss, val_loss_per_key, output_keys):
    """Save the per-epoch loss histories to ``loss_path`` (no-op if ``None``)."""
    logger.info("Saving training and validation loss...")
    if loss_path is None:
        return
    val_loss_per_key_arr = (
        jnp.stack(val_loss_per_key) if val_loss_per_key else jnp.zeros((0, len(output_keys)))
    )
    jnp.savez(
        loss_path,
        train_loss=train_loss,
        val_loss=val_loss,
        val_loss_per_key=val_loss_per_key_arr,
        output_keys=output_keys,
    )


def main():
    """Run the model-training command-line interface."""
    parser = create_parser()
    args = parser.parse_args()

    config = build_train_config(args)
    logger.info(config.to_yaml())

    # Training/model settings used directly by main(); dataset settings are read
    # inside _prepare_training_data.
    output_keys = tuple(config.get("model.output_keys"))
    model_name = config.get("run_options.run_name")

    save_path = os.path.abspath(args.save_path) if args.save_path else None
    plot_path = os.path.abspath(args.plot_path) if args.plot_path else None

    os.makedirs(save_path, exist_ok=True) if save_path else None
    os.makedirs(plot_path, exist_ok=True) if plot_path else None

    get_device()

    rng_key = random.PRNGKey(config.get("training.seed"))
    model_dir = os.path.join(plot_path, model_name)

    if config.get("training.generation") == "inloop":
        state, train_loss, val_loss, val_loss_per_key = _run_inloop_training(
            config, rng_key, model_dir, save_path
        )
    else:
        (
            train_galaxy_images,
            train_psf_images,
            train_labels,
            norm_params,
            img_params,
            eff_val_split,
            resample_noise_sd,
        ) = _prepare_training_data(config)

        _save_run_artifacts(config, model_dir, norm_params, img_params)

        train_cfg = TrainConfig.from_config(config, save_path=save_path)
        # After D4 augmentation the train/val boundary shifts; use the effective
        # split so train_model's internal split matches (a no-op when not
        # augmenting).
        train_cfg.val_split = eff_val_split
        # Fresh-noise training: the per-step noise std (physical / image
        # gal_std) is computed alongside the normalizer, not from the config
        # (a no-op when off).
        train_cfg.resample_noise_sd = resample_noise_sd

        state, train_loss, val_loss, val_loss_per_key = train_cfg.run(
            train_galaxy_images,
            train_labels,
            rng_key,
            psf_images=train_psf_images,
        )

    logger.info("Plotting learning curve...")
    plot_learning_curve(val_loss, train_loss, os.path.join(plot_path, model_name, "learning_curve.png"))

    loss_path = os.path.join(plot_path, model_name, f"{model_name}_loss.npz") if plot_path else None
    _save_losses(loss_path, train_loss, val_loss, val_loss_per_key, output_keys)


if __name__ == "__main__":
    main()
