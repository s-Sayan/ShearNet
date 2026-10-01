"""Train a model into a run directory.

:func:`train` is what ``shearnet-train`` runs: it renders the training
population (or, in-loop, the truth table), fits the normalizers on the training
portion only, trains, and leaves a self-contained run directory behind -- the
best model, the normalizers, the resolved config, the history and the learning
curve -- whether or not anyone asked for plots. Nothing in here is a scientific
measurement; ``shearnet-eval`` makes those from the finished run.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

import numpy as np

from ..artifacts import provenance
from ..artifacts.checkpoints import save_params
from ..artifacts.runs import RunDir, atomic_write
from ..config import Config
from ..core.augment import d4_augment
from ..core.dataset import split_combined_images
from ..core.models import is_fork_model
from ..core.specs import DatasetSpec, TrainConfig
from ..logging_utils import get_logger, run_log
from ..utils.normalization import (
    fit_image_normalizer,
    fit_normalizer,
    identity_normalizer,
    save_image_normalizer,
    save_normalizer,
    transform_images,
    transform_labels,
)
from .curves import plot_learning_curve
from .history import History

logger = get_logger(__name__)

__all__ = ["train"]


class _Checkpointer:
    """Saves the best parameters as they improve, and remembers which epoch."""

    def __init__(self, run: RunDir):
        self.run = run
        self.epoch: Optional[int] = None
        self.sha256: Optional[str] = None

    def __call__(self, variables, epoch: int) -> None:
        self.sha256 = save_params(variables, self.run.best_params)
        self.epoch = int(epoch)
        logger.info("saved model at epoch %d -> %s", epoch, self.run.best_params)


def train(config: Config, run: RunDir, *, input_text: Optional[str] = None,
          overwrite: bool = False) -> RunDir:
    """Train the model ``config`` describes into ``run``. Returns ``run``."""
    run.create(overwrite=overwrite)
    if input_text is not None:
        atomic_write(run.config_input, input_text.encode())
    config.save(run.config_resolved)
    manifest = {
        "schema_version": 1,
        "kind": "shearnet-training-run",
        "run_name": config.get("run_options.run_name"),
        "description": config.get("run_options.description"),
        "config_sha256": hashlib.sha256(config.to_yaml().encode()).hexdigest(),
        "output_keys": list(config.get("model.output_keys")),
        "model": {"type": config.get("model.type"), "file": "model/best.msgpack",
                  "parameters": "ema" if config.get("training.ema_decay") else "raw"},
        "provenance": provenance.collect(),
    }
    run.write_manifest(manifest)
    run.write_status("running")

    with run_log(run.train_log):
        logger.info("training run %s -> %s", manifest["run_name"], run.root)
        for note in config.notes:
            logger.warning("config: %s", note)
        start = time.time()
        try:
            history, checkpoint, n_parameters = _train(config, run)
            if checkpoint.epoch is None:
                raise RuntimeError("no epoch produced a finite validation loss; nothing "
                                   "was saved")
            plot_learning_curve(history, run.learning_curve)
        except BaseException as exc:
            run.write_status("failed", error=f"{type(exc).__name__}: {exc}")
            logger.error("training failed: %s", exc)
            raise
        manifest.update(
            checkpoint={"file": "model/best.msgpack", "epoch": checkpoint.epoch,
                        "sha256": checkpoint.sha256},
            best_val_loss=history.best_val_loss,
            epochs_run=len(history.records),
            n_parameters=n_parameters,
            normalizers={
                "labels": "normalizers/labels.npz",
                "images": ("normalizers/images.npz" if run.image_normalizer.is_file()
                           else None),
            },
            seconds=round(time.time() - start, 1),
        )
        run.write_manifest(manifest)
        run.write_status("completed")
        logger.info("")
        logger.info("TRAINING COMPLETE")
        logger.info("Run:            %s", run.root)
        logger.info("Model:          %s (epoch %d)", run.best_params, checkpoint.epoch)
        logger.info("History:        %s", run.history_csv)
        logger.info("Learning curve: %s", run.learning_curve)
    return run


def _train(config: Config, run: RunDir):
    import jax
    import jax.random as random

    from ..utils.device import get_device

    get_device()
    output_keys = tuple(config.get("model.output_keys"))
    history = History(output_keys, run.history_csv, run.history_npz)
    checkpoint = _Checkpointer(run)
    rng_key = random.PRNGKey(config.get("training.seed"))

    if config.get("training.generation") == "inloop":
        state = _train_inloop(config, run, rng_key, history, checkpoint)
    else:
        state = _train_upfront(config, run, rng_key, history, checkpoint)
    n_parameters = int(sum(x.size for x in jax.tree_util.tree_leaves(state.params)))
    return history, checkpoint, n_parameters


def _save_normalizers(run: RunDir, norm_params, img_params, output_keys) -> None:
    save_normalizer(norm_params, str(run.label_normalizer), output_keys=output_keys)
    if img_params is not None:
        save_image_normalizer(img_params, str(run.image_normalizer))


# ----------------------------------------------------------------------
# up-front: render the dataset, then train on the arrays
# ----------------------------------------------------------------------
def prepare_training_data(config: Config):
    """Simulate the training set and standardize its labels (and, optionally, images).

    Everything is fit on the TRAINING portion only: the split is taken first,
    on the raw stamps, so neither the D4 augmentation (an ablation for
    non-equivariant models) nor either normalizer ever sees a validation stamp.

    Returns ``(galaxy_images, psf_images, labels, norm_params, img_params,
    eff_val_split, resample_noise_sd)``. ``psf_images`` is ``None`` for
    single-branch models; ``labels`` are already normalized; ``eff_val_split``
    is the validation fraction that lands ``train_model``'s own split on the
    same boundary after augmentation; ``resample_noise_sd`` is the fresh-noise
    std in model-input units (``0.0`` unless ``training.resample_noise``).
    """
    spec = DatasetSpec.from_config(config)
    val_split = config.get("training.val_split")
    output_keys = tuple(config.get("model.output_keys"))

    galaxy_images, labels = spec.build()
    psf_images = None
    if spec.return_psf:
        galaxy_images, psf_images = split_combined_images(
            galaxy_images, has_psf=True, has_clean=False
        )
        logger.info(f"Shape of train PSF images: {psf_images.shape}")
    logger.info(f"Shape of train images: {galaxy_images.shape}")
    logger.info(f"Shape of train labels: {labels.shape}")

    split_idx = int(len(labels) * (1 - val_split))
    eff_val_split = val_split

    if config.get("training.d4_augment"):
        # augment the train portion 8x, keep val untouched, and reassemble as
        # [aug_train, val]; eff_val_split puts train_model's split on this boundary
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

    if config.get("training.normalize_labels"):
        norm_params = fit_normalizer(labels[:split_idx])
    else:
        norm_params = identity_normalizer(labels)
    labels = transform_labels(labels, norm_params)

    resample_noise = config.get("training.resample_noise")
    nse_sd = spec.nse_sd
    img_params = None
    if config.get("training.normalize_images"):
        img_params = fit_image_normalizer(
            galaxy_images[:split_idx],
            psf_images[:split_idx] if psf_images is not None else None,
        )
        if resample_noise:
            # the stamps are noise-free here; inflate to the noisy-equivalent
            # sqrt(var_clean + nse_sd^2) so the saved scale matches a baked-noise run
            img_params["gal_std"] = float(np.sqrt(img_params["gal_std"] ** 2 + nse_sd**2))
        galaxy_images = transform_images(galaxy_images, img_params, channel="gal")
        if psf_images is not None:
            psf_images = transform_images(psf_images, img_params, channel="psf")

    gal_std = img_params["gal_std"] if img_params is not None else 1.0
    resample_noise_sd = (nse_sd / gal_std) if resample_noise else 0.0
    return (galaxy_images, psf_images, labels, norm_params, img_params, eff_val_split,
            resample_noise_sd)


def _train_upfront(config, run, rng_key, history, checkpoint):
    (galaxy_images, psf_images, labels, norm_params, img_params, eff_val_split,
     resample_noise_sd) = prepare_training_data(config)
    _save_normalizers(run, norm_params, img_params, config.get("model.output_keys"))

    train_cfg = TrainConfig.from_config(config)
    train_cfg.val_split = eff_val_split
    train_cfg.resample_noise_sd = resample_noise_sd
    state, *_ = train_cfg.run(galaxy_images, labels, rng_key, psf_images=psf_images,
                              checkpoint_fn=checkpoint, history_fn=history)
    return state


# ----------------------------------------------------------------------
# in-loop: render inside the jitted step
# ----------------------------------------------------------------------
def _train_inloop(config, run, rng_key, history, checkpoint):
    """``training.generation: inloop`` -- no dataset is materialised.

    Mirrors the up-front path where it must: the label normalizer is fit on the
    same contiguous training portion, and image standardisation is fit on
    rendered stamps and applied inside the step.
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
    noise_range, noise_condition = noise_schedule_from_config(config.get("training.noise"))
    if not jax.config.jax_enable_x64:
        logger.warning(
            "JAX_ENABLE_X64 is not set: in-loop rendering will run in float32 "
            "(render accuracy ~1e-7). Export JAX_ENABLE_X64=1 before starting for "
            "float64; the stamps are cast to float32 before the network, so "
            "training memory and speed are unchanged either way."
        )
    logger.info("in-loop generation: render dtype %s", render_dtype().__name__)

    gen = spec.build_inloop_generator(batch_size)

    val_split = config.get("training.val_split")
    split_idx = int(gen.n * (1 - val_split))
    raw_labels = np.asarray(gen.labels(output_keys))
    if config.get("training.normalize_labels"):
        norm_params = fit_normalizer(raw_labels[:split_idx])
    else:
        norm_params = identity_normalizer(raw_labels)

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

    _save_normalizers(run, norm_params, img_params, output_keys)

    tc = TrainConfig.from_config(config)
    state, *_ = train_model_inloop(
        gen,
        rng_key,
        output_keys=output_keys,
        label_norm=norm_params,
        img_norm=img_params,
        nse_sd=spec.nse_sd,
        epochs=tc.epochs,
        nn=tc.nn,
        galaxy_type=tc.galaxy_type,
        psf_type=tc.psf_type,
        fusion=tc.fusion,
        head=tc.head,
        val_split=val_split,
        eval_interval=tc.eval_interval,
        patience=tc.patience,
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        gap=tc.gap,
        weights=tc.weights,
        loss=tc.loss,
        ema_decay=tc.ema_decay,
        dropout=tc.dropout,
        branch_features=tc.branch_features,
        d4_features=tc.d4_features,
        d4_depths_galaxy=tc.d4_depths_galaxy,
        d4_depths_psf=tc.d4_depths_psf,
        d4_multiscale=tc.d4_multiscale,
        orbit_scan=tc.orbit_scan,
        fusion_pos=tc.fusion_pos,
        design=tc.design,
        d_model=tc.d_model,
        num_heads=tc.num_heads,
        num_pool_heads=tc.num_pool_heads,
        num_self_attn_layers=tc.num_self_attn_layers,
        ffn_dim=tc.ffn_dim,
        response=response,
        noise_range=noise_range,
        noise_condition=noise_condition,
        response_report=config.get("training.response.report"),
        # a per-object PSF shear in the truth table is only rendered if the
        # transform is traced; the response terms switch it on for themselves
        trace_psf_shear=config.get("simulation.apply_psf_shear"),
        checkpoint_fn=checkpoint,
        history_fn=history,
    )
    return state


def _silence_absl() -> None:
    logging.getLogger("absl").setLevel(logging.ERROR)


_silence_absl()
