"""Training driven by in-loop generation.

:func:`train_model_inloop` is the counterpart of
:func:`~shearnet.core.train.train_model` for ``dataset.generation: inloop``.
It takes no image arrays -- stamps are rendered inside the jitted step from a
device-resident truth table -- but keeps the same optimiser, schedule, EMA,
early stopping, checkpointing and return signature, so everything downstream
(learning-curve plots, loss files, evaluation) is unchanged.

Written as a separate driver rather than a branch inside ``train_model``: the
two loops differ in what a "batch" *is* (an index vector versus a slice of a
materialised array), and threading that through a well-tested function would
have put every existing run at risk for no benefit. Everything that is not the
loop -- model construction, loss resolution, checkpointing, per-key MSE -- is
imported from :mod:`shearnet.core.train`, so there is one implementation of
each.

Differences from the up-front path, all deliberate:

* **The remainder of a split is left out.** Both the training and validation
  index lists are truncated to whole batches. A short batch is a different
  input shape and would cost a second XLA compile of the fused graph. Training
  reshuffles every epoch so nothing is systematically excluded; validation does
  not shuffle, so the same handful of stamps is always omitted -- deterministic,
  and at ``batch_size`` out of a 20% split it is a rounding error on the loss.
* **Fresh noise every training step**, drawn inside the step. Validation noise
  is keyed by batch position, so a given stamp sees the same realisation every
  epoch and the validation loss stays comparable across epochs.
* **Only ``g1``/``g2``/``hlr``/``flux`` labels.** ``psf_e1``/``psf_e2``/``psf_T``
  need an ngmix adaptive-moments fit on the rendered PSF, which cannot run
  inside a jitted step.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from ..logging_utils import get_logger
from .inloop import (
    RESPONSE_TERMS,
    InLoopGenerator,
    ResponseRegularization,
    format_response_report,
    make_batch_render,
    make_fused_eval_step,
    make_fused_train_step,
    make_response_report,
    normalize_state,
)
from .losses import resolve_loss
from .models import attention_pool_diagnostics, build_model, is_fork_model
from .train import save_checkpoint

logger = get_logger(__name__)

__all__ = ["train_model_inloop"]


def _make_forward(model, nn, output_keys, gap, use_dropout):
    """Close over everything static so the traced signature stays minimal."""
    is_fork = is_fork_model(nn)

    def forward(params, gal, psf, dropout_key):
        # output_keys sizes the output Dense, so it must be passed on apply as
        # well as init -- single-branch models included, or the head falls back
        # to its 2-key default and the parameter shapes disagree.
        kw = dict(output_keys=output_keys, gap=gap)
        if use_dropout and dropout_key is not None:
            kw.update(deterministic=False, rngs={"dropout": dropout_key})
        elif use_dropout:
            kw.update(deterministic=True)
        if is_fork:
            return model.apply(params, gal, psf, **kw)
        return model.apply(params, gal, **kw)

    return forward


def train_model_inloop(
    gen: InLoopGenerator,
    rng_key,
    output_keys: Sequence[str] = ("g1", "g2"),
    label_norm=None,
    img_norm=None,
    nse_sd: float = 0.0,
    epochs: int = 10,
    nn: str = "cnn",
    galaxy_type: str = "research_backed",
    psf_type: str = "forklens_psf",
    fusion: str = "concat",
    head: str = "gap",
    save_path: Optional[str] = None,
    model_name: str = "my_model",
    val_split: float = 0.2,
    eval_interval: int = 1,
    patience: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gap: bool = False,
    weights=None,
    loss: str = "mse",
    ema_decay: Optional[float] = None,
    dropout: float = 0.0,
    trace_psf_shear: bool = False,
    response: Optional[ResponseRegularization] = None,
    noise_range=None,
    noise_condition: bool = False,
    response_report: Optional[bool] = None,
    branch_features=None,
    d4_features=None,
    d4_depths_galaxy=None,
    d4_depths_psf=None,
    orbit_scan=True,
    fusion_pos="learned",
    design=None,
    d_model=None,
    num_heads=None,
    num_pool_heads=None,
    num_self_attn_layers=None,
    ffn_dim=None,
):
    """Train with stamps generated inside the jitted step.

    Args:
        gen: an :class:`~shearnet.core.inloop.InLoopGenerator` over the full
            sample; the train/validation split is taken on its index range.
        rng_key: PRNG key for init, shuffling and noise.
        label_norm: optional ``{'mean','std'}`` applied to the truth labels, so
            the network trains on the same scale as the up-front path.
        img_norm: optional image standardisation applied inside the step, after
            the render and the float32 cast.
        nse_sd: pixel noise std in *image* units (before ``img_norm``).
        trace_psf_shear: keep the PSF-shear transform in the graph so the
            ``R^PSF`` tangent is available. Costs a transform; off by default.
        branch_features: channel widths of the ``d4cnn`` backbone inside
            ``d4-fork-like`` (``None`` keeps the default).
        response: optional response/orbit loss configuration. Its component
            means are logged each epoch; validation remains supervised-only.
        noise_range: optional per-batch uniform noise range ``(min_sd,max_sd)``.
            Validation is evaluated at the midpoint of the range, so the
            validation loss is a fixed-depth number comparable across epochs.
        noise_condition: represent inputs in their sampled noise units.
        response_report: log the measured response matrices at every validation
            pass. Defaults to on whenever ``response`` is active -- validation
            MSE cannot see a drifting ``R^gamma`` or a growing PSF leakage, so
            training with response terms and only watching the loss is flying
            blind. It costs one extra jitted call per eval.

    Returns ``(state, train_losses, val_losses, val_losses_per_key)`` -- the
    same tuple :func:`~shearnet.core.train.train_model` returns.
    """
    output_keys = tuple(output_keys)
    n_out = len(output_keys)
    response = response or ResponseRegularization()
    if noise_condition and img_norm is not None:
        raise ValueError("noise conditioning cannot be combined with normalize_images")
    if response.enabled and not {"g1", "g2"}.issubset(output_keys):
        raise ValueError("response regularisation requires g1 and g2 output keys")
    shear_indices = (
        tuple(output_keys.index(key) for key in ("g1", "g2")) if response.enabled else (0, 1)
    )
    labels = gen.labels(output_keys)
    if label_norm is not None:
        labels = (labels - jnp.asarray(label_norm["mean"])) / jnp.asarray(label_norm["std"])
    # The network predicts normalised labels, so every response target has to be
    # divided by the same scale before it can be compared to a measured JVP.
    label_scale = (
        jnp.asarray(label_norm["std"])[jnp.asarray(shear_indices)]
        if label_norm is not None
        else None
    )

    # Index split. Contiguous, matching the up-front path, so the same seed puts
    # the same objects in validation on both backends.
    split_idx = int(gen.n * (1 - val_split))
    train_ids = jnp.arange(0, split_idx)
    val_ids = jnp.arange(split_idx, gen.n)
    steps_per_epoch = int(train_ids.shape[0]) // gen.batch_size
    val_steps = int(val_ids.shape[0]) // gen.batch_size
    if steps_per_epoch == 0 or val_steps == 0:
        raise ValueError(
            f"{gen.n} objects with batch_size={gen.batch_size} and "
            f"val_split={val_split} leaves {steps_per_epoch} train / "
            f"{val_steps} validation batches; need at least one of each."
        )
    logger.info(
        "in-loop training: %d objects -> %d train batches + %d val batches "
        "of %d (remainder left out to keep one compilation)",
        gen.n,
        steps_per_epoch,
        val_steps,
        gen.batch_size,
    )

    if weights is None:
        weights = jnp.ones(n_out)
    else:
        weights = jnp.asarray(weights, dtype=jnp.float32)
        if len(weights) != n_out:
            raise ValueError(f"loss_weights length {len(weights)} != output_keys length {n_out}")

    model = build_model(
        nn,
        galaxy_type=galaxy_type,
        psf_type=psf_type,
        fusion=fusion,
        head=head,
        dropout=dropout,
        branch_features=branch_features,
        d4_features=d4_features,
        d4_depths_galaxy=d4_depths_galaxy,
        d4_depths_psf=d4_depths_psf,
        orbit_scan=orbit_scan,
        fusion_pos=fusion_pos,
        design=design,
        d_model=d_model,
        num_heads=num_heads,
        num_pool_heads=num_pool_heads,
        num_self_attn_layers=num_self_attn_layers,
        ffn_dim=ffn_dim,
    )
    use_dropout = bool(dropout) and dropout > 0.0
    if use_dropout:
        rng_key, dropout_init_key = jax.random.split(rng_key)
        init_rngs = {"params": rng_key, "dropout": dropout_init_key}
    else:
        init_rngs = rng_key

    npix = gen.cfg.npix
    dummy = jnp.ones((npix, npix), dtype=jnp.float32)
    if is_fork_model(nn):
        params = model.init(init_rngs, dummy, dummy, output_keys=output_keys, gap=gap)
    else:
        params = model.init(init_rngs, dummy, output_keys=output_keys, gap=gap)

    total_steps = max(epochs * steps_per_epoch, 1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=int(0.05 * total_steps), decay_steps=total_steps
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    # Without this the second step presents `step` as an array rather than a
    # Python int and XLA compiles the whole fused graph a second time.
    state = normalize_state(state)

    use_ema = ema_decay is not None
    ema_params = state.params if use_ema else None
    if use_ema:
        logger.info(
            "Weight EMA enabled (decay=%s); validation and checkpoint " "use the averaged weights.",
            ema_decay,
        )

        @jax.jit
        def _ema_update(ema, live):
            return jax.tree_util.tree_map(
                lambda e, p: ema_decay * e + (1.0 - ema_decay) * p, ema, live
            )

    loss_callable = resolve_loss(loss)
    forward = _make_forward(model, nn, output_keys, gap, use_dropout)

    def objective(preds, lab):
        return loss_callable(preds, lab, weights)

    train_step = make_fused_train_step(
        gen,
        forward,
        objective,
        labels,
        nse_sd,
        trace_psf_shear=trace_psf_shear,
        img_norm=img_norm,
        response=response,
        shear_indices=shear_indices,
        label_scale=label_scale,
        return_metrics=response.enabled,
        noise_range=noise_range,
        noise_condition=noise_condition,
    )
    # Validation runs at a single fixed depth even when training randomises it:
    # a validation loss drawn at a random depth every epoch is not comparable
    # across epochs, and early stopping would react to the draw.
    eval_nse_sd = nse_sd if noise_range is None else 0.5 * (noise_range[0] + noise_range[1])
    eval_step = make_fused_eval_step(
        gen,
        forward,
        objective,
        labels,
        eval_nse_sd,
        trace_psf_shear=trace_psf_shear,
        img_norm=img_norm,
        per_key=True,
        noise_condition=noise_condition,
    )

    # The attention head already computes four spatial probability maps.  Once
    # per validation pass, capture those maps through Flax's intermediates
    # collection and report statistics that reveal diffuse maps, duplicate
    # heads, and full head collapse.  This does not alter ordinary forward
    # returns, checkpoints, or the training objective.
    attention_report = None
    if nn == "d4-fork-like" and head == "attention":
        attention_render = make_batch_render(
            gen,
            eval_nse_sd,
            trace_psf_shear=trace_psf_shear,
            img_norm=img_norm,
            noise_condition=noise_condition,
        )

        @jax.jit
        def attention_report(params, idx, noise_key):
            gal, psf = attention_render(idx, noise_key)
            _, captured = model.apply(
                params,
                gal,
                psf,
                output_keys=output_keys,
                gap=gap,
                deterministic=True,
                capture_attention=True,
                mutable=["intermediates"],
            )
            maps = captured["intermediates"]["pool_attention"][0]
            return attention_pool_diagnostics(maps)

    if noise_range is not None:
        logger.info(
            "in-loop noise randomization: uniform [%.4g, %.4g]%s; validation at %.4g",
            noise_range[0],
            noise_range[1],
            " with noise-unit conditioning" if noise_condition else "",
            eval_nse_sd,
        )

    if response_report is None:
        response_report = response.enabled
    report_step = None
    if response_report:
        report_step = make_response_report(
            gen,
            forward,
            eval_nse_sd,
            img_norm=img_norm,
            shear_indices=shear_indices,
            label_scale=label_scale,
            noise_condition=noise_condition,
        )

    rng_key, val_noise_key = jax.random.split(rng_key)
    val_batches = gen.batches(val_ids)  # fixed order -> frozen noise

    train_losses, val_losses, val_losses_per_key = [], [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        logger.info("Epoch %d/%d", epoch + 1, epochs)

        rng_key, shuffle_key = jax.random.split(rng_key)
        idx_mat = gen.batches(train_ids, key=shuffle_key)

        train_loss = 0.0
        # [supervised, *RESPONSE_TERMS, n_active]
        response_sum = jnp.zeros(len(RESPONSE_TERMS) + 2)
        for s in range(steps_per_epoch):
            rng_key, noise_key, dropout_key = jax.random.split(rng_key, 3)
            result = train_step(state, idx_mat[s], noise_key, dropout_key)
            if response.enabled:
                state, batch_loss, response_metrics = result
                response_sum = response_sum + response_metrics
            else:
                state, batch_loss = result
            if use_ema:
                ema_params = _ema_update(ema_params, state.params)
            train_loss += float(batch_loss)
        train_loss /= steps_per_epoch
        train_losses.append(train_loss)
        if response.enabled:
            # The response terms are only evaluated on active steps, so divide
            # them by that count -- averaging over every step would report a
            # value every_n_steps times too small and look like convergence.
            n_active = max(float(response_sum[-1]), 1.0)
            logger.info(
                "  response terms: supervised=%.4e %s (%d/%d active steps)",
                float(response_sum[0]) / steps_per_epoch,
                " ".join(
                    f"{name}={float(response_sum[i + 1]) / n_active:.4e}"
                    for i, name in enumerate(RESPONSE_TERMS)
                ),
                int(n_active),
                steps_per_epoch,
            )

        if (epoch + 1) % eval_interval:
            continue

        eval_state = state.replace(params=ema_params) if use_ema else state
        val_loss = 0.0
        per_key_sum = jnp.zeros(n_out)
        for s in range(val_steps):
            # Keyed by batch position, not epoch: a given stamp sees the same
            # noise every epoch, so the validation loss is comparable and
            # patience reacts to the model rather than to the draw.
            l, pk = eval_step(eval_state, val_batches[s], jax.random.fold_in(val_noise_key, s))
            val_loss += float(l)
            per_key_sum = per_key_sum + pk
        val_loss /= val_steps
        val_losses.append(val_loss)
        val_per_key = per_key_sum / val_steps
        val_losses_per_key.append(val_per_key)
        logger.info("Validation Loss: %.4e", val_loss)
        logger.info(
            "  Per-key validation MSE: %s",
            ", ".join(f"{k}={float(v):.4e}" for k, v in zip(output_keys, val_per_key)),
        )
        if report_step is not None:
            values, _ = report_step(
                eval_state.params, val_batches[0], jax.random.fold_in(val_noise_key, 0)
            )
            logger.info("  %s", format_response_report(values))
        if attention_report is not None:
            diagnostics = attention_report(
                eval_state.params, val_batches[0], jax.random.fold_in(val_noise_key, 0)
            )
            entropy = diagnostics["entropy"]
            logger.info(
                "  attention pooling: entropy=[%s] similarity(mean/max)=%.4f/%.4f "
                "effective_heads=%.3f/%d",
                ", ".join(f"{float(value):.4f}" for value in entropy),
                float(diagnostics["mean_similarity"]),
                float(diagnostics["max_similarity"]),
                float(diagnostics["effective_rank"]),
                len(entropy),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info("New best validation loss: %.4e", val_loss)
            if save_path:
                save_checkpoint(
                    eval_state,
                    step=epoch + 1,
                    checkpoint_dir=save_path,
                    model_name=model_name,
                    overwrite=True,
                )
        else:
            patience_counter += 1
            logger.info(
                "No improvement in validation loss. Patience: %d/%d", patience_counter, patience
            )
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    final_state = state.replace(params=ema_params) if use_ema else state
    return final_state, train_losses, val_losses, val_losses_per_key
