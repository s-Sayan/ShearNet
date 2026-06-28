"""Core training functions for ShearNet models."""

import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from .models import SimpleGalaxyNN, EnhancedGalaxyNN, GalaxyResNet, ResearchBackedGalaxyResNet, ForkLensPSFNet, ForkLike
import functools


def save_checkpoint(state, step, checkpoint_dir, model_name, overwrite=True):
    """Save the model checkpoint."""
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step,
        prefix=model_name,
        overwrite=overwrite
    )
    print(f"Checkpoint saved at step {step}.")


def loss_fn(state, params, images, labels, gap, weights):
    """Weighted mean-squared-error loss for single-branch models."""
    preds = state.apply_fn(params, images, gap=gap)
    sq_err = (preds - labels) ** 2
    loss = (sq_err * weights[None, :]).mean()
    return loss

def fork_loss_fn(state, params, galaxy_images, psf_images, labels, output_keys, gap, weights):
    """Weighted MSE loss for the two-branch ``fork-like`` model."""
    preds = state.apply_fn(params, galaxy_images, psf_images, output_keys, gap=gap)
    sq_err = (preds - labels) ** 2
    loss = (sq_err * weights[None, :]).mean()
    return loss
    
def loss_fn_per_key(state, params, images, labels, gap, weights):
    """Like :func:`loss_fn` but also returns the per-output-key MSE."""
    preds = state.apply_fn(params, images, gap=gap)
    sq_err = (preds - labels) ** 2
    loss = (sq_err * weights[None, :]).mean()
    per_key = sq_err.mean(axis=0)
    return loss, per_key

def fork_loss_fn_per_key(state, params, galaxy_images, psf_images, labels, output_keys, gap, weights):
    """Like :func:`fork_loss_fn` but also returns the per-output-key MSE."""
    preds = state.apply_fn(params, galaxy_images, psf_images, output_keys, gap=gap)
    sq_err = (preds - labels) ** 2
    loss = (sq_err * weights[None, :]).mean()
    per_key = sq_err.mean(axis=0)
    return loss, per_key

@functools.partial(jax.jit, static_argnums=(3,))
def train_step(state, images, labels, gap, weights):
    """One JIT-compiled gradient-descent step for single-branch models."""
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, images, labels, gap, weights)
    state = state.apply_gradients(grads=grads)
    return state, loss

@functools.partial(jax.jit, static_argnums=(4,5))
def fork_train_step(state, galaxy_images, psf_images, labels, output_keys, gap, weights):
    """One JIT-compiled gradient-descent step for the ``fork-like`` model."""
    grad_fn = jax.value_and_grad(fork_loss_fn, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, galaxy_images, psf_images, labels, output_keys, gap, weights)
    state = state.apply_gradients(grads=grads)
    return state, loss

@functools.partial(jax.jit, static_argnums=(3,))
def eval_step(state, images, labels, gap, weights):
    """JIT-compiled validation loss for single-branch models (no gradient step)."""
    loss = loss_fn(state, state.params, images, labels, gap, weights)
    return loss

@functools.partial(jax.jit, static_argnums=(4,5))
def fork_eval_step(state, galaxy_images, psf_images, labels, output_keys, gap, weights):
    """JIT-compiled validation loss for the ``fork-like`` model."""
    loss = fork_loss_fn(state, state.params, galaxy_images, psf_images, labels, output_keys, gap, weights)
    return loss

@functools.partial(jax.jit, static_argnums=(3,))
def eval_step_per_key(state, images, labels, gap, weights):
    """Validation loss plus per-output-key MSE for single-branch models."""
    return loss_fn_per_key(state, state.params, images, labels, gap, weights)

@functools.partial(jax.jit, static_argnums=(4,5))
def fork_eval_step_per_key(state, galaxy_images, psf_images, labels, output_keys, gap, weights):
    """Validation loss plus per-output-key MSE for the ``fork-like`` model."""
    return fork_loss_fn_per_key(state, state.params, galaxy_images, psf_images, labels, output_keys, gap, weights)


def train_model(galaxy_images, labels, rng_key, psf_images=None, epochs=10,
                  batch_size=32, nn="simple", galaxy_type='cnn',
                  psf_type='cnn', save_path=None, model_name="my_model",
                  val_split=0.2, eval_interval=1, patience=5, lr=1e-3,
                  weight_decay=1e-4, output_keys=("g1", "g2"), gap=False,
                  weights=None, fusion="concat"):
    """Train a ShearNet model with validation and early stopping.

    Builds the requested architecture, trains it with an AdamW optimizer and a
    warmup + cosine-decay learning-rate schedule, and (if ``save_path`` is given)
    saves only the best checkpoint by validation loss — the checkpoint on disk is
    always the best epoch, never the final one.

    Args:
        galaxy_images: Galaxy stamps, shape ``(N, npix, npix)``.
        labels: Targets, shape ``(N, len(output_keys))``.
        rng_key: A ``jax.random.PRNGKey`` for parameter init and shuffling.
        psf_images: PSF stamps, shape ``(N, npix, npix)``. Required only for the
            ``fork-like`` architecture; ignored (and optional) otherwise.
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        nn: Architecture name — one of ``'mlp'``, ``'cnn'``, ``'resnet'``,
            ``'research_backed'``, ``'forklens_psfnet'``, or ``'fork-like'``.
        galaxy_type, psf_type: Sub-model types for the two ``fork-like`` branches.
        save_path: Directory to write the best checkpoint to (no save if ``None``).
        model_name: Checkpoint filename prefix.
        val_split: Fraction of the data held out for validation.
        eval_interval: Validate every this many epochs.
        patience: Stop after this many evals without validation improvement.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        output_keys: Names of the predicted parameters.
        gap: Use global-average-pooling in the model head where supported.
        weights: Optional per-key loss weights (defaults to all ones).
        fusion: ``fork-like`` fusion strategy, ``'concat'`` or ``'transformer'``.

    Returns:
        ``(state, train_losses, val_losses, val_losses_per_key)`` where ``state``
        is the final ``TrainState`` and the remaining items are per-epoch loss
        histories.
    """
    # The two-branch 'fork-like' model needs PSF stamps; single-branch models
    # ignore them, so psf_images is optional for everything else.
    if nn == 'fork-like' and psf_images is None:
        raise ValueError("nn='fork-like' requires psf_images, but none were given.")

    # Split into train and validation sets
    split_idx = int(len(galaxy_images) * (1 - val_split))
    train_galaxy_images, val_galaxy_images = galaxy_images[:split_idx], galaxy_images[split_idx:]
    if psf_images is not None:
        train_psf_images, val_psf_images = psf_images[:split_idx], psf_images[split_idx:]
    else:
        train_psf_images = val_psf_images = None
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    n_out = len(output_keys)
    if weights is None:
        weights = jnp.ones(n_out)
    else:
        weights = jnp.array(weights, dtype=jnp.float32)
        assert len(weights) == n_out, f"loss_weights length {len(weights)} != output_keys length {n_out}"

    if nn == "mlp":
        model = SimpleGalaxyNN()
    elif nn == "cnn":
        model = EnhancedGalaxyNN()
    elif nn == "resnet":
        model = GalaxyResNet()
    elif nn == "research_backed":
        model = ResearchBackedGalaxyResNet()
    elif nn == "forklens_psfnet":
        model = ForkLensPSFNet()
    elif nn == "fork-like":
        model = ForkLike(galaxy_model_type=galaxy_type, psf_model_type=psf_type, fusion=fusion)
    else:
        raise ValueError(f"Invalid model type specified: {nn}")
    
    if nn == 'fork-like':
        params = model.init(rng_key, jnp.ones_like(galaxy_images[0]),
                            jnp.ones_like(psf_images[0]), output_keys=output_keys, gap=gap)
    else:
        params = model.init(rng_key, jnp.ones_like(galaxy_images[0]),
                            output_keys=output_keys, gap=gap)
    
    total_steps = epochs * (len(train_galaxy_images) // batch_size)
    warmup_steps = int(0.05 * total_steps)
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_losses, val_losses, val_losses_per_key = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0

    if nn == 'fork-like':
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Shuffle training data
            rng_key, subkey = jax.random.split(rng_key)
            perm = jax.random.permutation(subkey, len(train_galaxy_images))
            shuffled_train_galaxy_images = train_galaxy_images[perm]
            shuffled_train_psf_images = train_psf_images[perm]
            shuffled_train_labels = train_labels[perm]

            # Training phase
            train_loss, total_samples = 0, 0
            for i in range(0, len(train_galaxy_images), batch_size):
                batch_galaxy_images = shuffled_train_galaxy_images[i:i + batch_size]
                batch_psf_images = shuffled_train_psf_images[i:i + batch_size]
                batch_labels = shuffled_train_labels[i:i + batch_size]
                batch_size_actual = len(batch_galaxy_images)
                state, loss = fork_train_step(state, batch_galaxy_images, batch_psf_images, batch_labels, output_keys, gap=gap, weights=weights)
                train_loss += loss * batch_size_actual
                total_samples += batch_size_actual
            train_loss /= total_samples
            train_losses.append(train_loss)

            # Validation phase
            if (epoch + 1) % eval_interval == 0:
                val_loss, total_samples = 0, 0
                val_per_key_sum = jnp.zeros(n_out)
                for i in range(0, len(val_galaxy_images), batch_size):
                    batch_galaxy_images = val_galaxy_images[i:i + batch_size]
                    batch_psf_images = val_psf_images[i:i + batch_size]
                    batch_labels = val_labels[i:i + batch_size]
                    batch_size_actual = len(batch_galaxy_images)
                    loss, per_key = fork_eval_step_per_key(state, batch_galaxy_images, batch_psf_images, batch_labels, output_keys, gap=gap, weights=weights)
                    val_loss += loss * batch_size_actual
                    val_per_key_sum += per_key * batch_size_actual
                    total_samples += batch_size_actual
                val_loss /= total_samples
                val_per_key = val_per_key_sum / total_samples
                val_losses.append(val_loss)
                val_losses_per_key.append(val_per_key)
                print(f"Validation Loss: {val_loss:.4e}")
                per_key_str = ", ".join(f"{k}={float(v):.4e}" for k, v in zip(output_keys, val_per_key))
                print(f"  Per-key validation MSE: {per_key_str}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"New best validation loss: {val_loss:.4e}")
                    if save_path:
                        save_checkpoint(state, step=epoch + 1, checkpoint_dir=save_path,
                                      model_name=model_name, overwrite=True)
                else:
                    patience_counter += 1
                    print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    else:
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            # Shuffle training data
            rng_key, subkey = jax.random.split(rng_key)
            perm = jax.random.permutation(subkey, len(train_galaxy_images))
            shuffled_train_galaxy_images = train_galaxy_images[perm]
            shuffled_train_labels = train_labels[perm]

            train_loss, total_samples = 0, 0
            for i in range(0, len(train_galaxy_images), batch_size):
                batch_images = shuffled_train_galaxy_images[i:i + batch_size]
                batch_labels = shuffled_train_labels[i:i + batch_size]
                batch_size_actual = len(batch_images)
                state, loss = train_step(state, batch_images, batch_labels, gap, weights=weights)
                train_loss += loss * batch_size_actual
                total_samples += batch_size_actual
            train_loss /= total_samples
            train_losses.append(train_loss)

            if (epoch + 1) % eval_interval == 0:
                val_loss, total_samples = 0, 0
                for i in range(0, len(val_galaxy_images), batch_size):
                    batch_images = val_galaxy_images[i:i + batch_size]
                    batch_labels = val_labels[i:i + batch_size]
                    batch_size_actual = len(batch_images)
                    loss = eval_step(state, batch_images, batch_labels, gap, weights=weights)
                    val_loss += loss * batch_size_actual
                    total_samples += batch_size_actual
                val_loss /= total_samples
                val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        save_checkpoint(state, step=epoch + 1, checkpoint_dir=save_path, 
                                      model_name=model_name, overwrite=True)
                    print(f"New best validation loss: {val_loss:.4e}")
                else:
                    patience_counter += 1
                    print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    return state, train_losses, val_losses, val_losses_per_key
