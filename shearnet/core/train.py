"""Core training functions for ShearNet models."""

from tqdm import tqdm
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
    preds = state.apply_fn(params, images, gap=gap)
    sq_err = (preds - labels) ** 2
    loss = (sq_err * weights[None, :]).mean()
    return loss

def fork_loss_fn(state, params, galaxy_images, psf_images, labels, output_keys, gap, weights):
    preds = state.apply_fn(params, galaxy_images, psf_images, output_keys, gap=gap)
    sq_err = (preds - labels) ** 2
    loss = (sq_err * weights[None, :]).mean()
    return loss

@functools.partial(jax.jit, static_argnums=(3,))
def train_step(state, images, labels, gap, weights):
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, images, labels, gap, weights)
    state = state.apply_gradients(grads=grads)
    return state, loss

@functools.partial(jax.jit, static_argnums=(4,5))
def fork_train_step(state, galaxy_images, psf_images, labels, output_keys, gap, weights):
    grad_fn = jax.value_and_grad(fork_loss_fn, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, galaxy_images, psf_images, labels, output_keys, gap, weights)
    state = state.apply_gradients(grads=grads)
    return state, loss

@functools.partial(jax.jit, static_argnums=(3,))
def eval_step(state, images, labels, gap, weights):
    loss = loss_fn(state, state.params, images, labels, gap, weights)
    return loss

@functools.partial(jax.jit, static_argnums=(4,5))
def fork_eval_step(state, galaxy_images, psf_images, labels, output_keys, gap, weights):
    loss = fork_loss_fn(state, state.params, galaxy_images, psf_images, labels, output_keys, gap, weights)
    return loss


def train_modelv1(images, labels, rng_key, epochs=10, batch_size=32, nn="simple", save_path=None, model_name="my_model"):
    """Original training function without validation."""
    if nn == "simple":
        model = SimpleGalaxyNN()
    elif nn == "enhanced":
        model = EnhancedGalaxyNN()
    elif nn == "resnet":
        model = GalaxyResNet()
    else:
        raise ValueError("Invalid model type specified.")
    
    params = model.init(rng_key, jnp.ones_like(images[0]))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    
    epoch_losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        rng_key, subkey = jax.random.split(rng_key)
        perm = jax.random.permutation(subkey, len(images))
        shuffled_images = images[perm]
        shuffled_labels = labels[perm]
        epoch_loss = 0
        count = 0

        with tqdm(total=len(images) // batch_size) as pbar:
            for i in range(0, len(images), batch_size):
                batch_images = shuffled_images[i:i + batch_size]
                batch_labels = shuffled_labels[i:i + batch_size]
                state, loss = train_step(state, batch_images, batch_labels)  
                epoch_loss += loss
                count += 1
                pbar.update(1)
        print(f"Loss: {loss.item()}")
        epoch_loss /= count
        epoch_losses.append(epoch_loss)

        if save_path:
            save_checkpoint(state, step=epoch + 1, checkpoint_dir=save_path, model_name=model_name, overwrite=True)
    
    return state, epoch_losses


def train_model(galaxy_images, psf_images, labels, rng_key, epochs=10, 
                  batch_size=32, nn="simple", galaxy_type='cnn', 
                  psf_type='cnn', save_path=None, model_name="my_model",
                  val_split=0.2, eval_interval=1, patience=5, lr=1e-3,
                  weight_decay=1e-4, output_keys=("g1", "g2"), gap=False,
                  weights=None):
    """Enhanced training function with validation and early stopping.
    
    Saves only the best checkpoint (by val loss) using model_name as the prefix.
    No final checkpoint is saved — the checkpoint on disk is always the best epoch.
    """
    # Split into train and validation sets
    split_idx = int(len(galaxy_images) * (1 - val_split))
    train_galaxy_images, val_galaxy_images = galaxy_images[:split_idx], galaxy_images[split_idx:]
    train_psf_images, val_psf_images = psf_images[:split_idx], psf_images[split_idx:]
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
        model = ForkLike(galaxy_model_type=galaxy_type, psf_model_type=psf_type)
    else:
        raise ValueError(f"Invalid model type specified: {nn}")
    
    params = model.init(rng_key, jnp.ones_like(galaxy_images[0]), jnp.ones_like(psf_images[0]), output_keys=output_keys, gap=gap)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr, 
        decay_steps=epochs * (len(train_galaxy_images) // batch_size)
    )
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_losses, val_losses = [], []
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
                for i in range(0, len(val_galaxy_images), batch_size):
                    batch_galaxy_images = val_galaxy_images[i:i + batch_size]
                    batch_psf_images = val_psf_images[i:i + batch_size]
                    batch_labels = val_labels[i:i + batch_size]
                    batch_size_actual = len(batch_galaxy_images)
                    loss = fork_eval_step(state, batch_galaxy_images, batch_psf_images, batch_labels, output_keys, gap=gap, weights=weights)
                    val_loss += loss * batch_size_actual
                    total_samples += batch_size_actual
                val_loss /= total_samples
                val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4e}")

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

    return state, train_losses, val_losses