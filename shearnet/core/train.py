"""Core training functions for ShearNet models."""

from tqdm import tqdm
import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from .models import ForkLike


class ReduceLROnPlateau:
    """JAX/Flax implementation of PyTorch's ReduceLROnPlateau scheduler."""
    
    def __init__(self, mode='min', factor=0.1, patience=10, threshold=1e-4, 
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False):
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        
        # Internal state
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.cooldown_counter = 0
        self.last_epoch = 0
        
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        
    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:
            self.mode_worse = -float('inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def _is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon
        else:
            return a > best + self.threshold

    def step(self, metrics, current_lr):
        current = float(metrics)
        self.last_epoch += 1

        if self.best is None:
            self.best = current
        elif self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1

        if self.num_bad_epochs > self.patience:
            new_lr = max(current_lr * self.factor, self.min_lr)
            if current_lr - new_lr > self.eps:
                if self.verbose:
                    print(f'Reducing learning rate from {current_lr:.6e} to {new_lr:.6e}')
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                return new_lr
        
        return current_lr

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0


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


def loss_fn(state, params, galaxy_images, psf_images, labels):
    """Compute loss for batch."""
    preds = state.apply_fn(params, galaxy_images, psf_images)
    loss = optax.l2_loss(preds, labels).mean()
    return loss


@jax.jit
def train_step(state, galaxy_images, psf_images, labels):
    """Single training step."""
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, galaxy_images, psf_images, labels)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(state, galaxy_images, psf_images, labels):
    """Single evaluation step."""
    loss = loss_fn(state, state.params, galaxy_images, psf_images, labels)
    return loss


def create_sgd_optimizer_with_plateau(initial_lr, momentum=0.9, weight_decay=1e-4, 
                                    plateau_patience=10, plateau_factor=0.1, 
                                    plateau_min_lr=1e-8, plateau_verbose=True):
    """
    Create SGD optimizer with momentum and ReduceLROnPlateau scheduler.
    
    Args:
        initial_lr: Initial learning rate
        momentum: SGD momentum parameter (default: 0.9, like PyTorch default)
        weight_decay: L2 regularization weight
        plateau_patience: Number of epochs with no improvement to wait before reducing LR
        plateau_factor: Factor by which to reduce LR
        plateau_min_lr: Minimum learning rate
        plateau_verbose: Whether to print LR reduction messages
    """
    # Create SGD optimizer with momentum and weight decay
    optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay),  # L2 regularization
        optax.sgd(learning_rate=initial_lr, momentum=momentum)
    )
    
    # Create ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(
        mode='min',
        factor=plateau_factor,
        patience=plateau_patience,
        min_lr=plateau_min_lr,
        verbose=plateau_verbose
    )
    
    return optimizer, scheduler


def update_optimizer_lr(state, new_lr, momentum=0.9, weight_decay=1e-4):
    """Update the learning rate in the optimizer state."""
    # Create new optimizer with updated learning rate
    new_tx = optax.chain(
        optax.add_decayed_weights(weight_decay),
        optax.sgd(learning_rate=new_lr, momentum=momentum)
    )
    
    # Create new state with updated optimizer
    new_state = train_state.TrainState.create(
        apply_fn=state.apply_fn,
        params=state.params,
        tx=new_tx
    )
    
    return new_state


def train_model(galaxy_images, psf_images, labels, rng_key, epochs=10, batch_size=32, nn="forklike", 
                save_path=None, model_name="my_model", val_split=0.2, eval_interval=1, 
                patience=5, lr=1e-3, weight_decay=1e-4, galaxy_model_type="cnn", psf_model_type="cnn",
                momentum=0.9, plateau_patience=10, plateau_factor=0.1, plateau_min_lr=1e-8):
    """Enhanced training function with SGD + ReduceLROnPlateau scheduler."""
    
    # Split into train and validation sets
    split_idx = int(len(galaxy_images) * (1 - val_split))
    train_galaxy_images, val_galaxy_images = galaxy_images[:split_idx], galaxy_images[split_idx:]
    train_psf_images, val_psf_images = psf_images[:split_idx], psf_images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    if nn == "forklike":
        model = ForkLike(galaxy_model_type=galaxy_model_type, psf_model_type=psf_model_type)
    else:
        raise ValueError("Only 'forklike' model type is supported.")
    
    params = model.init(rng_key, jnp.ones_like(galaxy_images[0]), jnp.ones_like(psf_images[0]))
    
    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr, 
        decay_steps=epochs * (len(train_galaxy_images) // batch_size)
    )
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    current_lr = lr

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
            state, loss = train_step(state, batch_galaxy_images, batch_psf_images, batch_labels)
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
                loss = eval_step(state, batch_galaxy_images, batch_psf_images, batch_labels)
                val_loss += loss * batch_size_actual
                total_samples += batch_size_actual
            val_loss /= total_samples
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4e}")

            '''
            # ReduceLROnPlateau step
            new_lr = lr_scheduler.step(val_loss, current_lr)
            if new_lr != current_lr:
                current_lr = new_lr
                state = update_optimizer_lr(state, current_lr, momentum, weight_decay)
            '''

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"New best validation loss: {val_loss:.4e}")
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Save checkpoint
    if save_path:
        save_checkpoint(state, step=epoch + 1, checkpoint_dir=save_path, 
                        model_name=model_name, overwrite=True)

    return state, train_losses, val_losses
