"""Core training functions for DeconvNet models."""

import os
import jax
import jax.numpy as jnp
import jax.scipy.signal as jsp
import optax
import galsim
from flax.training import train_state, checkpoints
from .models import PSFDeconvolutionNet, EnhancedPSFDeconvNet, ResearchBackedPSFDeconvolutionUNet

@jax.jit
def convolve_pair(img_a, img_b):
    return jsp.convolve(img_a, img_b, mode='same')

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

def loss_fn(state, params, galaxy_images, psf_images, target_images, config,  deterministic=False):
    """Compute loss for batch."""

    # # Drop channel dimension if present
    # if galaxy_images.ndim == 4:       # (B, H, W, C)
    #     galaxy_images = galaxy_images.squeeze(-1)
    # if target_images.ndim == 4:
    #     target_images = target_images.squeeze(-1)
    # if psf_images.ndim == 4:
    #     psf_images = psf_images.squeeze(-1)

    # batch_convolve = jax.vmap(convolve_pair) # vectorized differentiable convolution with jax

    # # New implementation of loss
    # dropout_key = jax.random.PRNGKey(state.step)
    # preds = state.apply_fn(params, galaxy_images, psf_images, deterministic=deterministic, rngs={'dropout': dropout_key})
    # if preds.ndim == 4:
    #     preds = preds.squeeze(-1)
    # l2_loss = optax.l2_loss(preds, target_images).mean()

    # # Reblur convolution-consistency loss
    # dirty_images = batch_convolve(preds, psf_images)
    # reblur_loss = (optax.l2_loss(dirty_images, target_images) / config[4]**2).mean()

    # # Total Variation Loss
    # dx = jnp.abs(preds[:, 1:, :] - preds[:, :-1, :])
    # dy = jnp.abs(preds[:, :, 1:] - preds[:, :, :-1])
    # tv_loss = (dx.mean() + dy.mean())

    # # Laplacian Loss
    # laplace_kernel = jnp.array([[0,  1, 0],
    #                             [1, -4, 1],
    #                             [0,  1, 0]], dtype=preds.dtype) 
    #                             # Discrete Laplacian  kernel
    # lap = batch_convolve(preds, jnp.broadcast_to(laplace_kernel[None, :, :], (preds.shape[0], 3, 3)))
    # lap_loss = (lap**2).mean()

    # # weighted sum of different loss types
    # loss = (
    #         (config[0] * l2_loss) 
    #         + (config[1] * reblur_loss) 
    #         + (config[2] * tv_loss) 
    #         + (config[3] * lap_loss)
    #     )

    dropout_key = jax.random.PRNGKey(state.step)
    deconvolved = state.apply_fn(params, galaxy_images, psf_images, deterministic=deterministic, rngs={'dropout': dropout_key})
    loss = optax.l2_loss(deconvolved.squeeze(-1), target_images).mean()

    return loss


@jax.jit
def train_step(state, galaxy_images, psf_images, target_images, config):
    """Single training step for deconvolution."""
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, galaxy_images, psf_images, target_images, config=config, deterministic=False)
    state = state.apply_gradients(grads=grads)
    
    return state, loss

@jax.jit  
def eval_step(state, galaxy_images, psf_images, target_images, config):
    """Single evaluation step for deconvolution."""
    loss = loss_fn(state, state.params, galaxy_images, psf_images, target_images, config=config, deterministic=True)

    return loss

def get_optimizer_config(model_type: str, lr: float, weight_decay: float, epochs: int, steps_per_epoch: int):
    """
    Get optimizer configuration optimized for different model types.
    
    Research-backed models may benefit from different optimization strategies.
    """
    
    if model_type.startswith("research_backed"):
        # Research-backed models: cosine schedule with warmup
        warmup_steps = min(1000, steps_per_epoch)
        decay_steps = epochs * steps_per_epoch - warmup_steps
        
        # Warmup schedule
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=lr,
            transition_steps=warmup_steps
        )
        
        # Cosine decay after warmup
        cosine_schedule = optax.cosine_decay_schedule(
            init_value=lr,
            decay_steps=decay_steps,
            alpha=0.01  # Final LR = 1% of initial
        )
        
        # Combine schedules
        lr_schedule = optax.join_schedules(
            schedules=[warmup_schedule, cosine_schedule],
            boundaries=[warmup_steps]
        )
        
        # AdamW with gradient clipping for stability
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
                b1=0.9,
                b2=0.999,
                eps=1e-8
            )
        )
        
    else:
        # Standard models: simple cosine decay
        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr, 
            decay_steps=epochs * steps_per_epoch
        )
        tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    
    return tx

def train_model(galaxy_images, psf_images, target_images, rng_key, 
                epochs=50, batch_size=32, model_type="research_backed", 
                save_path=None, model_name="deconvnet_model", val_split=0.2, 
                eval_interval=1, patience=10, lr=1e-3, weight_decay=1e-4, loss_config=[1.0,1.0,1.0,1.0, 1e-5]):

    # Split into train and validation sets
    split_idx = int(len(galaxy_images) * (1 - val_split))
    train_galaxy = galaxy_images[:split_idx]
    train_psf = psf_images[:split_idx] 
    train_targets = target_images[:split_idx]
    val_galaxy = galaxy_images[split_idx:]
    val_psf = psf_images[split_idx:]
    val_targets = target_images[split_idx:]

    if model_type == "base":
        model = PSFDeconvolutionNet()
    elif model_type == "enhanced":
        model = EnhancedPSFDeconvNet()
    elif model_type == "research_backed":
        model = ResearchBackedPSFDeconvolutionUNet()
    else:
        raise ValueError(f"Invalid model type specified: {model_type}")

    # Initialize parameters and state
    sample_galaxy = jnp.expand_dims(train_galaxy[0], axis=0)
    sample_psf = jnp.expand_dims(train_psf[0], axis=0)

    params = model.init(rng_key, sample_galaxy, sample_psf, deterministic=False)

    # Get optimized training configuration
    steps_per_epoch = len(train_galaxy) // batch_size
    tx = get_optimizer_config(model_type, lr, weight_decay, epochs, steps_per_epoch)
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # Initialize best validation loss
    patience_counter = 0  # Counter for early stopping

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Shuffle training data
        rng_key, subkey = jax.random.split(rng_key)
        perm = jax.random.permutation(subkey, len(train_galaxy))
        shuffled_galaxy = train_galaxy[perm]
        shuffled_psf = train_psf[perm]
        shuffled_targets = train_targets[perm]

        # Training phase
        train_loss, total_samples = 0, 0
        for i in range(0, len(train_galaxy), batch_size):
            batch_galaxy = shuffled_galaxy[i:i + batch_size]
            batch_psf = shuffled_psf[i:i + batch_size]
            batch_targets = shuffled_targets[i:i + batch_size]
            batch_size_actual = len(batch_galaxy)
            
            state, loss = train_step(state, batch_galaxy, batch_psf, batch_targets, loss_config)
            train_loss += loss * batch_size_actual
            total_samples += batch_size_actual
            
        train_loss /= total_samples
        train_losses.append(train_loss)
        print(f"  Training Loss: {train_loss:.6e}")

        # Validation phase
        if (epoch + 1) % eval_interval == 0:
            val_loss, total_samples = 0, 0
            for i in range(0, len(val_galaxy), batch_size):
                batch_galaxy = val_galaxy[i:i + batch_size]
                batch_psf = val_psf[i:i + batch_size]
                batch_targets = val_targets[i:i + batch_size]
                batch_size_actual = len(batch_galaxy)
                
                loss =eval_step(state, batch_galaxy, batch_psf, batch_targets, config=loss_config)
                val_loss += loss * batch_size_actual
                total_samples += batch_size_actual
                
            val_loss /= total_samples
            val_losses.append(val_loss)
            print(f"  Validation Loss: {val_loss:.6e}")

            # Check for improvement and save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  New best validation loss: {val_loss:.6e}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"\n Early stopping triggered at epoch {epoch + 1}")
                    break

    # Save checkpoint
    if save_path:
        save_checkpoint(state, step=epoch+1, checkpoint_dir=save_path, 
                       model_name=model_name, overwrite=True)

    print(f"\n Training completed!")
    print(f"Best validation loss: {best_val_loss:.6e}")
    
    return state, train_losses, val_losses