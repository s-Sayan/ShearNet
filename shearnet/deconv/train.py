"""Core training functions for PSF deconvolution networks."""

import os
import jax
import jax.numpy as jnp
import optax
import pickle
from flax.training import train_state
from .models import PSFDeconvolutionNet, SimplePSFDeconvNet, create_deconv_net
from flax.training import checkpoints

def save_checkpoint(state, step, checkpoint_dir, model_name, overwrite=True):
    """Save the model checkpoint using Flax's built-in method."""
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=state,
        step=step,
        prefix=model_name,
        overwrite=overwrite
    )
    print(f"Checkpoint saved at step {step} to {checkpoint_dir}")


def deconv_loss_fn(state, params, galaxy_images, psf_images, target_images, training=True):
    """Compute reconstruction loss for deconvolution."""
    deconvolved = state.apply_fn(params, galaxy_images, psf_images, training=training)
    loss = optax.l2_loss(deconvolved.squeeze(-1), target_images).mean()
    return loss


@jax.jit
def deconv_train_step(state, galaxy_images, psf_images, target_images):
    """Single training step for deconvolution."""
    grad_fn = jax.value_and_grad(deconv_loss_fn, argnums=1, has_aux=False)
    loss, grads = grad_fn(state, state.params, galaxy_images, psf_images, target_images, training=True)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit  
def deconv_eval_step(state, galaxy_images, psf_images, target_images):
    """Single evaluation step for deconvolution."""
    loss = deconv_loss_fn(state, state.params, galaxy_images, psf_images, target_images, training=False)
    return loss


def train_deconv_model(galaxy_images, psf_images, target_images, rng_key, 
                      epochs=50, batch_size=32, model_type="unet", 
                      save_path=None, model_name="deconv_model", val_split=0.2, 
                      eval_interval=1, patience=10, lr=1e-3, weight_decay=1e-4):
    """Train a PSF deconvolution model with improved checkpoint saving."""
    
    # Split into train and validation sets
    split_idx = int(len(galaxy_images) * (1 - val_split))
    train_galaxy = galaxy_images[:split_idx]
    train_psf = psf_images[:split_idx] 
    train_targets = target_images[:split_idx]
    val_galaxy = galaxy_images[split_idx:]
    val_psf = psf_images[split_idx:]
    val_targets = target_images[split_idx:]

    # Initialize model
    if model_type == "unet":
        model = PSFDeconvolutionNet()
    elif model_type == "simple":
        model = SimplePSFDeconvNet()
    else:
        raise ValueError(f"Invalid model type specified: {model_type}")
    
    # Initialize parameters and state
    print(f"Initializing {model_type} model...")
    sample_galaxy = jnp.expand_dims(train_galaxy[0], axis=0)
    sample_psf = jnp.expand_dims(train_psf[0], axis=0)
    params = model.init(rng_key, sample_galaxy, sample_psf, training=True)
    
    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr, 
        decay_steps=epochs * (len(train_galaxy) // batch_size)
    )
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None  # To store the best model state
    patience_counter = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
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
            
            state, loss = deconv_train_step(state, batch_galaxy, batch_psf, batch_targets)
            train_loss += loss * batch_size_actual
            total_samples += batch_size_actual
            
        train_loss /= total_samples
        train_losses.append(train_loss)

        # Validation phase
        if (epoch + 1) % eval_interval == 0:
            val_loss, total_samples = 0, 0
            for i in range(0, len(val_galaxy), batch_size):
                batch_galaxy = val_galaxy[i:i + batch_size]
                batch_psf = val_psf[i:i + batch_size]
                batch_targets = val_targets[i:i + batch_size]
                batch_size_actual = len(batch_galaxy)
                
                loss = deconv_eval_step(state, batch_galaxy, batch_psf, batch_targets)
                val_loss += loss * batch_size_actual
                total_samples += batch_size_actual
                
            val_loss /= total_samples
            val_losses.append(val_loss)
            print(f"Validation Loss: {val_loss:.4e}")

            # Check for improvement and save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = state  # Save the best state
                patience_counter = 0
                print(f"New best validation loss: {val_loss:.4e}")
                
                # Save best checkpoint immediately
                if save_path:
                    save_checkpoint(state, step=epoch+1, checkpoint_dir=save_path, 
                                   model_name=f"{model_name}_best", overwrite=True)
            else:
                patience_counter += 1
                print(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    # Save final checkpoint
    if save_path:
        # Save the best model if we have one, otherwise save the last model
        final_state = best_state if best_state is not None else state
        save_checkpoint(final_state, step=epoch+1, checkpoint_dir=save_path, 
                       model_name=f"{model_name}_final", overwrite=True)

    return state, train_losses, val_losses


def generate_deconv_predictions(state, galaxy_images, psf_images, batch_size=32):
    """
    Generate deconvolved predictions for a set of galaxy and PSF images.
    
    Args:
        state: Trained model state
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        batch_size: Batch size for inference
        
    Returns:
        Deconvolved galaxy images
    """
    predictions = []
    
    for i in range(0, len(galaxy_images), batch_size):
        batch_galaxy = galaxy_images[i:i + batch_size]
        batch_psf = psf_images[i:i + batch_size]
        
        # Generate predictions (training=False for inference mode)
        batch_preds = state.apply_fn(state.params, batch_galaxy, batch_psf, training=False)
        predictions.append(batch_preds)
    
    return jnp.concatenate(predictions, axis=0)


def evaluate_deconv_model(state, galaxy_images, psf_images, target_images, batch_size=32):
    """
    Evaluate a trained deconvolution model.
    
    Args:
        state: Trained model state
        galaxy_images: Test galaxy images
        psf_images: Test PSF images  
        target_images: True target images
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with evaluation metrics
    """
    total_loss = 0
    total_batches = 0
    
    for i in range(0, len(galaxy_images), batch_size):
        batch_galaxy = galaxy_images[i:i + batch_size]
        batch_psf = psf_images[i:i + batch_size]
        batch_targets = target_images[i:i + batch_size]
        
        loss = deconv_eval_step(state, batch_galaxy, batch_psf, batch_targets)
        total_loss += loss
        total_batches += 1
    
    avg_loss = total_loss / total_batches
    
    # Generate all predictions for additional metrics
    predictions = generate_deconv_predictions(state, galaxy_images, psf_images, batch_size)
    
    # Calculate additional metrics
    mse = jnp.mean((predictions - target_images) ** 2)
    psnr = -10 * jnp.log10(mse)
    
    results = {
        'loss': float(avg_loss),
        'mse': float(mse), 
        'psnr': float(psnr),
        'predictions': predictions
    }
    
    print(f"\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.6e}")
    print(f"MSE: {mse:.6e}")
    print(f"PSNR: {psnr:.2f} dB")
    
    return results