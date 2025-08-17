"""Enhanced training functions for PSF deconvolution networks.

This module extends the existing training infrastructure to support research-backed
U-Net architectures while maintaining full compatibility with existing workflows.
"""

import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from .models import (
    PSFDeconvolutionNet, 
    SimplePSFDeconvNet, 
    EnhancedPSFDeconvNet,
    create_deconv_net,
    get_model_for_training,
    create_research_backed_deconv_unet
)


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
    dropout_key = jax.random.PRNGKey(state.step)
    deconvolved = state.apply_fn(params, galaxy_images, psf_images, training=training, rngs={'dropout': dropout_key})
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
        
        print(f"Using optimized training config for {model_type}:")
        print(f"  - Warmup steps: {warmup_steps}")
        print(f"  - Cosine decay steps: {decay_steps}")
        print(f"  - Gradient clipping: 1.0")
        
    else:
        # Standard models: simple cosine decay
        lr_schedule = optax.cosine_decay_schedule(
            init_value=lr, 
            decay_steps=epochs * steps_per_epoch
        )
        tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    
    return tx


def train_deconv_model(galaxy_images, psf_images, target_images, rng_key, 
                      epochs=50, batch_size=32, model_type="unet", 
                      save_path=None, model_name="deconv_model", val_split=0.2, 
                      eval_interval=1, patience=10, lr=1e-3, weight_decay=1e-4,
                      architecture="full", **model_kwargs):
    """
    Enhanced training function with support for research-backed models.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        target_images: Clean target images
        rng_key: JAX random key
        epochs: Number of training epochs
        batch_size: Training batch size
        model_type: Type of model to train ('unet', 'simple', 'enhanced', 
                   'research_backed', 'research_backed_lite', 'research_backed_minimal')
        save_path: Path to save checkpoints
        model_name: Name for saved model
        val_split: Validation split fraction
        eval_interval: Evaluate every N epochs
        patience: Early stopping patience
        lr: Learning rate
        weight_decay: Weight decay
        architecture: Architecture preset for research-backed models
        **model_kwargs: Additional model parameters
    
    Returns:
        (state, train_losses, val_losses)
    """
    
    # Split into train and validation sets
    split_idx = int(len(galaxy_images) * (1 - val_split))
    train_galaxy = galaxy_images[:split_idx]
    train_psf = psf_images[:split_idx] 
    train_targets = target_images[:split_idx]
    val_galaxy = galaxy_images[split_idx:]
    val_psf = psf_images[split_idx:]
    val_targets = target_images[split_idx:]

    # Initialize model with enhanced creation logic
    print(f"Initializing {model_type} model...")
    
    if model_type.startswith("research_backed"):
        if model_type == "research_backed":
            model = create_research_backed_deconv_unet(
                architecture=architecture,
                **model_kwargs
            )
        else:
            # Extract preset from model type
            preset = model_type.split("_", 2)[-1]  # e.g., "lite" or "minimal"
            model = create_research_backed_deconv_unet(
                architecture=preset,
                **model_kwargs
            )
    else:
        # Use existing model creation
        model = get_model_for_training(model_type, **model_kwargs)
    
    # Initialize parameters and state
    sample_galaxy = jnp.expand_dims(train_galaxy[0], axis=0)
    sample_psf = jnp.expand_dims(train_psf[0], axis=0)
    
    print(f"Model input shapes - Galaxy: {sample_galaxy.shape}, PSF: {sample_psf.shape}")
    
    params = model.init(rng_key, sample_galaxy, sample_psf, training=True)
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameter count: {param_count:,}")
    
    # Get optimized training configuration
    steps_per_epoch = len(train_galaxy) // batch_size
    tx = get_optimizer_config(model_type, lr, weight_decay, epochs, steps_per_epoch)
    
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(train_galaxy):,}")
    print(f"Validation samples: {len(val_galaxy):,}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")

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
            
            state, loss = deconv_train_step(state, batch_galaxy, batch_psf, batch_targets)
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
                
                loss = deconv_eval_step(state, batch_galaxy, batch_psf, batch_targets)
                val_loss += loss * batch_size_actual
                total_samples += batch_size_actual
                
            val_loss /= total_samples
            val_losses.append(val_loss)
            print(f"  Validation Loss: {val_loss:.6e}")

            # Check for improvement and save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = state
                patience_counter = 0
                print(f"  ✓ New best validation loss: {val_loss:.6e}")
                
                # Save best checkpoint immediately
                if save_path:
                    save_checkpoint(state, step=epoch+1, checkpoint_dir=save_path, 
                                   model_name=f"{model_name}_best", overwrite=True)
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print(f"\n⏹ Early stopping triggered at epoch {epoch + 1}")
                    break

    # Save final checkpoint
    if save_path:
        final_state = best_state if best_state is not None else state
        save_checkpoint(final_state, step=epoch+1, checkpoint_dir=save_path, 
                       model_name=f"{model_name}_final", overwrite=True)

    print(f"\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.6e}")
    
    return state, train_losses, val_losses


@jax.jit
def _predict_batch_jit(state, galaxy_batch, psf_batch):
    """JIT-compiled batch prediction function."""
    return state.apply_fn(state.params, galaxy_batch, psf_batch, training=False)

def generate_deconv_predictions(state, galaxy_images, psf_images, batch_size=32):
    """Generate deconvolved predictions for a set of galaxy and PSF images."""
    predictions = []
    
    # Compile the function on first call
    print(f"Generating predictions for {len(galaxy_images)} images...")
    
    for i in range(0, len(galaxy_images), batch_size):
        batch_galaxy = galaxy_images[i:i + batch_size]
        batch_psf = psf_images[i:i + batch_size]
        
        # Use JIT-compiled prediction
        batch_preds = _predict_batch_jit(state, batch_galaxy, batch_psf)
        predictions.append(batch_preds)
    
    return jnp.concatenate(predictions, axis=0)


def evaluate_deconv_model(state, galaxy_images, psf_images, target_images, batch_size=32):
    """
    Evaluate a trained deconvolution model with enhanced metrics.
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
    mae = jnp.mean(jnp.abs(predictions - target_images))
    psnr = -10 * jnp.log10(mse) if mse > 0 else float('inf')
    
    # Calculate bias and std
    residuals = predictions - target_images
    bias = jnp.mean(residuals)
    residual_std = jnp.std(residuals)
    
    # Calculate additional astronomical metrics
    # Signal preservation ratio
    signal_ratio = jnp.mean(predictions) / jnp.mean(target_images)
    
    results = {
        'loss': float(avg_loss),
        'mse': float(mse), 
        'mae': float(mae),
        'psnr': float(psnr),
        'bias': float(bias),
        'residual_std': float(residual_std),
        'signal_ratio': float(signal_ratio),
        'predictions': predictions
    }
    
    print(f"\nEvaluation Results:")
    print(f"Average Loss: {avg_loss:.6e}")
    print(f"MSE: {mse:.6e}")
    print(f"MAE: {mae:.6e}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Bias: {bias:+.6e}")
    print(f"Residual Std: {residual_std:.6e}")
    print(f"Signal Ratio: {signal_ratio:.4f}")
    
    return results


def benchmark_models(galaxy_images, psf_images, target_images, rng_key, 
                    models_to_test=None, epochs=10, batch_size=16):
    """
    Benchmark different model architectures on the same dataset.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images  
        target_images: Clean target images
        rng_key: JAX random key
        models_to_test: List of model types to test
        epochs: Number of epochs for each model
        batch_size: Training batch size
    
    Returns:
        Dictionary with results for each model
    """
    
    if models_to_test is None:
        models_to_test = [
            "simple",
            "unet", 
            "enhanced",
            "research_backed_minimal",
            "research_backed_lite"
        ]
    
    results = {}
    
    print(f"🚀 Benchmarking {len(models_to_test)} model architectures")
    print(f"Dataset size: {len(galaxy_images):,} samples")
    print(f"Training epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print("="*60)
    
    for i, model_type in enumerate(models_to_test):
        print(f"\n[{i+1}/{len(models_to_test)}] Training {model_type}...")
        
        try:
            # Train model
            state, train_losses, val_losses = train_deconv_model(
                galaxy_images=galaxy_images,
                psf_images=psf_images,
                target_images=target_images,
                rng_key=rng_key,
                epochs=epochs,
                batch_size=batch_size,
                model_type=model_type,
                save_path=None,  # Don't save during benchmarking
                patience=epochs//2,  # Reduce patience for benchmarking
                lr=1e-3,
                weight_decay=1e-4
            )
            
            # Evaluate model
            eval_results = evaluate_deconv_model(
                state, galaxy_images, psf_images, target_images, batch_size
            )
            
            results[model_type] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_val_loss': val_losses[-1] if val_losses else float('inf'),
                'eval_results': eval_results,
                'param_count': sum(x.size for x in jax.tree_util.tree_leaves(params)),
                'status': 'success'
            }
            
            print(f"✅ {model_type} completed successfully")
            print(f"   Final val loss: {results[model_type]['final_val_loss']:.6e}")
            print(f"   Parameters: {results[model_type]['param_count']:,}")
            
        except Exception as e:
            print(f"❌ {model_type} failed: {str(e)}")
            results[model_type] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # Split RNG key for next model
        rng_key, _ = jax.random.split(rng_key)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    successful_models = {k: v for k, v in results.items() if v['status'] == 'success'}
    
    if successful_models:
        # Sort by validation loss
        sorted_models = sorted(
            successful_models.items(), 
            key=lambda x: x[1]['final_val_loss']
        )
        
        print(f"{'Model':<25} {'Val Loss':<12} {'PSNR (dB)':<10} {'Parameters':<12}")
        print("-" * 60)
        
        for model_name, results_dict in sorted_models:
            val_loss = results_dict['final_val_loss']
            psnr = results_dict['eval_results']['psnr']
            params = results_dict['param_count']
            
            print(f"{model_name:<25} {val_loss:<12.3e} {psnr:<10.1f} {params:<12,}")
        
        best_model = sorted_models[0][0]
        print(f"\n🏆 Best model: {best_model}")
    
    return results


# Example usage for testing
if __name__ == "__main__":
    print("Enhanced deconvolution training module loaded!")
    print("Available features:")
    print("  - Research-backed U-Net architectures")
    print("  - Optimized training configurations")
    print("  - Model benchmarking capabilities")
    print("  - Enhanced evaluation metrics")