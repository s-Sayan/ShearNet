"""Plotting utilities for PSF deconvolution evaluation."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import jax.numpy as jnp
from typing import Optional

def ensure_consistent_shapes(*arrays):
    """Ensure all arrays have the same number of dimensions by adding channel dim if needed."""
    result = []
    for arr in arrays:
        if arr.ndim == 3:
            arr = arr[..., None]  # Add channel dimension
        result.append(arr)
    return result


def plot_deconv_samples(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                       target_images: jnp.ndarray, neural_predictions: jnp.ndarray,
                       num_samples: int = 5, path: Optional[str] = None):
    num_samples = min(num_samples, len(galaxy_images))
    
    # 4 columns now (Observed, PSF, Target, Neural)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Observed galaxy image
        im1 = axes[i, 0].imshow(galaxy_images[i].squeeze(), cmap='viridis', origin='lower')
        axes[i, 0].set_title(f'Sample {i+1}: Observed')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
        
        # PSF image
        im2 = axes[i, 1].imshow(psf_images[i].squeeze(), cmap='viridis', origin='lower')
        axes[i, 1].set_title('PSF')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
        
        # Target (ground truth)
        im3 = axes[i, 2].imshow(target_images[i].squeeze(), cmap='viridis', origin='lower')
        axes[i, 2].set_title('Target (Truth)')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)
        
        # Neural prediction
        im4 = axes[i, 3].imshow(neural_predictions[i].squeeze(), cmap='viridis', origin='lower')
        axes[i, 3].set_title('Neural Deconv')
        axes[i, 3].axis('off')
        plt.colorbar(im4, ax=axes[i, 3], shrink=0.8)
        
        # Calculate metrics for both methods
        neural_mse = float(jnp.mean((neural_predictions[i] - target_images[i]) ** 2))
        neural_psnr = -10 * jnp.log10(neural_mse) if neural_mse > 0 else float('inf')
        
        # Add metrics text
        metrics_text = (f'Neural: MSE={neural_mse:.2e}, PSNR={neural_psnr:.1f}dB')
        
        fig.text(0.02, 0.95 - (i * 0.95 / num_samples), metrics_text,
                fontsize=10, weight='bold')
    
    plt.tight_layout()
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Deconvolution samples plot saved to: {path}")
    else:
        plt.show()
    
    plt.close()

def plot_deconv_compare_samples( 
                       target_images: jnp.ndarray, neural_predictions: jnp.ndarray,
                       fft_predictions: jnp.ndarray, weiner_predictions: jnp.ndarray, richardson_lucy_predictions: jnp.ndarray,
                       num_samples: int = 5, path: Optional[str] = None):
    num_samples = min(num_samples, len(target_images))
    
    # 5 columns (Target, Neural, FFT, Weiner, RL)
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        
        # Target (ground truth)
        im1= axes[i, 0].imshow(target_images[i].squeeze(), cmap='viridis', origin='lower')
        axes[i, 0].set_title('Target (Truth)')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
        
        # Neural prediction
        im2 = axes[i, 1].imshow(neural_predictions[i].squeeze(), cmap='viridis', origin='lower')
        axes[i, 1].set_title('Neural Deconv')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)

        # FFT prediction
        fft_pred = fft_predictions[i].squeeze()
        im3 = axes[i, 2].imshow(fft_pred, cmap='viridis', origin='lower')
        axes[i, 2].set_title('FFT Deconv')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)

        # Weiner prediction
        weiner_pred = weiner_predictions[i].squeeze()
        im4 = axes[i, 3].imshow(weiner_pred, cmap='viridis', origin='lower')
        axes[i, 3].set_title('Weiner Deconv')
        axes[i, 3].axis('off')
        plt.colorbar(im4, ax=axes[i, 3], shrink=0.8)

        # RL prediction
        richardson_lucy_pred = richardson_lucy_predictions[i].squeeze()
        im5 = axes[i, 4].imshow(richardson_lucy_pred, cmap='viridis', origin='lower')
        axes[i, 4].set_title('RL Deconv')
        axes[i, 4].axis('off')
        plt.colorbar(im5, ax=axes[i, 4], shrink=0.8)
    
    plt.tight_layout()
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Deconvolution samples plot saved to: {path}")
    else:
        plt.show()
    
    plt.close()


def plot_deconv_metrics(target_images: jnp.ndarray, 
                       neural_predictions: jnp.ndarray,
                       fft_predictions: jnp.ndarray,
                       path: Optional[str] = None, 
                       title: str = "Deconvolution Performance"):
    # Ensure consistent shapes
    target_images, neural_predictions, fft_predictions = ensure_consistent_shapes(
        target_images, neural_predictions, fft_predictions
    )
    
    # Calculate per-image metrics for both methods
    num_images = len(target_images)
    neural_mse_per_image = []
    neural_psnr_per_image = []
    fft_mse_per_image = []
    fft_psnr_per_image = []
    
    for i in range(num_images):
        # Neural metrics
        neural_mse = float(jnp.mean((neural_predictions[i] - target_images[i]) ** 2))
        neural_psnr = -10 * jnp.log10(neural_mse) if neural_mse > 0 else 100
        neural_mse_per_image.append(neural_mse)
        neural_psnr_per_image.append(neural_psnr)
        
        # FFT metrics
        fft_mse = float(jnp.mean((fft_predictions[i] - target_images[i]) ** 2))
        fft_psnr = -10 * jnp.log10(fft_mse) if fft_mse > 0 else 100
        fft_mse_per_image.append(fft_mse)
        fft_psnr_per_image.append(fft_psnr)
    
    # Calculate residuals for both methods
    neural_residuals = neural_predictions - target_images
    fft_residuals = fft_predictions - target_images
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # MSE distribution - both methods
    bins = np.linspace(0, max(max(neural_mse_per_image), max(fft_mse_per_image)), 50)
    axes[0, 0].hist(neural_mse_per_image, bins=bins, alpha=0.7, label='Neural')
    axes[0, 0].hist(fft_mse_per_image, bins=bins, alpha=0.7, label='FFT')
    axes[0, 0].set_xlabel('Mean Squared Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MSE Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR distribution - both methods
    bins = np.linspace(min(min(neural_psnr_per_image), min(fft_psnr_per_image)), 
                      max(max(neural_psnr_per_image), max(fft_psnr_per_image)), 50)
    axes[0, 1].hist(neural_psnr_per_image, bins=bins, alpha=0.7, label='Neural')
    axes[0, 1].hist(fft_psnr_per_image, bins=bins, alpha=0.7, label='FFT')
    axes[0, 1].set_xlabel('PSNR (dB)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('PSNR Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals comparison
    neural_residuals_flat = neural_residuals.flatten()
    fft_residuals_flat = fft_residuals.flatten()
    
    # Clip for visualization
    all_residuals = np.concatenate([neural_residuals_flat, fft_residuals_flat])
    percentile_1 = np.percentile(all_residuals, 1)
    percentile_99 = np.percentile(all_residuals, 99)
    
    neural_clipped = neural_residuals_flat[
        (neural_residuals_flat >= percentile_1) & 
        (neural_residuals_flat <= percentile_99)
    ]
    fft_clipped = fft_residuals_flat[
        (fft_residuals_flat >= percentile_1) & 
        (fft_residuals_flat <= percentile_99)
    ]
    
    bins = np.linspace(percentile_1, percentile_99, 100)
    axes[0, 2].hist(neural_clipped, bins=bins, alpha=0.7, label='Neural')
    axes[0, 2].hist(fft_clipped, bins=bins, alpha=0.7, label='FFT')
    axes[0, 2].axvline(0, color='red', linestyle='--', label='Zero residual')
    axes[0, 2].set_xlabel('Residual (Predicted - Target)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Residuals Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # MSE per image comparison
    axes[1, 0].plot(range(num_images), neural_mse_per_image, 'b-', alpha=0.7, label='Neural')
    axes[1, 0].plot(range(num_images), fft_mse_per_image, 'r-', alpha=0.7, label='FFT')
    axes[1, 0].set_xlabel('Image Index')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE per Image')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # PSNR per image comparison
    axes[1, 1].plot(range(num_images), neural_psnr_per_image, 'b-', alpha=0.7, label='Neural')
    axes[1, 1].plot(range(num_images), fft_psnr_per_image, 'r-', alpha=0.7, label='FFT')
    axes[1, 1].set_xlabel('Image Index')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title('PSNR per Image')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Improvement scatter plot
    mse_improvement = jnp.array(fft_mse_per_image) - jnp.array(neural_mse_per_image)
    axes[1, 2].scatter(range(num_images), mse_improvement, alpha=0.5)
    axes[1, 2].axhline(0, color='red', linestyle='--')
    axes[1, 2].set_xlabel('Image Index')
    axes[1, 2].set_ylabel('MSE Improvement (FFT - Neural)')
    axes[1, 2].set_title('MSE Improvement per Image')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Deconvolution metrics plot saved to: {path}")
    else:
        plt.show()
    
    plt.close()


def plot_deconv_comparison(galaxy_images: jnp.ndarray, target_images: jnp.ndarray,
                          neural_predictions: jnp.ndarray, fft_predictions: jnp.ndarray,
                          num_samples: int = 3, path: Optional[str] = None):
    """
    Plot comparison between neural network and FFT deconvolution methods.
    
    Args:
        galaxy_images: Observed galaxy images
        target_images: Ground truth clean images
        neural_predictions: Neural network deconvolution results
        fft_predictions: FFT deconvolution results
        num_samples: Number of samples to plot
        path: Path to save the plot
    """
    # Ensure consistent shapes
    galaxy_images, target_images, neural_predictions, fft_predictions = ensure_consistent_shapes(
        galaxy_images, target_images, neural_predictions, fft_predictions
    )

    num_samples = min(num_samples, len(galaxy_images))
    
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        galaxy_img = galaxy_images[i].squeeze()
        target_img = target_images[i].squeeze()
        neural_img = neural_predictions[i].squeeze()
        fft_img = fft_predictions[i].squeeze()

        # Observed galaxy
        im1 = axes[i, 0].imshow(galaxy_img, cmap='viridis', origin='lower')
        axes[i, 0].set_title(f'Sample {i+1}: Observed')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
        
        # Target
        im2 = axes[i, 1].imshow(target_img, cmap='viridis', origin='lower')
        axes[i, 1].set_title('Target (Truth)')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
        
        # Neural prediction
        im3 = axes[i, 2].imshow(neural_img, cmap='viridis', origin='lower')
        axes[i, 2].set_title('Neural Deconv')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)
        
        # FFT prediction
        fft_pred = fft_img
        if fft_pred.ndim == 3 and fft_pred.shape[-1] == 1:
            fft_pred = fft_pred[..., 0]
        im4 = axes[i, 3].imshow(fft_pred, cmap='viridis', origin='lower')
        axes[i, 3].set_title('FFT Deconv')
        axes[i, 3].axis('off')
        plt.colorbar(im4, ax=axes[i, 3], shrink=0.8)
        
        # Difference map (Neural - FFT)
        diff = neural_img - fft_pred
        im5 = axes[i, 4].imshow(diff, cmap='RdBu_r', origin='lower')
        axes[i, 4].set_title('Difference\n(Neural - FFT)')
        axes[i, 4].axis('off')
        plt.colorbar(im5, ax=axes[i, 4], shrink=0.8)
        
        # Calculate metrics for both methods
        neural_mse = float(jnp.mean((neural_img - target_img) ** 2))
        fft_mse = float(jnp.mean((fft_pred - target_img) ** 2))
        neural_psnr = -10 * jnp.log10(neural_mse) if neural_mse > 0 else float('inf')
        fft_psnr = -10 * jnp.log10(fft_mse) if fft_mse > 0 else float('inf')
        
        # Add metrics text
        metrics_text = (f'Neural: MSE={neural_mse:.2e}, PSNR={neural_psnr:.1f}dB\n'
                       f'FFT: MSE={fft_mse:.2e}, PSNR={fft_psnr:.1f}dB')
        
        fig.text(0.02, 0.95 - (i * 0.95 / num_samples), metrics_text,
                fontsize=10, weight='bold')
    
    plt.suptitle('Deconvolution Method Comparison', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Deconvolution comparison plot saved to: {path}")
    else:
        plt.show()
    
    plt.close()


def plot_deconv_residuals(target_images: jnp.ndarray, neural_predictions: jnp.ndarray,
                         fft_predictions: jnp.ndarray, num_samples: int = 3,
                         path: Optional[str] = None):
    """
    Plot residual maps for both deconvolution methods.
    
    Args:
        target_images: Ground truth images
        neural_predictions: Neural network predictions
        fft_predictions: FFT predictions
        num_samples: Number of samples to plot
        path: Path to save the plot
    """
    # Ensure consistent shapes
    target_images, neural_predictions, fft_predictions = ensure_consistent_shapes(
        target_images, neural_predictions, fft_predictions
    )

    num_samples = min(num_samples, len(target_images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        target_img = target_images[i].squeeze()
        neural_img = neural_predictions[i].squeeze()
        fft_img = fft_predictions[i].squeeze()

        # Neural residuals
        neural_residual = neural_img - target_img
        im1 = axes[i, 0].imshow(neural_residual, cmap='RdBu_r', origin='lower')
        axes[i, 0].set_title(f'Sample {i+1}: Neural Residuals')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
        
        # FFT residuals
        fft_pred = fft_img
        if fft_pred.ndim == 3 and fft_pred.shape[-1] == 1:
            fft_pred = fft_pred[..., 0]
        fft_residual = fft_pred - target_images[i]
        im2 = axes[i, 1].imshow(fft_residual, cmap='RdBu_r', origin='lower')
        axes[i, 1].set_title('FFT Residuals')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
        
        # Difference in residuals
        residual_diff = neural_residual - fft_residual
        im3 = axes[i, 2].imshow(residual_diff, cmap='RdBu_r', origin='lower')
        axes[i, 2].set_title('Residual Difference\n(Neural - FFT)')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)
        
        # Add RMS residual values
        neural_rms = float(jnp.sqrt(jnp.mean(neural_residual**2)))
        fft_rms = float(jnp.sqrt(jnp.mean(fft_residual**2)))
        
        fig.text(0.02, 0.95 - (i * 0.95 / num_samples), 
                f'Sample {i+1} RMS - Neural: {neural_rms:.3e}, FFT: {fft_rms:.3e}',
                fontsize=10, weight='bold')
    
    plt.suptitle('Deconvolution Residual Analysis', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Deconvolution residuals plot saved to: {path}")
    else:
        plt.show()
    
    plt.close()


def plot_learning_curve_deconv(train_losses, val_losses, path: Optional[str] = None):
    """
    Plot learning curves for deconvolution training.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs_train = range(1, len(train_losses) + 1)
    epochs_val = range(1, len(val_losses) + 1)
    
    plt.plot(epochs_train, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs_val, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('PSF Deconvolution Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add best validation loss marker
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        plt.plot(best_epoch, best_loss, 'ro', markersize=8, 
                label=f'Best Val Loss: {best_loss:.3e} (Epoch {best_epoch})')
        plt.legend()
    
    plt.tight_layout()
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Learning curve plot saved to: {path}")
    else:
        plt.show()
    
    plt.close()