"""Plotting utilities for PSF deconvolution evaluation with NGmix comparison."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import jax.numpy as jnp
from typing import Optional

def plot_comparison(target_images, neural_preds, galsim_deconv_preds, num_samples=5, save_path=None):
    """Create a simple comparison plot showing truth, neural, and galsim.deconv results."""
    
    # Ensure all images have the same shape
    if target_images.ndim == 4:
        target_images = target_images.squeeze(-1)
    if neural_preds.ndim == 4:
        neural_preds = neural_preds.squeeze(-1)
    if galsim_deconv_preds.ndim == 4:
        galsim_deconv_preds = galsim_deconv_preds.squeeze(-1)
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    
    for i in range(num_samples):
        # Truth
        axes[0, i].imshow(target_images[i], cmap='viridis')
        axes[0, i].set_title(f'Truth {i+1}')
        axes[0, i].axis('off')
        
        # Neural
        axes[1, i].imshow(neural_preds[i], cmap='viridis')
        axes[1, i].set_title(f'Neural {i+1}')
        axes[1, i].axis('off')
        
        # Galsim.deconv
        axes[2, i].imshow(galsim_deconv_preds[i], cmap='viridis')
        axes[2, i].set_title(f'Galsim.deconv {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_spatial_residuals(target_images: jnp.ndarray, neural_predictions: jnp.ndarray,
                                 galsim_deconv_predictions: jnp.ndarray, path: Optional[str] = None,
                                 title: str = "Spatial Deconvolution Residuals"):
    """
    Plot spatial residual heat maps showing systematic biases across image coordinates.
    
    This shows WHERE in the galaxy each deconvolution method systematically fails,
    averaged across all test images. Zero = perfect deconvolution at that pixel.
    
    Args:
        target_images: Ground truth clean images [N, H, W] or [N, H, W, 1]
        neural_predictions: Neural network deconvolution results [N, H, W] or [N, H, W, 1]  
        galsim_deconv_predictions: Galsim deconvolution results [N, H, W] or [N, H, W, 1]
        path: Path to save the plot
        title: Plot title
    """
    
    # Ensure consistent shapes - squeeze out channel dimensions
    if target_images.ndim == 4:
        target_images = target_images.squeeze(-1)
    if neural_predictions.ndim == 4:
        neural_predictions = neural_predictions.squeeze(-1)
    if galsim_deconv_predictions.ndim == 4:
        galsim_deconv_predictions = galsim_deconv_predictions.squeeze(-1)
    
    print(f"Computing spatial residuals for {len(target_images)} images...")
    
    # Calculate mean residuals across all images at each pixel coordinate
    # This gives systematic bias: positive = over-deconvolution, negative = under-deconvolution
    neural_spatial_bias = jnp.mean(neural_predictions - target_images, axis=0)
    galsim_deconv_spatial_bias = jnp.mean(galsim_deconv_predictions - target_images, axis=0)
    
    # Calculate difference in spatial bias patterns
    bias_difference = neural_spatial_bias - galsim_deconv_spatial_bias
    
    # Calculate standard deviation of residuals at each pixel
    neural_residual_std = jnp.std(neural_predictions - target_images, axis=0)
    galsim_deconv_residual_std = jnp.std(galsim_deconv_predictions - target_images, axis=0)
    std_difference = neural_residual_std - galsim_deconv_residual_std
    
    # Find the maximum absolute value across all bias maps for consistent scaling
    global_vmax = jnp.max(jnp.array([
        jnp.max(jnp.abs(neural_spatial_bias)),
        jnp.max(jnp.abs(galsim_deconv_spatial_bias)),
        jnp.max(jnp.abs(bias_difference))
    ]))
    
    # Find the maximum value across std maps for consistent scaling
    global_std_vmax = jnp.max(jnp.array([
        jnp.max(neural_residual_std),
        jnp.max(galsim_deconv_residual_std)
    ]))
    
    # Maximum absolute value for std difference
    std_diff_vmax = jnp.max(jnp.abs(std_difference))
    
    # Set up the plot with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # First row: Bias maps
    # Neural spatial bias
    im1 = axes[0].imshow(neural_spatial_bias, cmap='RdBu_r', origin='lower', 
                           vmin=-global_vmax, vmax=global_vmax)
    axes[0].set_title('Neural Network\nSpatial Bias Map')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Mean Residual')
    
    # Galsim.deconv spatial bias  
    im2 = axes[1].imshow(galsim_deconv_spatial_bias, cmap='RdBu_r', origin='lower',
                           vmin=-global_vmax, vmax=global_vmax)
    axes[1].set_title('Galsim.deconv\nSpatial Bias Map')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Mean Residual')
    
    # Difference in spatial bias
    im3 = axes[2].imshow(bias_difference, cmap='RdBu_r', origin='lower',
                           vmin=-global_vmax, vmax=global_vmax)
    axes[2].set_title('Bias Difference\n(Neural - Galsim.deconv)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Bias Difference')
    
    # Second row: Standard deviation maps
    # Neural standard deviation
    im4 = axes[3].imshow(neural_residual_std, cmap='viridis', origin='lower',
                           vmin=0, vmax=global_std_vmax)
    axes[3].set_title('Neural Residual\nStandard Deviation')
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04, label='Residual Std')
    
    # Galsim.deconv standard deviation
    im5 = axes[4].imshow(galsim_deconv_residual_std, cmap='viridis', origin='lower',
                           vmin=0, vmax=global_std_vmax)
    axes[4].set_title('Galsim.deconv Residual\nStandard Deviation')
    axes[4].axis('off')
    plt.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04, label='Residual Std')
    
    # Standard deviation difference
    im6 = axes[5].imshow(std_difference, cmap='RdBu_r', origin='lower',
                           vmin=-std_diff_vmax, vmax=std_diff_vmax)
    axes[5].set_title('Std Difference\n(Neural - Galsim.deconv )')
    axes[5].axis('off')
    plt.colorbar(im6, ax=axes[5], fraction=0.046, pad=0.04, label='Std Difference')
    
    plt.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Spatial residuals plot saved to: {path}")
    else:
        plt.show()
    
    plt.close()