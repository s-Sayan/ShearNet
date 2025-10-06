"""Plotting utilities for PSF deconvolution evaluation with NGmix comparison."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import jax.numpy as jnp
from typing import Optional, Dict

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

def plot_spatial_residuals(target_images: jnp.ndarray, 
                          predictions_dict: Dict[str, jnp.ndarray],
                          path: Optional[str] = None,
                          title: str = "Spatial Deconvolution Residuals"):
    """
    Plot spatial residual heat maps showing systematic biases across image coordinates.
    
    This shows WHERE in the image each deconvolution method systematically fails,
    averaged across all test images. Zero = perfect deconvolution at that pixel.
    
    Args:
        target_images: Ground truth clean images [N, H, W] or [N, H, W, 1]
        predictions_dict: Dictionary mapping method names to predictions [N, H, W] or [N, H, W, 1]
        path: Path to save the plot
        title: Plot title
    """
    
    # Ensure consistent shapes - squeeze out channel dimensions
    if target_images.ndim == 4:
        target_images = target_images.squeeze(-1)
    
    # Clean up all predictions
    for method_name in predictions_dict:
        if predictions_dict[method_name].ndim == 4:
            predictions_dict[method_name] = predictions_dict[method_name].squeeze(-1)
    
    print(f"Computing spatial residuals for {len(target_images)} images across {len(predictions_dict)} methods...")
    
    # Calculate spatial bias for each method
    spatial_biases = {}
    residual_stds = {}
    
    for method_name, predictions in predictions_dict.items():
        spatial_biases[method_name] = jnp.mean(predictions - target_images, axis=0)
        residual_stds[method_name] = jnp.std(predictions - target_images, axis=0)
    
    # Find global scaling for bias maps
    all_biases = jnp.array([bias for bias in spatial_biases.values()])
    global_bias_vmax = jnp.max(jnp.abs(all_biases))
    
    # Find global scaling for std maps
    all_stds = jnp.array([std for std in residual_stds.values()])
    global_std_vmax = jnp.max(all_stds)
    
    # Determine if we should plot difference (only if exactly 2 methods)
    plot_difference = len(predictions_dict) == 2
    
    # Set up the plot
    n_methods = len(predictions_dict)
    if plot_difference:
        # 2 rows (bias, std) x 3 columns (method1, method2, difference)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    else:
        # 2 rows (bias, std) x n_methods columns
        fig, axes = plt.subplots(2, n_methods, figsize=(6*n_methods, 12))
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        axes = axes.flatten()
    
    method_names = list(predictions_dict.keys())
    
    # Row 1: Bias maps
    for i, method_name in enumerate(method_names):
        ax_idx = i
        im = axes[ax_idx].imshow(
            spatial_biases[method_name], 
            cmap='RdBu_r', 
            origin='lower',
            vmin=-global_bias_vmax, 
            vmax=global_bias_vmax
        )
        axes[ax_idx].set_title(f'{method_name}\nSpatial Bias Map')
        axes[ax_idx].axis('off')
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04, label='Mean Residual')
    
    # If exactly 2 methods, plot difference
    if plot_difference:
        bias_diff = spatial_biases[method_names[0]] - spatial_biases[method_names[1]]
        im = axes[2].imshow(
            bias_diff,
            cmap='RdBu_r',
            origin='lower',
            vmin=-global_bias_vmax,
            vmax=global_bias_vmax
        )
        axes[2].set_title(f'Bias Difference\n({method_names[0]} - {method_names[1]})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Bias Difference')
        
        row2_start = 3
    else:
        row2_start = n_methods
    
    # Row 2: Standard deviation maps
    for i, method_name in enumerate(method_names):
        ax_idx = row2_start + i
        im = axes[ax_idx].imshow(
            residual_stds[method_name],
            cmap='viridis',
            origin='lower',
            vmin=0,
            vmax=global_std_vmax
        )
        axes[ax_idx].set_title(f'{method_name}\nResidual Std Dev')
        axes[ax_idx].axis('off')
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04, label='Residual Std')
    
    # If exactly 2 methods, plot std difference
    if plot_difference:
        std_diff = residual_stds[method_names[0]] - residual_stds[method_names[1]]
        std_diff_vmax = jnp.max(jnp.abs(std_diff))
        im = axes[row2_start + 2].imshow(
            std_diff,
            cmap='RdBu_r',
            origin='lower',
            vmin=-std_diff_vmax,
            vmax=std_diff_vmax
        )
        axes[row2_start + 2].set_title(f'Std Difference\n({method_names[0]} - {method_names[1]})')
        axes[row2_start + 2].axis('off')
        plt.colorbar(im, ax=axes[row2_start + 2], fraction=0.046, pad=0.04, label='Std Difference')
    
    plt.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Spatial residuals plot saved to: {path}")
    else:
        plt.show()
    
    plt.close()