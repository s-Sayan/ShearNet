"""Metrics and evaluation functions for PSF deconvolution."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any
from ..methods.fft_deconv import fourier_deconvolve

# ANSI color codes for pretty printing
BOLD = '\033[1m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'


def calculate_psnr(target_images: jnp.ndarray, predicted_images: jnp.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        target_images: Ground truth images
        predicted_images: Predicted/reconstructed images
        
    Returns:
        PSNR value in dB
    """
    if target_images.ndim != predicted_images.ndim:
        if target_images.ndim == 3:
            target_images = target_images[..., None]
        elif predicted_images.ndim == 3:
            predicted_images = predicted_images[..., None]

    mse = jnp.mean((target_images - predicted_images) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel_value = jnp.max(target_images)
    psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
    return float(psnr)


def calculate_ssim(target_images: jnp.ndarray, predicted_images: jnp.ndarray, 
                  window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Calculate Structural Similarity Index (SSIM) - simplified version.
    
    Args:
        target_images: Ground truth images
        predicted_images: Predicted/reconstructed images
        window_size: Size of the sliding window
        k1, k2: SSIM parameters
        
    Returns:
        SSIM value (0 to 1, higher is better)
    """
    if target_images.ndim != predicted_images.ndim:
        if target_images.ndim == 3:
            target_images = target_images[..., None]
        elif predicted_images.ndim == 3:
            predicted_images = predicted_images[..., None]

    # Convert to float for calculations
    target = target_images.astype(jnp.float32)
    predicted = predicted_images.astype(jnp.float32)
    
    # Calculate means
    mu1 = jnp.mean(target)
    mu2 = jnp.mean(predicted)
    
    # Calculate variances and covariance
    var1 = jnp.var(target)
    var2 = jnp.var(predicted)
    cov12 = jnp.mean((target - mu1) * (predicted - mu2))
    
    # Dynamic range (assume normalized images)
    L = 1.0
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    # SSIM formula
    ssim = ((2 * mu1 * mu2 + c1) * (2 * cov12 + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    
    return float(ssim)


def calculate_lpips_approx(target_images: jnp.ndarray, predicted_images: jnp.ndarray) -> float:
    """
    Approximate LPIPS using simple feature-based comparison.
    This is a simplified version - real LPIPS would use a pre-trained network.
    
    Args:
        target_images: Ground truth images
        predicted_images: Predicted/reconstructed images
        
    Returns:
        Approximate perceptual distance (lower is better)
    """
    if target_images.ndim != predicted_images.ndim:
        if target_images.ndim == 3:
            target_images = target_images[..., None]
        elif predicted_images.ndim == 3:
            predicted_images = predicted_images[..., None]

    # Simple gradient-based features as proxy for perceptual features
    def get_gradients(images):
        grad_x = jnp.diff(images, axis=-1, prepend=0)
        grad_y = jnp.diff(images, axis=-2, prepend=0)
        return grad_x, grad_y
    
    target_grad_x, target_grad_y = get_gradients(target_images)
    pred_grad_x, pred_grad_y = get_gradients(predicted_images)
    
    # L2 distance in gradient space as proxy for perceptual distance
    grad_dist = jnp.mean((target_grad_x - pred_grad_x)**2 + (target_grad_y - pred_grad_y)**2)
    
    return float(grad_dist)

@jax.jit
def predict_batch(state, batch_galaxy, batch_psf):
    return state.apply_fn(state.params, batch_galaxy, batch_psf, training=False)

def eval_deconv_model(state, galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                     target_images: jnp.ndarray, batch_size: int = 32) -> Dict[str, Any]:
    """
    Evaluate a trained deconvolution model with comprehensive metrics.
    
    Args:
        state: Trained model state
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        target_images: Ground truth clean images
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary containing all evaluation metrics and predictions
    """

    start_time = time.time()
    
    # Generate predictions
    predictions = []
    for i in range(0, len(galaxy_images), batch_size):
        batch_galaxy = galaxy_images[i:i + batch_size]
        batch_psf = psf_images[i:i + batch_size]
        
        # Generate predictions (training=False for inference mode)
        batch_preds = predict_batch(state, batch_galaxy, batch_psf)
        predictions.append(batch_preds)
    
    predictions = jnp.concatenate(predictions, axis=0)

    if target_images.ndim == 3:
        target_images = target_images[..., None]  # Add channel dimension
    
    # Calculate metrics
    mse = float(jnp.mean((predictions - target_images) ** 2))
    mae = float(jnp.mean(jnp.abs(predictions - target_images)))
    psnr = calculate_psnr(target_images, predictions)
    ssim = calculate_ssim(target_images, predictions)
    lpips_approx = calculate_lpips_approx(target_images, predictions)
    
    # Calculate bias
    bias = float(jnp.mean(predictions - target_images))
    
    # Calculate normalized metrics
    target_std = float(jnp.std(target_images))
    normalized_mse = mse / (target_std ** 2) if target_std > 0 else mse
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{BOLD}=== Neural Deconvolution Results ==={END}")
    print(f"Mean Squared Error (MSE): {BOLD}{YELLOW}{mse:.6e}{END}")
    print(f"Mean Absolute Error (MAE): {BOLD}{YELLOW}{mae:.6e}{END}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {BOLD}{CYAN}{psnr:.2f} dB{END}")
    print(f"Structural Similarity Index (SSIM): {BOLD}{CYAN}{ssim:.4f}{END}")
    print(f"Approximate Perceptual Distance: {BOLD}{CYAN}{lpips_approx:.6e}{END}")
    print(f"Bias: {BOLD}{bias:+.6e}{END}")
    print(f"Normalized MSE: {BOLD}{normalized_mse:.6e}{END}")
    print(f"Evaluation time: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    
    return {
        'method': 'Neural Network',
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'ssim': ssim,
        'lpips_approx': lpips_approx,
        'bias': bias,
        'normalized_mse': normalized_mse,
        'predictions': predictions,
        'time_taken': total_time
    }


def eval_fft_deconv(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                   target_images: jnp.ndarray, epsilon: float = 1e-3) -> Dict[str, Any]:
    """
    Evaluate FFT-based deconvolution method.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        target_images: Ground truth clean images
        epsilon: Regularization parameter for FFT deconvolution
        
    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    start_time = time.time()
    
    # Generate FFT deconvolution predictions
    predictions = fourier_deconvolve(galaxy_images, psf_images, epsilon)
    
    # Remove extra dimension if present
    if predictions.ndim == 4 and predictions.shape[-1] == 1:
        predictions = predictions[..., 0]

    if target_images.ndim == 3:
        target_images = target_images[..., None]
    
    # Calculate metrics
    mse = float(jnp.mean((predictions - target_images) ** 2))
    mae = float(jnp.mean(jnp.abs(predictions - target_images)))
    psnr = calculate_psnr(target_images, predictions)
    ssim = calculate_ssim(target_images, predictions)
    lpips_approx = calculate_lpips_approx(target_images, predictions)
    
    # Calculate bias
    bias = float(jnp.mean(predictions - target_images))
    
    # Calculate normalized metrics
    target_std = float(jnp.std(target_images))
    normalized_mse = mse / (target_std ** 2) if target_std > 0 else mse
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{BOLD}=== FFT Deconvolution Results ==={END}")
    print(f"Regularization parameter (ε): {BOLD}{epsilon:.1e}{END}")
    print(f"Mean Squared Error (MSE): {BOLD}{YELLOW}{mse:.6e}{END}")
    print(f"Mean Absolute Error (MAE): {BOLD}{YELLOW}{mae:.6e}{END}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {BOLD}{CYAN}{psnr:.2f} dB{END}")
    print(f"Structural Similarity Index (SSIM): {BOLD}{CYAN}{ssim:.4f}{END}")
    print(f"Approximate Perceptual Distance: {BOLD}{CYAN}{lpips_approx:.6e}{END}")
    print(f"Bias: {BOLD}{bias:+.6e}{END}")
    print(f"Normalized MSE: {BOLD}{normalized_mse:.6e}{END}")
    print(f"Evaluation time: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    
    return {
        'method': 'FFT Deconvolution',
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'ssim': ssim,
        'lpips_approx': lpips_approx,
        'bias': bias,
        'normalized_mse': normalized_mse,
        'predictions': predictions,
        'time_taken': total_time,
        'epsilon': epsilon
    }


def compare_deconv_methods(neural_results: Dict[str, Any], fft_results: Dict[str, Any]) -> None:
    """
    Compare results from different deconvolution methods.
    
    Args:
        neural_results: Results from neural network deconvolution
        fft_results: Results from FFT deconvolution
    """
    print(f"\n{BOLD}Method Comparison:{END}")
    print(f"{'Metric':<25} {'Neural':<15} {'FFT':<15} {'Winner':<10}")
    print("-" * 70)
    
    metrics = ['mse', 'mae', 'psnr', 'ssim', 'normalized_mse', 'time_taken']
    higher_better = {'psnr': True, 'ssim': True}
    
    for metric in metrics:
        if metric in neural_results and metric in fft_results:
            neural_val = neural_results[metric]
            fft_val = fft_results[metric]
            
            # Determine winner (lower is better for most metrics, except PSNR and SSIM)
            if metric in higher_better:
                winner = "Neural" if neural_val > fft_val else "FFT"
                winner_color = GREEN if winner == "Neural" else RED
            else:
                winner = "Neural" if neural_val < fft_val else "FFT"
                winner_color = GREEN if winner == "Neural" else RED
            
            # Format values
            if metric == 'time_taken':
                neural_str = f"{neural_val:.2f}s"
                fft_str = f"{fft_val:.2f}s"
            elif metric in ['psnr']:
                neural_str = f"{neural_val:.2f}"
                fft_str = f"{fft_val:.2f}"
            elif metric in ['ssim']:
                neural_str = f"{neural_val:.4f}"
                fft_str = f"{fft_val:.4f}"
            else:
                neural_str = f"{neural_val:.3e}"
                fft_str = f"{fft_val:.3e}"
            
            print(f"{metric.upper():<25} {neural_str:<15} {fft_str:<15} {winner_color}{winner}{END}")
    
    # Overall summary
    print("\n" + "="*70)
    
    # Count wins for each method
    neural_wins = 0
    fft_wins = 0
    
    for metric in ['mse', 'mae', 'normalized_mse']:  # Lower is better
        if neural_results[metric] < fft_results[metric]:
            neural_wins += 1
        else:
            fft_wins += 1
    
    for metric in ['psnr', 'ssim']:  # Higher is better
        if neural_results[metric] > fft_results[metric]:
            neural_wins += 1
        else:
            fft_wins += 1
    
    if neural_wins > fft_wins:
        print(f"{BOLD}{GREEN}Overall Winner: Neural Network Deconvolution{END}")
        print(f"Neural network wins {neural_wins} out of {neural_wins + fft_wins} key metrics")
    elif fft_wins > neural_wins:
        print(f"{BOLD}{RED}Overall Winner: FFT Deconvolution{END}")
        print(f"FFT method wins {fft_wins} out of {neural_wins + fft_wins} key metrics")
    else:
        print(f"{BOLD}{YELLOW}Result: Tie between methods{END}")
        print(f"Each method wins {neural_wins} out of {neural_wins + fft_wins} key metrics")


def calculate_improvement_metrics(baseline_results: Dict[str, Any], 
                                improved_results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate improvement percentages between two methods.
    
    Args:
        baseline_results: Results from baseline method
        improved_results: Results from improved method
        
    Returns:
        Dictionary of improvement percentages
    """
    improvements = {}
    
    # Metrics where lower is better
    for metric in ['mse', 'mae', 'normalized_mse']:
        if metric in baseline_results and metric in improved_results:
            baseline_val = baseline_results[metric]
            improved_val = improved_results[metric]
            if baseline_val > 0:
                improvement_pct = ((baseline_val - improved_val) / baseline_val) * 100
                improvements[metric] = improvement_pct
    
    # Metrics where higher is better
    for metric in ['psnr', 'ssim']:
        if metric in baseline_results and metric in improved_results:
            baseline_val = baseline_results[metric]
            improved_val = improved_results[metric]
            if baseline_val > 0:
                improvement_pct = ((improved_val - baseline_val) / baseline_val) * 100
                improvements[metric] = improvement_pct
    
    return improvements