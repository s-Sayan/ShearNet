"""Metrics and evaluation functions for PSF deconvolution with Metacalibration comparison."""

import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any
from ..methods.ngmix_deconv import metacal_deconvolve

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
    # Ensure same shape
    if target_images.ndim != predicted_images.ndim:
        if target_images.ndim == 4 and predicted_images.ndim == 3:
            target_images = target_images.squeeze(-1)
        elif target_images.ndim == 3 and predicted_images.ndim == 4:
            predicted_images = predicted_images.squeeze(-1)

    mse = jnp.mean((target_images - predicted_images) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel_value = jnp.max(target_images)
    psnr = 20 * jnp.log10(max_pixel_value / jnp.sqrt(mse))
    return float(psnr)


def calculate_ssim(target_images: jnp.ndarray, predicted_images: jnp.ndarray, 
                  window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Calculate Structural Similarity Index (SSIM) - per-image then averaged.
    
    Args:
        target_images: Ground truth images
        predicted_images: Predicted/reconstructed images
        window_size: Size of the sliding window
        k1, k2: SSIM parameters
        
    Returns:
        SSIM value (0 to 1, higher is better)
    """
    # Ensure same shape
    if target_images.ndim != predicted_images.ndim:
        if target_images.ndim == 4 and predicted_images.ndim == 3:
            target_images = target_images.squeeze(-1)
        elif target_images.ndim == 3 and predicted_images.ndim == 4:
            predicted_images = predicted_images.squeeze(-1)

    # Convert to float for calculations
    target = target_images.astype(jnp.float32)
    predicted = predicted_images.astype(jnp.float32)
    
    # Calculate SSIM per image, then average (more accurate than global SSIM)
    ssim_values = []
    
    for i in range(target.shape[0]):
        img1 = target[i]
        img2 = predicted[i]
        
        # Calculate means for this image pair
        mu1 = jnp.mean(img1)
        mu2 = jnp.mean(img2)
        
        # Calculate variances and covariance for this image pair
        var1 = jnp.var(img1)
        var2 = jnp.var(img2)
        cov12 = jnp.mean((img1 - mu1) * (img2 - mu2))
        
        # Dynamic range - use actual range of target image
        data_range = jnp.max(img1) - jnp.min(img1)
        if data_range == 0:
            data_range = 1.0
            
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        
        # SSIM formula for this image pair
        ssim_single = ((2 * mu1 * mu2 + c1) * (2 * cov12 + c2)) / \
                     ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        
        ssim_values.append(ssim_single)
    
    # Average SSIM across all images
    return float(jnp.mean(jnp.array(ssim_values)))


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
    # Ensure same shape
    if target_images.ndim != predicted_images.ndim:
        if target_images.ndim == 4 and predicted_images.ndim == 3:
            target_images = target_images.squeeze(-1)
        elif target_images.ndim == 3 and predicted_images.ndim == 4:
            predicted_images = predicted_images.squeeze(-1)

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
def _eval_batch_jit(state, galaxy_batch, psf_batch):
    """JIT-compiled evaluation batch function."""
    return state.apply_fn(state.params, galaxy_batch, psf_batch, training=False)

def eval_deconv_model(state, galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                     target_images: jnp.ndarray, batch_size: int = 64) -> Dict[str, Any]:
    """Evaluate a trained deconvolution model with comprehensive metrics."""
    
    # Pre-compile the function
    print("Compiling evaluation function...")
    sample_galaxy = galaxy_images[:1]
    sample_psf = psf_images[:1]
    _ = _eval_batch_jit(state, sample_galaxy, sample_psf)  # Trigger compilation
    print("Compilation complete. Running evaluation...")

    start_time = time.time()
    
    # Generate predictions
    predictions = []
    for i in range(0, len(galaxy_images), batch_size):
        batch_galaxy = galaxy_images[i:i + batch_size]
        batch_psf = psf_images[i:i + batch_size]
        
        # Generate predictions (training=False for inference mode)
        batch_preds = _eval_batch_jit(state, batch_galaxy, batch_psf)
        predictions.append(batch_preds)
    
    predictions = jnp.concatenate(predictions, axis=0)

    # Ensure same dimensionality for consistent metrics calculation
    if target_images.ndim == 3 and predictions.ndim == 4:
        target_images = target_images[..., None]  # Add channel dimension
    elif target_images.ndim == 4 and predictions.ndim == 3:
        predictions = predictions[..., None]  # Add channel dimension
    
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
    
    # SANITY CHECK: Are these metrics realistic?
    print(f"\n{BOLD}=== SANITY CHECKS ==={END}")
    if psnr > 50:
        print(f"{RED}WARNING: PSNR > 50 dB is unusually high. Check for data leakage or model copying input.{END}")
    if ssim > 0.99:
        print(f"{RED}WARNING: SSIM > 0.99 is unusually high. Model may be learning to copy rather than deconvolve.{END}")
    if mse < 1e-6:
        print(f"{RED}WARNING: MSE < 1e-6 is unusually low. Verify model is actually deconvolving.{END}")
        
    # Check if predictions are very similar to targets (possible copying)
    correlation = float(jnp.corrcoef(predictions.flatten(), target_images.flatten())[0, 1])
    print(f"Prediction-Target Correlation: {correlation:.6f}")
    if correlation > 0.999:
        print(f"{RED}WARNING: Correlation > 0.999 suggests model may be copying inputs rather than deconvolving.{END}")
    
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
        'time_taken': total_time,
        'correlation': correlation
    }


def eval_ngmix_deconv(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                     target_images: jnp.ndarray, model: str = 'exp', 
                     **kwargs) -> Dict[str, Any]:
    """
    Evaluate Metacalibration-based deconvolution method.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        target_images: Ground truth clean images
        model: Model type ('exp', 'gauss', 'dev') - all use metacal now
        **kwargs: Additional arguments for metacalibration deconvolution
        
    Returns:
        Dictionary containing evaluation metrics and predictions
    """
    start_time = time.time()
    
    # Choose the appropriate deconvolution function (all use metacal now)
    deconv_func = metacal_deconvolve
    model_name = 'Metacal'
    
    # FORCE single-threaded execution to avoid JAX multiprocessing issues
    kwargs_copy = kwargs.copy()
    kwargs_copy['n_jobs'] = 1  # This should override any multiprocessing
    
    print(f"Original target_images shape: {target_images.shape}")
    print(f"Target images - min: {jnp.min(target_images):.2e}, max: {jnp.max(target_images):.2e}, mean: {jnp.mean(target_images):.2e}")
    
    # Generate metacalibration deconvolution predictions
    predictions = deconv_func(galaxy_images, psf_images, **kwargs_copy)
    
    print(f"Raw predictions shape: {predictions.shape}")
    print(f"Raw predictions - min: {jnp.min(predictions):.2e}, max: {jnp.max(predictions):.2e}, mean: {jnp.mean(predictions):.2e}")
    
    # Check if predictions are all zeros or near-zeros
    non_zero_count = jnp.sum(jnp.abs(predictions) > 1e-10)
    total_pixels = predictions.size
    print(f"Non-zero pixels in predictions: {non_zero_count} / {total_pixels} ({100*non_zero_count/total_pixels:.1f}%)")
    
    # Make working copies to avoid modifying the original arrays
    pred_work = jnp.array(predictions)
    target_work = jnp.array(target_images)
    
    # Aggressively squeeze all singleton dimensions
    pred_work = jnp.squeeze(pred_work)
    target_work = jnp.squeeze(target_work)
    
    print(f"After squeezing - Predictions: {pred_work.shape}, Targets: {target_work.shape}")
    
    # Both should now be 3D (N, H, W) - if not, force it
    if pred_work.ndim == 4:
        pred_work = pred_work[:, :, :, 0]  # Take first channel
    if target_work.ndim == 4:
        target_work = target_work[:, :, :, 0]  # Take first channel
        
    print(f"Final shapes - Predictions: {pred_work.shape}, Targets: {target_work.shape}")
    
    # Verify shapes match before calculation
    if pred_work.shape != target_work.shape:
        raise ValueError(f"Shape mismatch after processing: predictions {pred_work.shape} vs targets {target_work.shape}")
    
    # Calculate metrics - both arrays should now have same shape
    mse = float(jnp.mean((pred_work - target_work) ** 2))
    mae = float(jnp.mean(jnp.abs(pred_work - target_work)))
    psnr = calculate_psnr(target_work, pred_work)
    ssim = calculate_ssim(target_work, pred_work)
    lpips_approx = calculate_lpips_approx(target_work, pred_work)
    
    # Calculate bias
    bias = float(jnp.mean(pred_work - target_work))
    
    # Calculate normalized metrics
    target_std = float(jnp.std(target_work))
    normalized_mse = mse / (target_std ** 2) if target_std > 0 else mse
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{BOLD}=== Metacalibration {model_name} Deconvolution ==={END}")
    print(f"Evaluation Time: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    print(f"Mean Squared Error (MSE): {BOLD}{YELLOW}{mse:.6e}{END}")
    print(f"Mean Absolute Error (MAE): {BOLD}{YELLOW}{mae:.6e}{END}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {BOLD}{CYAN}{psnr:.2f} dB{END}")
    print(f"Bias: {BOLD}{bias:+.6e}{END}")
    
    return {
        'method': f'Metacal {model_name}',
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'ssim': ssim,
        'lpips_approx': lpips_approx,
        'bias': bias,
        'normalized_mse': normalized_mse,
        'predictions': pred_work,  # Return the processed predictions
        'time_taken': total_time,
        'model': model
    }


def eval_all_ngmix_methods(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                          target_images: jnp.ndarray, **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all Metacalibration deconvolution methods.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        target_images: Ground truth clean images
        **kwargs: Additional arguments for metacalibration methods
        
    Returns:
        Dictionary with results from all metacalibration methods
    """
    models = ['exp', 'gauss', 'dev']
    results = {}
    
    for model in models:
        try:
            print(f"\nEvaluating Metacalibration {model} variant...")
            results[model] = eval_ngmix_deconv(
                galaxy_images, psf_images, target_images, model=model, **kwargs
            )
        except Exception as e:
            print(f"Error evaluating Metacalibration {model}: {e}")
            results[model] = {'error': str(e)}
    
    return results


def compare_deconv_methods(neural_results: Dict[str, Any], metacal_results: Dict[str, Any]) -> None:
    """
    Compare results from neural network and Metacalibration deconvolution methods.
    
    Args:
        neural_results: Results from neural network deconvolution
        metacal_results: Results from Metacalibration deconvolution
    """
    print(f"\n{BOLD}Method Comparison:{END}")
    print(f"{'Metric':<25} {'Neural':<15} {'MetaCal':<15} {'Winner':<10}")
    print("-" * 70)
    
    metrics = ['mse', 'mae', 'psnr', 'ssim', 'normalized_mse', 'time_taken']
    higher_better = {'psnr': True, 'ssim': True}
    
    for metric in metrics:
        if metric in neural_results and metric in metacal_results:
            neural_val = neural_results[metric]
            metacal_val = metacal_results[metric]
            
            # Determine winner (lower is better for most metrics, except PSNR and SSIM)
            if metric in higher_better:
                winner = "Neural" if neural_val > metacal_val else "MetaCal"
                winner_color = GREEN if winner == "Neural" else RED
            else:
                winner = "Neural" if neural_val < metacal_val else "MetaCal"
                winner_color = GREEN if winner == "Neural" else RED
            
            # Format values
            if metric == 'time_taken':
                neural_str = f"{neural_val:.2f}s"
                metacal_str = f"{metacal_val:.2f}s"
            elif metric in ['psnr']:
                neural_str = f"{neural_val:.2f}"
                metacal_str = f"{metacal_val:.2f}"
            elif metric in ['ssim']:
                neural_str = f"{neural_val:.4f}"
                metacal_str = f"{metacal_val:.4f}"
            else:
                neural_str = f"{neural_val:.3e}"
                metacal_str = f"{metacal_val:.3e}"
            
            print(f"{metric.upper():<25} {neural_str:<15} {metacal_str:<15} {winner_color}{winner}{END}")
    
    # Overall summary
    print("\n" + "="*70)
    
    # Count wins for each method
    neural_wins = 0
    metacal_wins = 0
    for metric in ['mse', 'mae', 'normalized_mse']:  # Lower is better
        if neural_results[metric] < metacal_results[metric]:
            neural_wins += 1
        else:
            metacal_wins += 1
    
    for metric in ['psnr', 'ssim']:  # Higher is better
        if neural_results[metric] > metacal_results[metric]:
            neural_wins += 1
        else:
            metacal_wins += 1
    
    if neural_wins > metacal_wins:
        print(f"{BOLD}{GREEN}Overall Winner: Neural Network Deconvolution{END}")
        print(f"Neural network wins {neural_wins} out of {neural_wins + metacal_wins} key metrics")
    elif metacal_wins > neural_wins:
        print(f"{BOLD}{RED}Overall Winner: Metacalibration Deconvolution{END}")
        print(f"Metacalibration method wins {metacal_wins} out of {neural_wins + metacal_wins} key metrics")
    else:
        print(f"{BOLD}{YELLOW}Result: Tie between methods{END}")
        print(f"Each method wins {neural_wins} out of {neural_wins + metacal_wins} key metrics")


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


def evaluate_metacal_deconv_comprehensive(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                                        target_images: jnp.ndarray) -> Dict[str, Any]:
    """
    Comprehensive evaluation of all metacalibration deconvolution variants.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        target_images: Ground truth clean images
        
    Returns:
        Dictionary with comprehensive comparison results
    """
    print(f"\n{BOLD}=== Comprehensive Metacalibration Deconvolution Evaluation ==={END}")
    
    # Evaluate all variants
    results = eval_all_ngmix_methods(galaxy_images, psf_images, target_images)
    
    # Find best performing variant
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print(f"{RED}No metacalibration methods succeeded{END}")
        return results
    
    # Sort by MSE (lower is better)
    best_variant = min(valid_results.keys(), key=lambda k: valid_results[k]['mse'])
    best_result = valid_results[best_variant]
    
    print(f"\n{BOLD}=== Summary ==={END}")
    print(f"Best performing variant: {BOLD}{GREEN}Metacalibration {best_variant.upper()}{END}")
    print(f"Best MSE: {BOLD}{best_result['mse']:.3e}{END}")
    print(f"Best PSNR: {BOLD}{best_result['psnr']:.2f} dB{END}")
    
    # Performance comparison between variants
    print(f"\n{BOLD}=== Variant Comparison ==={END}")
    print(f"{'Variant':<15} {'MSE':<12} {'PSNR (dB)':<10} {'Time (s)':<10}")
    print("-" * 50)
    
    for variant, result in valid_results.items():
        if 'error' not in result:
            print(f"{variant.upper():<15} {result['mse']:<12.3e} {result['psnr']:<10.1f} {result['time_taken']:<10.1f}")
    
    return {
        'all_results': results,
        'best_variant': best_variant,
        'best_result': best_result,
        'summary': {
            'num_successful': len(valid_results),
            'num_failed': len(results) - len(valid_results),
            'best_mse': best_result['mse'],
            'best_psnr': best_result['psnr']
        }
    }