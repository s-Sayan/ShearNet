"""FFT-based deconvolution methods for benchmarking."""

import jax.numpy as jnp
import numpy as np
from typing import Union
import jax.numpy as jnp
from jax import vmap


def fourier_deconvolve(blurred, psf, epsilon=1e-3):
    """
    Perform Fourier deconvolution on a batch of blurred images with a given PSF.
    
    Args:
        blurred: Array of blurred images [batch, H, W]
        psf: Array of PSF images [batch, H, W]
        epsilon: Regularization parameter to avoid division by zero
        
    Returns:
        Deconvolved images in real space
    """
    # Remember original shapes
    original_blurred_had_channel = blurred.ndim == 4
    
    # Ensure proper dimensions
    if blurred.ndim == 3:
        blurred = blurred[:, :, :, None]  # Add channel dimension
    if psf.ndim == 3:
        psf = psf[:, :, :, None]  # Add channel dimension
    
    # Vectorize the operation over batches and channels
    def deconv_single(blurred_single, psf_single):
        # Remove channel dimension for processing
        if blurred_single.ndim == 3 and blurred_single.shape[-1] == 1:
            blurred_single = blurred_single[..., 0]
        if psf_single.ndim == 3 and psf_single.shape[-1] == 1:
            psf_single = psf_single[..., 0]
        
        # Normalize PSF
        psf_normalized = psf_single / jnp.sum(psf_single)
        
        # Critical: Center the PSF by putting peak at (0,0) for FFT
        # Find peak and roll to put it at top-left corner
        peak_pos = jnp.unravel_index(jnp.argmax(psf_normalized), psf_normalized.shape)
        psf_shifted = jnp.roll(psf_normalized, (-peak_pos[0], -peak_pos[1]), axis=(0, 1))
        
        # Compute FFTs
        blurred_fft = jnp.fft.fft2(blurred_single)
        psf_fft = jnp.fft.fft2(psf_shifted)
        
        # More robust regularization: Wiener-like filter
        psf_power = jnp.abs(psf_fft)**2
        max_psf_power = jnp.max(psf_power)
        
        # Adaptive regularization based on PSF strength
        adaptive_epsilon = epsilon * max_psf_power
        
        # Wiener-style deconvolution for stability
        deblurred_fft = (blurred_fft * jnp.conj(psf_fft)) / (psf_power + adaptive_epsilon)
        
        # Inverse FFT
        deblurred = jnp.real(jnp.fft.ifft2(deblurred_fft))
        
        return deblurred
    
    # Apply to all images in batch
    deblurred = vmap(deconv_single, in_axes=(0, 0))(blurred, psf)
    
    # Restore channel dimension if original input had it
    if original_blurred_had_channel:
        deblurred = deblurred[..., None]
    
    return deblurred


def wiener_deconvolve(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray,
                     noise_power: float = 1e-3, signal_power: float = 1.0) -> jnp.ndarray:
    """
    Wiener deconvolution with explicit noise and signal power estimates.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        noise_power: Estimated noise power spectral density
        signal_power: Estimated signal power spectral density
        
    Returns:
        Deconvolved galaxy images
    """
    deconvolved = []
    
    for gal, psf in zip(galaxy_images, psf_images):
        # Handle channel dimension
        if gal.ndim == 3 and gal.shape[-1] == 1:
            gal = gal[..., 0]
        if psf.ndim == 3 and psf.shape[-1] == 1:
            psf = psf[..., 0]
        
        # Normalize PSF
        psf_normalized = psf / jnp.sum(psf)
        
        # Apply same PSF centering fix as FFT method
        peak_pos = jnp.unravel_index(jnp.argmax(psf_normalized), psf_normalized.shape)
        psf_shifted = jnp.roll(psf_normalized, (-peak_pos[0], -peak_pos[1]), axis=(0, 1))
        
        # FFT of both images
        gal_fft = jnp.fft.fft2(gal)
        psf_fft = jnp.fft.fft2(psf_shifted)
        
        # Wiener filter
        psf_power = jnp.abs(psf_fft)**2
        wiener_filter = jnp.conj(psf_fft) / (psf_power + noise_power / signal_power)
        deconv_fft = gal_fft * wiener_filter
        
        # Inverse FFT
        deconv = jnp.fft.ifft2(deconv_fft).real
        
        deconvolved.append(deconv)
    
    return jnp.array(deconvolved)


def richardson_lucy_deconvolve(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray,
                              num_iterations: int = 10, 
                              damping: float = 1.0) -> jnp.ndarray:
    """
    Richardson-Lucy deconvolution algorithm.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        num_iterations: Number of R-L iterations
        damping: Damping factor for stability (1.0 = no damping)
        
    Returns:
        Deconvolved galaxy images
    """
    deconvolved = []
    
    for gal, psf in zip(galaxy_images, psf_images):
        # Handle channel dimension
        if gal.ndim == 3 and gal.shape[-1] == 1:
            gal = gal[..., 0]
        if psf.ndim == 3 and psf.shape[-1] == 1:
            psf = psf[..., 0]
        
        # Normalize PSF
        psf_normalized = psf / jnp.sum(psf)
        
        # Apply same PSF centering fix as FFT method
        peak_pos = jnp.unravel_index(jnp.argmax(psf_normalized), psf_normalized.shape)
        psf_shifted = jnp.roll(psf_normalized, (-peak_pos[0], -peak_pos[1]), axis=(0, 1))
        psf_flipped = jnp.flip(psf_shifted)  # For correlation
        
        # Initialize estimate as observed image
        estimate = jnp.maximum(gal, 1e-10)  # Avoid zeros
        
        for iteration in range(num_iterations):
            # Forward convolution: estimate * PSF
            conv_estimate = jnp.real(jnp.fft.ifft2(
                jnp.fft.fft2(estimate) * jnp.fft.fft2(psf_shifted)
            ))
            
            # Ratio of observed to forward model
            ratio = gal / (conv_estimate + 1e-10)
            
            # Correlation with flipped PSF
            correction = jnp.real(jnp.fft.ifft2(
                jnp.fft.fft2(ratio) * jnp.fft.fft2(psf_flipped)
            ))
            
            # Update estimate with damping
            estimate = estimate * (1 + damping * (correction - 1))
            estimate = jnp.maximum(estimate, 1e-10)  # Keep positive
        
        deconvolved.append(estimate)
    
    return jnp.array(deconvolved)


def lucy_richardson_with_tv_regularization(galaxy_images: jnp.ndarray, 
                                          psf_images: jnp.ndarray,
                                          num_iterations: int = 10,
                                          tv_weight: float = 0.01) -> jnp.ndarray:
    """
    Richardson-Lucy with Total Variation regularization.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images  
        num_iterations: Number of iterations
        tv_weight: Weight for TV regularization term
        
    Returns:
        Deconvolved galaxy images
    """
    def total_variation_grad(image):
        """Compute gradient of total variation term."""
        # Simple TV gradient approximation
        grad_x = jnp.diff(image, axis=1, prepend=image[:, :1])
        grad_y = jnp.diff(image, axis=0, prepend=image[:1, :])
        
        # Divergence approximation (negative gradient of TV)
        div_x = jnp.diff(grad_x, axis=1, append=grad_x[:, -1:])
        div_y = jnp.diff(grad_y, axis=0, append=grad_y[-1:, :])
        
        return -(div_x + div_y)
    
    deconvolved = []
    
    for gal, psf in zip(galaxy_images, psf_images):
        # Handle channel dimension
        if gal.ndim == 3 and gal.shape[-1] == 1:
            gal = gal[..., 0]
        if psf.ndim == 3 and psf.shape[-1] == 1:
            psf = psf[..., 0]
        
        # Normalize PSF
        psf_normalized = psf / jnp.sum(psf)
        
        # Apply same PSF centering fix as FFT method
        peak_pos = jnp.unravel_index(jnp.argmax(psf_normalized), psf_normalized.shape)
        psf_shifted = jnp.roll(psf_normalized, (-peak_pos[0], -peak_pos[1]), axis=(0, 1))
        psf_flipped = jnp.flip(psf_shifted)
        
        # Initialize estimate
        estimate = jnp.maximum(gal, 1e-10)
        
        for iteration in range(num_iterations):
            # Forward convolution
            conv_estimate = jnp.real(jnp.fft.ifft2(
                jnp.fft.fft2(estimate) * jnp.fft.fft2(psf_shifted)
            ))
            
            # Data fidelity term
            ratio = gal / (conv_estimate + 1e-10)
            data_term = jnp.real(jnp.fft.ifft2(
                jnp.fft.fft2(ratio) * jnp.fft.fft2(psf_flipped)
            ))
            
            # Total variation regularization term
            tv_term = total_variation_grad(estimate)
            
            # Combined update
            multiplicative_factor = data_term / (1 + tv_weight * tv_term + 1e-10)
            estimate = estimate * multiplicative_factor
            estimate = jnp.maximum(estimate, 1e-10)
        
        deconvolved.append(estimate)
    
    return jnp.array(deconvolved)


def adaptive_fourier_deconvolve(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray,
                               snr_threshold: float = 10.0) -> jnp.ndarray:
    """
    Adaptive Fourier deconvolution with frequency-dependent regularization.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        snr_threshold: Signal-to-noise ratio threshold for regularization
        
    Returns:
        Deconvolved galaxy images
    """
    deconvolved = []
    
    for gal, psf in zip(galaxy_images, psf_images):
        # Handle channel dimension
        if gal.ndim == 3 and gal.shape[-1] == 1:
            gal = gal[..., 0]
        if psf.ndim == 3 and psf.shape[-1] == 1:
            psf = psf[..., 0]
        
        # Add aggressive padding
        h, w = gal.shape
        pad_h, pad_w = h, w  # Double the size
        gal_padded = jnp.pad(gal, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        psf_padded = jnp.pad(psf, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        # Normalize PSF
        psf_padded = psf_padded / jnp.sum(psf_padded)
        
        # FFT of both images
        gal_fft = jnp.fft.fft2(gal_padded)
        psf_fft = jnp.fft.fft2(jnp.fft.ifftshift(psf_padded))
        
        # Estimate noise power from high-frequency components
        h, w = gal_padded.shape
        noise_region = gal_padded[h//4:3*h//4, w//4:3*w//4]  # Central region
        noise_power = jnp.var(noise_region) * 0.1  # Conservative estimate
        
        # Adaptive regularization based on local SNR
        psf_power = jnp.abs(psf_fft)**2
        signal_power = jnp.abs(gal_fft)**2
        local_snr = signal_power / (noise_power + 1e-10)
        
        # Frequency-dependent epsilon
        epsilon = noise_power / jnp.maximum(local_snr / snr_threshold, 1.0)
        
        # Regularized deconvolution (back to simple division)
        deconv_fft = gal_fft / (psf_fft + epsilon)
        
        # Inverse FFT
        deconv = jnp.fft.ifft2(deconv_fft).real
        
        deconvolved.append(deconv)
    
    return jnp.array(deconvolved)


def evaluate_fft_methods(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                        target_images: jnp.ndarray) -> dict:
    """
    Evaluate multiple FFT-based deconvolution methods.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        target_images: Ground truth clean images
        
    Returns:
        Dictionary with results from different methods
    """
    methods = {
        'Simple Fourier': lambda g, p: fourier_deconvolve(g, p, epsilon=1e-3),
        'Wiener': lambda g, p: wiener_deconvolve(g, p, noise_power=1e-3, signal_power=1.0),
        'Richardson-Lucy (10 iter)': lambda g, p: richardson_lucy_deconvolve(g, p, num_iterations=10),
        'Adaptive Fourier': lambda g, p: adaptive_fourier_deconvolve(g, p, snr_threshold=10.0),
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"Evaluating {method_name}...")
        
        try:
            predictions = method_func(galaxy_images, psf_images)
            
            # Calculate metrics
            mse = float(jnp.mean((predictions - target_images) ** 2))
            psnr = -10 * jnp.log10(mse) if mse > 0 else float('inf')
            
            results[method_name] = {
                'predictions': predictions,
                'mse': mse,
                'psnr': psnr
            }
            
            print(f"  MSE: {mse:.3e}, PSNR: {psnr:.1f} dB")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[method_name] = {'error': str(e)}
    
    return results