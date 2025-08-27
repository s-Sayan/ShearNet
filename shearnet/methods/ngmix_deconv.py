"""Metacalibration-based PSF deconvolution methods for ShearNet - UPDATED VERSION.

This implementation completely replaces the old parametric fitting approach
with direct metacalibration-based PSF deconvolution using ngmix's MetacalDilatePSF.
"""

import numpy as np
import jax.numpy as jnp
import ngmix
import galsim
from ngmix.metacal.metacal import MetacalDilatePSF
from typing import Union, Optional, Dict, Any
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def create_ngmix_observation(galaxy_image: np.ndarray, psf_image: np.ndarray, 
                           noise_var: float = 1e-10, pixel_scale: float = 0.141) -> ngmix.Observation:
    """
    Create an ngmix Observation object from galaxy and PSF images.
    
    Args:
        galaxy_image: Observed galaxy image
        psf_image: PSF image  
        noise_var: Noise variance for the observation
        pixel_scale: Pixel scale in arcsec/pixel
        
    Returns:
        ngmix.Observation object
    """
    # Ensure 2D arrays and convert to float64 (NGmix requirement)
    if galaxy_image.ndim == 3:
        galaxy_image = galaxy_image.squeeze()
    if psf_image.ndim == 3:
        psf_image = psf_image.squeeze()
    
    # Convert to float64 - NGmix is picky about data types
    galaxy_image = np.ascontiguousarray(galaxy_image, dtype=np.float64)
    psf_image = np.ascontiguousarray(psf_image, dtype=np.float64)
    
    # Validate images
    if np.any(~np.isfinite(galaxy_image)) or np.any(~np.isfinite(psf_image)):
        raise ValueError("Images contain NaN or inf values")
    
    # Ensure minimum noise variance to avoid numerical issues
    noise_var = max(noise_var, 1e-8)
    
    # Create weight maps (inverse variance)
    weight = np.ones_like(galaxy_image, dtype=np.float64) / noise_var
    psf_weight = np.ones_like(psf_image, dtype=np.float64) / (noise_var * 0.1)
    
    # Ensure weights are finite and positive
    weight = np.clip(weight, 1e-10, 1e10)
    psf_weight = np.clip(psf_weight, 1e-10, 1e10)
    
    # Create jacobians
    galaxy_shape = galaxy_image.shape
    psf_shape = psf_image.shape
    
    # Center coordinates (0-indexed, so center is (n-1)/2)
    galaxy_cen_row = (galaxy_shape[0] - 1) / 2.0
    galaxy_cen_col = (galaxy_shape[1] - 1) / 2.0
    psf_cen_row = (psf_shape[0] - 1) / 2.0
    psf_cen_col = (psf_shape[1] - 1) / 2.0
    
    # Create jacobians
    galaxy_jacobian = ngmix.jacobian.DiagonalJacobian(
        scale=pixel_scale, 
        row=galaxy_cen_row, 
        col=galaxy_cen_col
    )
    psf_jacobian = ngmix.jacobian.DiagonalJacobian(
        scale=pixel_scale, 
        row=psf_cen_row, 
        col=psf_cen_col
    )
    
    # Create PSF observation first
    psf_obs = ngmix.Observation(
        image=psf_image,
        weight=psf_weight,
        jacobian=psf_jacobian
    )
    
    # Validate PSF observation
    if not hasattr(psf_obs, 'image') or psf_obs.image is None:
        raise ValueError("Failed to create valid PSF observation")
    
    # Create galaxy observation with PSF
    galaxy_obs = ngmix.Observation(
        image=galaxy_image,
        weight=weight,
        jacobian=galaxy_jacobian,
        psf=psf_obs
    )
    
    # Validate galaxy observation
    if not hasattr(galaxy_obs, 'image') or galaxy_obs.image is None:
        raise ValueError("Failed to create valid galaxy observation")
    if not hasattr(galaxy_obs, 'psf') or galaxy_obs.psf is None:
        raise ValueError("Failed to attach PSF to galaxy observation")
    
    return galaxy_obs

def metacal_deconvolve_single(args):
    """Single galaxy deconvolution using metacalibration approach."""
    galaxy_img, psf_img, noise_var, pixel_scale, seed = args
    
    try:
        # Ensure arrays are contiguous and the right type
        galaxy_img = np.ascontiguousarray(galaxy_img, dtype=np.float64)
        psf_img = np.ascontiguousarray(psf_img, dtype=np.float64)
        
        # Create ngmix observation
        obs = create_ngmix_observation(galaxy_img, psf_img, noise_var, pixel_scale)
        
        # Use MetacalDilatePSF for deconvolution
        mc = MetacalDilatePSF(obs)
        
        # The key fix: image_int_nopsf is a GalSim Convolution object that needs to be drawn
        if hasattr(mc, 'image_int_nopsf'):
            # Get image size from input galaxy image
            img_height, img_width = galaxy_img.shape
            # Draw the GalSim object to get pixel values
            drawn_image = mc.image_int_nopsf.drawImage(nx=img_width, ny=img_height, scale=pixel_scale)
            deconv_img = drawn_image.array
        else:
            print("Warning: MetacalDilatePSF has no image_int_nopsf attribute")
            return np.zeros_like(galaxy_img)
        
        return deconv_img
        
    except Exception as e:
        print(f"Metacal deconvolution failed for single galaxy: {e}")
        return np.zeros_like(galaxy_img)
        
    except Exception as e:
        # Return zeros if deconvolution failed
        # print(f"Metacal deconvolution failed for single galaxy: {e}")
        return np.zeros_like(galaxy_img)


def metacal_deconvolve(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray,
                      noise_var: float = 1e-8, pixel_scale: float = 0.141, 
                      n_jobs: int = None, seed: int = 42) -> jnp.ndarray:
    """
    Perform PSF deconvolution using metacalibration approach.
    
    This method uses ngmix's MetacalDilatePSF to extract the deconvolved galaxy
    images (image_int_nopsf) which have PSF and pixel effects removed.
    
    Args:
        galaxy_images: Observed galaxy images [N, H, W] or [N, H, W, 1]
        psf_images: PSF images [N, H, W] or [N, H, W, 1]
        noise_var: Noise variance estimate
        pixel_scale: Pixel scale in arcsec/pixel
        n_jobs: Number of parallel processes (None for auto)
        seed: Random seed base
        
    Returns:
        Deconvolved galaxy images [N, H, W]
    """
    # Convert to numpy and ensure 3D
    galaxy_images = np.array(galaxy_images)
    psf_images = np.array(psf_images)
    
    if galaxy_images.ndim == 4:
        galaxy_images = galaxy_images.squeeze(-1)
    if psf_images.ndim == 4:
        psf_images = psf_images.squeeze(-1)
    
    # Ensure minimum noise variance
    noise_var = max(noise_var, 1e-8)
    
    n_galaxies = len(galaxy_images)
    
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), n_galaxies)
    
    print(f"Metacalibration deconvolution: {n_galaxies} galaxies, {n_jobs} processes")
    
    # Prepare arguments for multiprocessing
    args_list = [
        (galaxy_images[i], psf_images[i], noise_var, pixel_scale, seed + i)
        for i in range(n_galaxies)
    ]
    
    deconvolved_images = []
    
    if n_jobs == 1:
        # Single process
        for args in args_list:
            result = metacal_deconvolve_single(args)
            deconvolved_images.append(result)
    else:
        # Multiprocessing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(metacal_deconvolve_single, args): i 
                      for i, args in enumerate(args_list)}
            
            # Collect results in order
            results = [None] * n_galaxies
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Error processing galaxy {idx}: {e}")
                    results[idx] = np.zeros_like(galaxy_images[idx])
            
            deconvolved_images = results
    
    return jnp.array(deconvolved_images)