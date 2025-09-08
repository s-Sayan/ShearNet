"""Metacalibration-based PSF deconvolution methods for ShearNet

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
from typing import Union, Optional, Dict, Any, List

def _galsim_stuff(img, wcs, xinterp):
    """
    Taken straight from ngmix.metacal.metacal
    """
    def _cached_galsim_stuff(img, wcs_repr, xinterp):
        return _galsim_stuff_impl(np.array(img), eval(wcs_repr), xinterp)
    def _galsim_stuff_impl(img, wcs, xinterp):
        image = galsim.Image(img, wcs=wcs)
        image_int = galsim.InterpolatedImage(image, x_interpolant=xinterp)
        return image, image_int

    return _cached_galsim_stuff(
        tuple(tuple(ii) for ii in img),
        repr(wcs),
        xinterp,
    )


def get_wcs(obs):
    """
    Taken straight from ngmix.metacal.metacal
    """
    return obs.jacobian.get_galsim_wcs()

def get_psf_wcs(obs):
    """
    Taken straight from ngmix.metacal.metacal
    """
    return obs.psf.jacobian.get_galsim_wcs()

def metacal_deconvolve_single(obs):
    """Single galaxy deconvolution using metacalibration approach with pre-existing observation."""
    try:
        # Validate the observation object
        if not hasattr(obs, 'image') or obs.image is None:
            raise ValueError("Invalid observation: missing or None image")
        if not hasattr(obs, 'psf') or obs.psf is None:
            raise ValueError("Invalid observation: missing or None PSF")

        # Making the image_int_nopsf manually
        __, image_int = _galsim_stuff(
            obs.image,
            get_wcs(obs),
            'lanczos15',
        )
        __, psf_int = _galsim_stuff(
            obs.psf.image,
            get_psf_wcs(obs),
            'lanczos15',
        )
        psf_int_inv = galsim.Deconvolve(psf_int)
        image_int_nopsf = galsim.Convolve(image_int, psf_int_inv)
        
        # Get image size from the observation
        img_height, img_width = obs.image.shape
        # Get pixel scale from jacobian
        pixel_scale = obs.jacobian.get_scale()
        
        # Draw the GalSim object to get pixel values
        drawn_image = image_int_nopsf.drawImage(nx=img_width, ny=img_height, scale=pixel_scale)
        deconv_img = np.real(drawn_image.array)
        
        return deconv_img
        
    except Exception as e:
        print(f"Metacal deconvolution failed for single galaxy: {e}")
        return np.zeros_like(obs.image)


def metacal_deconvolve(observations: List[ngmix.Observation], 
                      n_jobs: int = None, seed: int = 42) -> jnp.ndarray:
    """
    Perform PSF deconvolution using metacalibration approach with pre-existing observations.
    
    This method uses ngmix's MetacalDilatePSF to extract the deconvolved galaxy
    images (image_int_nopsf) which have PSF and pixel effects removed.
    
    Args:
        observations: List of ngmix.Observation objects with properly configured jacobians
        n_jobs: Number of parallel processes (None for auto)
        seed: Random seed base (for consistency)
        
    Returns:
        Deconvolved galaxy images [N, H, W]
    """
    n_galaxies = len(observations)
    
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), n_galaxies)
    
    print(f"Metacalibration deconvolution: {n_galaxies} galaxies, {n_jobs} processes")
    print("Using pre-existing observation objects with proper jacobians")
    
    deconvolved_images = []
    
    if n_jobs == 1:
        # Single process
        for obs in observations:
            result = metacal_deconvolve_single(obs)
            deconvolved_images.append(result)
    else:
        # Multiprocessing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(metacal_deconvolve_single, obs): i 
                      for i, obs in enumerate(observations)}
            
            # Collect results in order
            results = [None] * n_galaxies
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"Error processing galaxy {idx}: {e}")
                    results[idx] = np.zeros_like(observations[idx].image)
            
            deconvolved_images = results
    
    return jnp.array(deconvolved_images)