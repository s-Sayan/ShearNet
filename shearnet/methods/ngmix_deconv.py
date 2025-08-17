"""NGmix-based parametric deconvolution methods for benchmarking - FIXED VERSION."""

import numpy as np
import jax.numpy as jnp
import ngmix
import galsim
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
    galaxy_image = np.asarray(galaxy_image, dtype=np.float64)
    psf_image = np.asarray(psf_image, dtype=np.float64)
    
    # Validate images
    if np.any(~np.isfinite(galaxy_image)) or np.any(~np.isfinite(psf_image)):
        raise ValueError("Images contain NaN or inf values")
    
    # Ensure minimum noise variance to avoid numerical issues
    noise_var = max(noise_var, 1e-8)
    
    # Create weight maps (inverse variance) - ensure they're reasonable
    weight = np.ones_like(galaxy_image, dtype=np.float64) / noise_var
    psf_weight = np.ones_like(psf_image, dtype=np.float64) / (noise_var * 0.1)  # PSF has less noise
    
    # Ensure weights are finite and positive
    weight = np.clip(weight, 1e-10, 1e10)
    psf_weight = np.clip(psf_weight, 1e-10, 1e10)
    
    # Create jacobians - handle both square and non-square images
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


def fit_galaxy_model(obs: ngmix.Observation, model: str = 'exp', 
                    max_attempts: int = 10, seed: int = None) -> Optional[Dict[str, Any]]:
    """
    Fit a parametric galaxy model to an observation.
    
    Args:
        obs: ngmix Observation
        model: Galaxy model type ('exp', 'gauss', 'dev')
        max_attempts: Maximum fitting attempts
        seed: Random seed
        
    Returns:
        Dictionary with fit results or None if failed
    """
    if seed is None:
        seed = np.random.randint(0, 2**31)
    
    rng = np.random.RandomState(seed)
    
    # Validate observation
    if not isinstance(obs, ngmix.Observation):
        raise TypeError(f"obs must be an ngmix Observation, got {type(obs)}")
    
    if obs.psf is None:
        raise ValueError("Observation must have a PSF attached")
    
    # Set up priors with more reasonable ranges
    g_prior = ngmix.priors.GPriorBA(0.3, rng=rng)
    cen_prior = ngmix.priors.CenPrior(0.0, 0.0, 0.2, 0.2, rng=rng)
    T_prior = ngmix.priors.FlatPrior(-1.0, 1000.0, rng=rng)
    F_prior = ngmix.priors.FlatPrior(-1e1, 1e5, rng=rng)
    
    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior, g_prior, T_prior, F_prior
    )
    
    # Set up fitter with more relaxed tolerances
    lm_pars = {'maxfev': 4000, 'xtol': 1.0e-4, 'ftol': 1.0e-4}
    fitter = ngmix.fitting.Fitter(model=model, prior=prior, fit_pars=lm_pars)
    
    # Set up guesser
    T_guess = 4.0 * obs.jacobian.get_scale()**2
    guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
        rng=rng, T=T_guess, prior=prior
    )
    
    # Set up PSF fitter
    psf_fitter = ngmix.fitting.Fitter(model='gauss', fit_pars=lm_pars)
    psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)
    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter, guesser=psf_guesser, ntry=max_attempts
    )
    
    # Set up galaxy runner
    runner = ngmix.runners.Runner(
        fitter=fitter, guesser=guesser, ntry=max_attempts
    )
    
    try:
        # Fit PSF first - pass as ObsList
        psf_obslist = ngmix.ObsList()
        psf_obslist.append(obs.psf)
        
        psf_res_list = psf_runner.go(psf_obslist)
        
        # Extract the first (and only) result from the list
        if isinstance(psf_res_list, list):
            if len(psf_res_list) == 0:
                return None
            psf_res = psf_res_list[0]
        else:
            psf_res = psf_res_list
        
        if psf_res['flags'] != 0:
            return None
            
        # Store PSF result
        obs.psf.meta = {'result': psf_res}
        
        # Fit galaxy - pass as ObsList
        galaxy_obslist = ngmix.ObsList()
        galaxy_obslist.append(obs)
        
        galaxy_res_list = runner.go(galaxy_obslist)
        
        # Extract the first (and only) result from the list
        if isinstance(galaxy_res_list, list):
            if len(galaxy_res_list) == 0:
                return None
            galaxy_res = galaxy_res_list[0]
        else:
            galaxy_res = galaxy_res_list
        
        if galaxy_res['flags'] != 0:
            return None
            
        return {
            'galaxy_result': galaxy_res,
            'psf_result': psf_res,
            'model': model,
            'flags': 0
        }
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        return None


def render_galaxy_model(fit_result: Dict[str, Any], image_shape: tuple, 
                       pixel_scale: float = 0.141, include_psf: bool = False) -> np.ndarray:
    """
    Render a fitted galaxy model.
    
    Args:
        fit_result: Result from fit_galaxy_model
        image_shape: Shape of output image (height, width)
        pixel_scale: Pixel scale in arcsec/pixel
        include_psf: Whether to include PSF convolution
        
    Returns:
        Rendered galaxy image
    """
    if fit_result is None or fit_result['flags'] != 0:
        return np.zeros(image_shape)
    
    galaxy_res = fit_result['galaxy_result']
    psf_res = fit_result['psf_result']
    model_type = fit_result['model']
    
    try:
        # Extract fitted parameters - handle different result structures
        if hasattr(galaxy_res, 'get_result'):
            # If it's an ngmix result object
            pars = galaxy_res.get_result()['pars']
        elif isinstance(galaxy_res, dict) and 'pars' in galaxy_res:
            # If it's already a dictionary
            pars = galaxy_res['pars']
        elif hasattr(galaxy_res, 'pars'):
            # If it's an object with pars attribute
            pars = galaxy_res.pars
        else:
            print(f"Unexpected galaxy_res structure: {type(galaxy_res)}")
            return np.zeros(image_shape)
        
        row, col, g1, g2, T, flux = pars
        
        # Ensure positive flux and reasonable T
        if flux <= 0 or T <= 0:
            return np.zeros(image_shape)
        
        # Convert to GalSim parameters
        sigma = np.sqrt(T / 2.0)  # T = 2 * sigma^2 for Gaussian
        
        # Create galaxy object
        if model_type == 'exp':
            # For exponential: half_light_radius = 1.678 * sigma (approximately)
            half_light_radius = 1.678 * sigma
            galaxy = galsim.Exponential(half_light_radius=half_light_radius, flux=flux)
        elif model_type == 'gauss':
            galaxy = galsim.Gaussian(sigma=sigma, flux=flux)
        elif model_type == 'dev':
            # For de Vaucouleurs: half_light_radius = 1.999 * sigma (approximately) 
            half_light_radius = 1.999 * sigma
            galaxy = galsim.DeVaucouleurs(half_light_radius=half_light_radius, flux=flux)
        else:
            # Default to Gaussian
            galaxy = galsim.Gaussian(sigma=sigma, flux=flux)
        
        # Apply shear (clip to reasonable values)
        g1 = np.clip(g1, -0.8, 0.8)
        g2 = np.clip(g2, -0.8, 0.8)
        galaxy = galaxy.shear(g1=g1, g2=g2)
        
        # Apply center offset (clip to reasonable values)
        dx = np.clip(col * pixel_scale, -pixel_scale, pixel_scale)
        dy = np.clip(row * pixel_scale, -pixel_scale, pixel_scale)
        galaxy = galaxy.shift(dx=dx, dy=dy)
        
        # Create PSF if needed
        if include_psf and psf_res['flags'] == 0:
            # Extract PSF parameters - handle different result structures
            if hasattr(psf_res, 'get_result'):
                psf_pars = psf_res.get_result()['pars']
            elif isinstance(psf_res, dict) and 'pars' in psf_res:
                psf_pars = psf_res['pars']
            elif hasattr(psf_res, 'pars'):
                psf_pars = psf_res.pars
            else:
                psf_pars = None
            
            if psf_pars is not None:
                psf_row, psf_col, psf_g1, psf_g2, psf_T, psf_flux = psf_pars
                
                if psf_T > 0:
                    psf_sigma = np.sqrt(psf_T / 2.0)
                    
                    psf = galsim.Gaussian(sigma=psf_sigma)
                    psf_g1 = np.clip(psf_g1, -0.8, 0.8)
                    psf_g2 = np.clip(psf_g2, -0.8, 0.8)
                    psf = psf.shear(g1=psf_g1, g2=psf_g2)
                    
                    # Convolve galaxy with PSF
                    galaxy = galsim.Convolve(galaxy, psf)
        
        # Render image
        image = galaxy.drawImage(
            nx=image_shape[1], ny=image_shape[0], 
            scale=pixel_scale, method='auto'
        )
        
        return image.array
        
    except Exception as e:
        print(f"Rendering failed: {e}")
        return np.zeros(image_shape)


def ngmix_parametric_deconvolve_single(args):
    """Single galaxy deconvolution for multiprocessing."""
    galaxy_img, psf_img, model, noise_var, pixel_scale, seed = args
    
    try:
        # Ensure arrays are contiguous and the right type
        galaxy_img = np.ascontiguousarray(galaxy_img, dtype=np.float64)
        psf_img = np.ascontiguousarray(psf_img, dtype=np.float64)
        
        # Create observation
        obs = create_ngmix_observation(galaxy_img, psf_img, noise_var, pixel_scale)
        
        # Fit model
        fit_result = fit_galaxy_model(obs, model=model, seed=seed)
        
        if fit_result is None:
            # Return zeros if fitting failed
            return np.zeros_like(galaxy_img)
        
        # Render deconvolved galaxy (without PSF)
        deconv_img = render_galaxy_model(
            fit_result, galaxy_img.shape, pixel_scale, include_psf=False
        )
        
        return deconv_img
        
    except Exception as e:
        # Print the error but don't raise it to avoid breaking multiprocessing
        # print(f"Deconvolution failed for single galaxy: {e}")
        return np.zeros_like(galaxy_img)


def ngmix_parametric_deconvolve(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray,
                               model: str = 'exp', noise_var: float = 1e-8,
                               pixel_scale: float = 0.141, n_jobs: int = None,
                               seed: int = 42) -> jnp.ndarray:
    """
    Perform parametric deconvolution using ngmix model fitting.
    
    Args:
        galaxy_images: Observed galaxy images [N, H, W] or [N, H, W, 1]
        psf_images: PSF images [N, H, W] or [N, H, W, 1]
        model: Galaxy model type ('exp', 'gauss', 'dev')
        noise_var: Noise variance estimate (increased default)
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
    
    print(f"NGmix parametric deconvolution: {n_galaxies} galaxies, {n_jobs} processes, model={model}")
    
    # Prepare arguments for multiprocessing
    args_list = [
        (galaxy_images[i], psf_images[i], model, noise_var, pixel_scale, seed + i)
        for i in range(n_galaxies)
    ]
    
    deconvolved_images = []
    
    if n_jobs == 1:
        # Single process
        for args in args_list:
            result = ngmix_parametric_deconvolve_single(args)
            deconvolved_images.append(result)
    else:
        # Multiprocessing
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(ngmix_parametric_deconvolve_single, args): i 
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


# Main deconvolution functions that replace the old FFT methods
def ngmix_exp_deconvolve(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                        **kwargs) -> jnp.ndarray:
    """
    NGmix exponential model deconvolution (replaces fourier_deconvolve).
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        **kwargs: Additional arguments for ngmix_parametric_deconvolve
        
    Returns:
        Deconvolved galaxy images
    """
    return ngmix_parametric_deconvolve(galaxy_images, psf_images, model='exp', **kwargs)


def ngmix_gauss_deconvolve(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                          **kwargs) -> jnp.ndarray:
    """
    NGmix Gaussian model deconvolution (replaces wiener_deconvolve).
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        **kwargs: Additional arguments for ngmix_parametric_deconvolve
        
    Returns:
        Deconvolved galaxy images
    """
    return ngmix_parametric_deconvolve(galaxy_images, psf_images, model='gauss', **kwargs)


def ngmix_dev_deconvolve(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                        **kwargs) -> jnp.ndarray:
    """
    NGmix de Vaucouleurs model deconvolution (replaces richardson_lucy_deconvolve).
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        **kwargs: Additional arguments for ngmix_parametric_deconvolve
        
    Returns:
        Deconvolved galaxy images
    """
    return ngmix_parametric_deconvolve(galaxy_images, psf_images, model='dev', **kwargs)


def evaluate_ngmix_methods(galaxy_images: jnp.ndarray, psf_images: jnp.ndarray, 
                          target_images: jnp.ndarray) -> dict:
    """
    Evaluate multiple NGmix-based deconvolution methods.
    
    Args:
        galaxy_images: Observed galaxy images
        psf_images: PSF images
        target_images: Ground truth clean images
        
    Returns:
        Dictionary with results from different methods
    """
    methods = {
        'NGmix Exponential': lambda g, p: ngmix_exp_deconvolve(g, p),
        'NGmix Gaussian': lambda g, p: ngmix_gauss_deconvolve(g, p),
        'NGmix de Vaucouleurs': lambda g, p: ngmix_dev_deconvolve(g, p),
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"Evaluating {method_name}...")
        
        try:
            start_time = time.time()
            predictions = method_func(galaxy_images, psf_images)
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            mse = float(jnp.mean((predictions - target_images) ** 2))
            mae = float(jnp.mean(jnp.abs(predictions - target_images)))
            
            if mse > 0:
                max_val = float(jnp.max(target_images))
                psnr = 20 * jnp.log10(max_val / jnp.sqrt(mse))
            else:
                psnr = float('inf')
            
            # Bias
            bias = float(jnp.mean(predictions - target_images))
            
            results[method_name] = {
                'predictions': predictions,
                'mse': mse,
                'mae': mae,
                'psnr': float(psnr),
                'bias': bias,
                'time_taken': elapsed_time
            }
            
            print(f"  MSE: {mse:.3e}, PSNR: {psnr:.1f} dB, Time: {elapsed_time:.1f}s")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[method_name] = {'error': str(e)}
    
    return results