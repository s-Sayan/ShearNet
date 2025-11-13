"""Metrics and evaluation functions for ShearNet.

This module contains functions for evaluating different shear estimation
methods including neural networks, NGmix, and metacalibration.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Tuple, Optional, Any

# ANSI color codes for pretty printing
BOLD = '\033[1m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
GREEN = '\033[92m'
END = '\033[0m'

def remove_nan_preds_multi(pred1: np.ndarray, pred2: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove rows where either pred1 or pred2 contains NaNs.

    Parameters
    ----------
    pred1 : np.ndarray
        First prediction array (e.g., model predictions), shape (N, D)
    pred2 : np.ndarray
        Second prediction array (e.g., ngmix predictions), shape (N, D)
    labels : np.ndarray
        Ground truth labels, shape (N, D)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (filtered_pred1, filtered_pred2, filtered_labels)
    """
    mask = ~np.any(np.isnan(pred1) | np.isnan(pred2), axis=1)
    num_removed = np.sum(~mask)
    if num_removed > 0:
        print(f"[NaN Filter] Removed {num_removed} rows with NaNs in predictions.")
    return pred1[mask], pred2[mask], labels[mask]

def remove_nan_preds(preds: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove rows where preds contain NaNs.

    Parameters
    ----------
    preds : np.ndarray
        Array of predicted values, shape (N, D)
    labels : np.ndarray
        Array of true values, shape (N, D)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (filtered_preds, corresponding_labels)
    """
    mask = ~np.any(np.isnan(preds), axis=1)
    num_removed = np.sum(~mask)
    if num_removed > 0:
        print(f"[NaN Filter] Removed {num_removed} rows with NaNs in predictions.")
    return preds[mask], labels[mask]

def loss_fn_mcal(images, labels, psf_fwhm):
    """Calculate the loss for the MCAL model.
    
    Parameters
    ----------
    images : jnp.ndarray
        Input images
    labels : jnp.ndarray
        True labels
    psf_fwhm : float
        PSF FWHM
        
    Returns
    -------
    loss : float
        Combined loss
    preds : jnp.ndarray
        Predictions (only g1, g2)
    loss_per_label : dict
        Per-label losses
    """
    from ..methods.metacal import mcal_preds
    
    preds = mcal_preds(images, psf_fwhm)
    
    # Combined loss for g1, g2
    loss = optax.l2_loss(preds[:, :2], labels[:, :2]).mean()
    
    # Per-label losses (only g1, g2 for mcal)
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], labels[:, 1]).mean(),
        'g1g2_combined': loss  # Same as combined loss
    }
    
    return loss, preds, loss_per_label


def loss_fn_ngmix(obs_list, labels, seed=1234, psf_model='gauss', gal_model='gauss'):
    """Calculate the loss for the NGmix model.
    
    Parameters
    ----------
    obs_list : list
        List of observation dictionaries
    labels : np.ndarray
        True labels
    seed : int, optional
        Random seed
    psf_model : str, optional
        PSF model type
    gal_model : str, optional
        Galaxy model type
        
    Returns
    -------
    loss : float
        Combined loss
    preds : np.ndarray
        Predictions
    loss_per_label : dict
        Per-label losses
    """
    from ..methods.ngmix import _get_priors, mp_fit_one, ngmix_pred
    
    prior = _get_priors(seed)
    rng = np.random.RandomState(seed)
    datalist = mp_fit_one(obs_list, prior, rng, psf_model=psf_model, gal_model=gal_model)
    preds = ngmix_pred(datalist)

    pred_filtered, labels_filtered = remove_nan_preds(preds, labels)
    
    # Combined loss
    loss = optax.l2_loss(pred_filtered, labels_filtered).mean()
    bias = (pred_filtered - labels_filtered).mean()
    # Per-label losses
    loss_per_label = {
        'g1': optax.l2_loss(pred_filtered[:, 0], labels_filtered[:, 0]).mean(),
        'g2': optax.l2_loss(pred_filtered[:, 1], labels_filtered[:, 1]).mean(),
        'g1g2_combined': optax.l2_loss(pred_filtered[:, :2], labels_filtered[:, :2]).mean(),  # Combined g1,g2
        'sigma': optax.l2_loss(pred_filtered[:, 2], labels_filtered[:, 2]).mean(),
        'flux': optax.l2_loss(pred_filtered[:, 3], labels_filtered[:, 3]).mean()
    }

    # Per-label biases
    bias_per_label = {
        'g1': (pred_filtered[:, 0] - labels_filtered[:, 0]).mean(),
        'g2': (pred_filtered[:, 1] - labels_filtered[:, 1]).mean(),
        'g1g2_combined': (pred_filtered[:, :2] - labels_filtered[:, :2]).mean(),  # Average bias for g1,g2
        'sigma': (pred_filtered[:, 2] - labels_filtered[:, 2]).mean(),
        'flux': (pred_filtered[:, 3] - labels_filtered[:, 3]).mean()
    }

    return loss, preds, loss_per_label, bias, bias_per_label


def loss_fn_eval(state, params, images, labels):
    """Calculate evaluation loss for neural network predictions.
    
    Parameters
    ----------
    state : train_state.TrainState
        The training state object
    params : dict
        Model parameters
    images : jnp.ndarray
        Input images
    labels : jnp.ndarray
        True labels (g1, g2, sigma, flux)
        
    Returns
    -------
    loss : float
        Combined mean squared error
    loss_per_label : dict
        Per-label MSE values
    """
    preds = state.apply_fn(params, images)
    
    # Combined loss (assuming preds shape matches labels shape)
    loss = optax.l2_loss(preds, labels).mean()
    
    # Per-label losses
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], labels[:, 1]).mean(),
        'g1g2_combined': optax.l2_loss(preds[:, :2], labels[:, :2]).mean(),
        'sigma': optax.l2_loss(preds[:, 2], labels[:, 2]).mean(),
        'flux': optax.l2_loss(preds[:, 3], labels[:, 3]).mean()
    }
    
    return loss, loss_per_label

def fork_loss_fn_eval(state, params, galaxy_images, psf_images, labels):

    preds = state.apply_fn(params, galaxy_images, psf_images)
    
    # Combined loss (assuming preds shape matches labels shape)
    loss = optax.l2_loss(preds, labels).mean()
    
    # Per-label losses
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], labels[:, 1]).mean(),
        'g1g2_combined': optax.l2_loss(preds[:, :2], labels[:, :2]).mean(),
        'sigma': optax.l2_loss(preds[:, 2], labels[:, 2]).mean(),
        'flux': optax.l2_loss(preds[:, 3], labels[:, 3]).mean()
    }
    
    return loss, loss_per_label


def eval_mcal(test_images, test_labels, psf_fwhm) -> Dict[str, Any]:
    """Evaluate metacalibration on the entire test set at once.
    
    Parameters
    ----------
    test_images : jnp.ndarray
        Test images
    test_labels : jnp.ndarray
        Test labels
    psf_fwhm : float
        PSF FWHM
        
    Returns
    -------
    dict
        Dictionary containing loss, bias, per-label metrics, and predictions
    """
    from ..methods.mcal import mcal_preds
    
    # Get all predictions
    start_time = time.time()
    preds = mcal_preds(test_images, psf_fwhm)
    
    # Combined metrics
    loss = optax.l2_loss(preds[:, :2], test_labels[:, :2]).mean()
    bias = (preds - test_labels[:, :2]).mean()
    
    # Per-label metrics
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], test_labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], test_labels[:, 1]).mean(),
        'g1g2_combined': loss
    }
    
    bias_per_label = {
        'g1': (preds[:, 0] - test_labels[:, 0]).mean(),
        'g2': (preds[:, 1] - test_labels[:, 1]).mean(),
        'g1g2_combined': bias
    }
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n=== Combined Metrics (Moment-Based Approach) ===")
    print(f"Mean Squared Error (MSE) from MOM: {loss:.6e}")
    print(f"Average Bias from MOM: {bias:.6e}")
    print(f"Time taken: {total_time:.2f} seconds")
    
    print("\n=== Per-Label Metrics ===")
    label_names = ['g1', 'g2', 'g1g2_combined']
    for label in label_names:
        print(f"{label:>15}: MSE = {loss_per_label[label]:.6e}, Bias = {bias_per_label[label]:+.6e}")
    print()
    
    return {
        'loss': loss,
        'bias': bias,
        'loss_per_label': loss_per_label,
        'bias_per_label': bias_per_label,
        'preds': preds,
        'time_taken': total_time
    }


def eval_ngmix(test_obs, test_labels, seed=1234, psf_model='gauss', gal_model='gauss') -> Dict[str, Any]:
    """Evaluate the model using ngmix on the entire test set.
    
    Parameters
    ----------
    test_obs : list
        List of observation dictionaries
    test_labels : np.ndarray
        True labels
    seed : int, optional
        Random seed
    psf_model : str, optional
        PSF model type
    gal_model : str, optional  
        Galaxy model type
        
    Returns
    -------
    dict
        Dictionary containing loss, bias, per-label metrics, and predictions
    """
    start_time = time.time()

    loss, preds, loss_per_label, bias, bias_per_label = loss_fn_ngmix(test_obs, test_labels, seed, 
                                                psf_model=psf_model, gal_model=gal_model)
    
    total_time = time.time() - start_time
    
    # Print combined metrics
    print(f"\n{BOLD}=== Combined Metrics (NGmix) ==={END}")
    print(f"Mean Squared Error (MSE) from NGmix: {BOLD}{YELLOW}{loss:.6e}{END}")
    print(f"Average Bias from NGmix: {BOLD}{YELLOW}{bias:.6e}{END}")
    print(f"Time taken: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    
    # Print per-label metrics
    print("\n=== Per-Label Metrics ===")
    label_names = ['g1', 'g2', 'g1g2_combined', 'sigma', 'flux']
    for label in label_names:
        print(f"{label:>15}: MSE = {loss_per_label[label]:.6e}, Bias = {bias_per_label[label]:+.6e}")
    print()
    
    return {
        'loss': loss,
        'bias': bias,
        'loss_per_label': loss_per_label,
        'bias_per_label': bias_per_label,
        'preds': preds,
        'time_taken': total_time
    }


@jax.jit
def eval_step(state, images, labels):
    """Evaluate the model on a single batch (JIT compiled).
    
    Parameters
    ----------
    state : train_state.TrainState
        The training state object
    images : jnp.ndarray
        Batch of input images
    labels : jnp.ndarray
        Batch of true labels
        
    Returns
    -------
    loss : float
        Batch loss
    preds : jnp.ndarray
        Predictions
    loss_per_label : dict
        Per-label losses
    bias_per_label : dict
        Per-label biases
    """
    loss, loss_per_label = loss_fn_eval(state, state.params, images, labels)
    preds = state.apply_fn(state.params, images, deterministic=True)
    
    # Calculate per-label biases
    bias_per_label = {
        'g1': (preds[:, 0] - labels[:, 0]).mean(),
        'g2': (preds[:, 1] - labels[:, 1]).mean(),
        'g1g2_combined': (preds[:, :2] - labels[:, :2]).mean(),
        'sigma': (preds[:, 2] - labels[:, 2]).mean(),
        'flux': (preds[:, 3] - labels[:, 3]).mean()
    }
    
    return loss, preds, loss_per_label, bias_per_label

@jax.jit
def fork_eval_step(state, images, psf_images, labels):
    
    loss, loss_per_label = fork_loss_fn_eval(state, state.params, images, psf_images, labels)
    preds = state.apply_fn(state.params, images, psf_images, deterministic=True)
    
    # Calculate per-label biases
    bias_per_label = {
        'g1': (preds[:, 0] - labels[:, 0]).mean(),
        'g2': (preds[:, 1] - labels[:, 1]).mean(),
        'g1g2_combined': (preds[:, :2] - labels[:, :2]).mean(),
        'sigma': (preds[:, 2] - labels[:, 2]).mean(),
        'flux': (preds[:, 3] - labels[:, 3]).mean()
    }
    
    return loss, preds, loss_per_label, bias_per_label


def eval_model(state, test_images, test_labels, batch_size=32) -> Dict[str, Any]:
    """Evaluate the neural network model on the entire test set.
    
    Parameters
    ----------
    state : train_state.TrainState
        The training state object
    test_images : jnp.ndarray
        Test images
    test_labels : jnp.ndarray
        Test labels
    batch_size : int, optional
        Batch size for evaluation
        
    Returns
    -------
    dict
        Dictionary containing loss, bias, per-label metrics, predictions, and timing
    """
    start_time = time.time()

    total_loss = 0
    total_samples = 0
    total_bias = 0
    
    # Initialize per-label accumulators
    total_loss_per_label = {
        'g1': 0, 'g2': 0, 'g1g2_combined': 0, 'sigma': 0, 'flux': 0
    }
    total_bias_per_label = {
        'g1': 0, 'g2': 0, 'g1g2_combined': 0, 'sigma': 0, 'flux': 0
    }
    
    all_preds = []

    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        loss, preds, loss_per_label, bias_per_label = eval_step(state, batch_images, batch_labels)
        
        all_preds.append(preds)
        
        batch_bias = (preds - batch_labels).mean()
        batch_size_actual = len(batch_images)
        
        # Accumulate combined metrics
        total_loss += loss * batch_size_actual
        total_bias += batch_bias * batch_size_actual
        total_samples += batch_size_actual
        
        # Accumulate per-label metrics
        for label in total_loss_per_label:
            total_loss_per_label[label] += loss_per_label[label] * batch_size_actual
            total_bias_per_label[label] += bias_per_label[label] * batch_size_actual

    # Calculate averages
    avg_loss = total_loss / total_samples
    avg_bias = total_bias / total_samples
    
    avg_loss_per_label = {
        label: total / total_samples 
        for label, total in total_loss_per_label.items()
    }
    avg_bias_per_label = {
        label: total / total_samples 
        for label, total in total_bias_per_label.items()
    }
    
    total_time = time.time() - start_time
    
    # Print combined metrics
    print(f"\n{BOLD}=== Combined Metrics (ShearNet) ==={END}")
    print(f"Mean Squared Error (MSE) from ShearNet: {BOLD}{YELLOW}{avg_loss:.6e}{END}")
    print(f"Average Bias from ShearNet: {BOLD}{YELLOW}{avg_bias:.6e}{END}")
    print(f"Time taken: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    
    # Print per-label metrics
    print("\n=== Per-Label Metrics ===")
    label_names = ['g1', 'g2', 'g1g2_combined', 'sigma', 'flux']
    for label in label_names:
        print(f"{label:>15}: MSE = {avg_loss_per_label[label]:.6e}, Bias = {avg_bias_per_label[label]:+.6e}")
    print()
    
    return {
        'loss': avg_loss,
        'bias': avg_bias,
        'loss_per_label': avg_loss_per_label,
        'bias_per_label': avg_bias_per_label,
        'all_preds': jnp.concatenate(all_preds) if all_preds else None,
        'time_taken': total_time
    }

def fork_eval_model(state, test_images, test_psf_images, test_labels, batch_size=32) -> Dict[str, Any]:
    
    start_time = time.time()

    total_loss = 0
    total_samples = 0
    total_bias = 0
    
    # Initialize per-label accumulators
    total_loss_per_label = {
        'g1': 0, 'g2': 0, 'g1g2_combined': 0, 'sigma': 0, 'flux': 0
    }
    total_bias_per_label = {
        'g1': 0, 'g2': 0, 'g1g2_combined': 0, 'sigma': 0, 'flux': 0
    }
    
    all_preds = []

    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_psf_images = test_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        loss, preds, loss_per_label, bias_per_label = fork_eval_step(state, batch_images, batch_psf_images, batch_labels)
        
        all_preds.append(preds)
        
        batch_bias = (preds - batch_labels).mean()
        batch_size_actual = len(batch_images)
        
        # Accumulate combined metrics
        total_loss += loss * batch_size_actual
        total_bias += batch_bias * batch_size_actual
        total_samples += batch_size_actual
        
        # Accumulate per-label metrics
        for label in total_loss_per_label:
            total_loss_per_label[label] += loss_per_label[label] * batch_size_actual
            total_bias_per_label[label] += bias_per_label[label] * batch_size_actual

    # Calculate averages
    avg_loss = total_loss / total_samples
    avg_bias = total_bias / total_samples
    
    avg_loss_per_label = {
        label: total / total_samples 
        for label, total in total_loss_per_label.items()
    }
    avg_bias_per_label = {
        label: total / total_samples 
        for label, total in total_bias_per_label.items()
    }
    
    total_time = time.time() - start_time
    
    # Print combined metrics
    print(f"\n{BOLD}=== Combined Metrics (ShearNet) ==={END}")
    print(f"Mean Squared Error (MSE) from ShearNet: {BOLD}{YELLOW}{avg_loss:.6e}{END}")
    print(f"Average Bias from ShearNet: {BOLD}{YELLOW}{avg_bias:.6e}{END}")
    print(f"Time taken: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    
    # Print per-label metrics
    print("\n=== Per-Label Metrics ===")
    label_names = ['g1', 'g2', 'g1g2_combined', 'sigma', 'flux']
    for label in label_names:
        print(f"{label:>15}: MSE = {avg_loss_per_label[label]:.6e}, Bias = {avg_bias_per_label[label]:+.6e}")
    print()
    
    return {
        'loss': avg_loss,
        'bias': avg_bias,
        'loss_per_label': avg_loss_per_label,
        'bias_per_label': avg_bias_per_label,
        'all_preds': jnp.concatenate(all_preds) if all_preds else None,
        'time_taken': total_time
    }

def calculate_response_matrix(state, observations, batch_size=32, h=0.01, model_type='standard', psf_images=None):
    """
    Calculate the shear response matrix R for calibration.
    
    The response matrix relates measured shear to true shear:
    R_ij = d(g_i_measured) / d(g_j_true)
    
    Args:
        state: Model state
        observations: List of ngmix observations with sheared images in metadata
        batch_size: Batch size for evaluation
        h: Shear step size (should match generation, default 0.01)
        model_type: 'standard' or 'fork'
        psf_images: PSF images array (required if model_type='fork')
    
    Returns:
        R: 2x2 response matrix averaged over all galaxies
        R_per_galaxy: Response matrices for each galaxy (N, 2, 2)
    """
    import jax.numpy as jnp
    
    print(f"\n{BOLD}Calculating Response Matrix...{END}")
    
    # Extract sheared images from observations metadata
    n_gal = len(observations)
    e1_positive_images = np.array([obs.meta['e1_positive'] for obs in observations])
    e1_negative_images = np.array([obs.meta['e1_negative'] for obs in observations])
    e2_positive_images = np.array([obs.meta['e2_positive'] for obs in observations])
    e2_negative_images = np.array([obs.meta['e2_negative'] for obs in observations])
    
    print(f"Extracted {n_gal} galaxies with sheared images")
    print(f"Shear step: ±{h}")
    
    # Get predictions for each sheared version
    def get_predictions(images, psf_imgs=None):
        preds = []
        for i in range(0, n_gal, batch_size):
            batch_slice = slice(i, min(i + batch_size, n_gal))
            if model_type == 'fork':
                batch_pred = state.apply_fn(
                    state.params,
                    images[batch_slice],
                    psf_imgs[batch_slice],
                    deterministic=True
                )
            else:
                batch_pred = state.apply_fn(
                    state.params,
                    images[batch_slice],
                    deterministic=True
                )
            preds.append(batch_pred)
        return jnp.concatenate(preds)
    
    # Get all predictions
    preds_e1p = get_predictions(e1_positive_images, psf_images)
    preds_e1m = get_predictions(e1_negative_images, psf_images)
    preds_e2p = get_predictions(e2_positive_images, psf_images)
    preds_e2m = get_predictions(e2_negative_images, psf_images)
    
    # Calculate response matrix components using finite differences
    # R_ij = d(g_i_measured) / d(g_j_true)
    R_11 = (preds_e1p[:, 0] - preds_e1m[:, 0]) / (2 * h)  # dg1_meas/dg1_true
    R_12 = (preds_e2p[:, 0] - preds_e2m[:, 0]) / (2 * h)  # dg1_meas/dg2_true
    R_21 = (preds_e1p[:, 1] - preds_e1m[:, 1]) / (2 * h)  # dg2_meas/dg1_true
    R_22 = (preds_e2p[:, 1] - preds_e2m[:, 1]) / (2 * h)  # dg2_meas/dg2_true
    
    # Stack into per-galaxy matrices
    R_per_galaxy = jnp.stack([
        jnp.stack([R_11, R_12], axis=1),
        jnp.stack([R_21, R_22], axis=1)
    ], axis=1)  # Shape: (n_gal, 2, 2)
    
    # Average response matrix
    R = jnp.mean(R_per_galaxy, axis=0)
    
    # Print results
    print(f"\n{BOLD}Response Matrix (averaged over {n_gal} galaxies):{END}")
    print(f"{CYAN}R = [[{R[0,0]:.6f}, {R[0,1]:.6f}],{END}")
    print(f"{CYAN}     [{R[1,0]:.6f}, {R[1,1]:.6f}]]{END}")
    print(f"\nResponse matrix statistics:")
    print(f"  R_11: mean={float(jnp.mean(R_11)):.6f}, std={float(jnp.std(R_11)):.6f}")
    print(f"  R_22: mean={float(jnp.mean(R_22)):.6f}, std={float(jnp.std(R_22)):.6f}")
    print(f"  R_12: mean={float(jnp.mean(R_12)):.6f}, std={float(jnp.std(R_12)):.6f}")
    print(f"  R_21: mean={float(jnp.mean(R_21)):.6f}, std={float(jnp.std(R_21)):.6f}")
    
    return R, R_per_galaxy


def calculate_ngmix_response_matrix(observations, test_labels, h=0.1, seed=1234, psf_model='gauss', gal_model='gauss'):
    """
    Calculate response matrix for NGmix estimator.
    
    For NGmix, we need to re-run the fitter on sheared images.
    """
    from ..methods.ngmix import _get_priors, mp_fit_one, ngmix_pred
    import ngmix
    import jax.numpy as jnp
    
    print(f"\n{BOLD}Calculating NGmix Response Matrix...{END}")
    
    # Create new observations with sheared images
    n_gal = len(observations)
    
    # Helper to create obs list with different images
    def create_sheared_obs(image_key):
        new_obs = []
        for obs in observations:
            new_obs.append(ngmix.Observation(
                image=obs.meta[image_key],
                weight=obs.weight,
                jacobian=obs.jacobian,
                psf=obs.psf,
                bmask=obs.bmask,
                ormask=obs.ormask
            ))
        return new_obs
    
    # Run NGmix on all sheared versions
    prior = _get_priors(seed)
    rng = np.random.RandomState(seed)
    
    print("Running NGmix on e1_positive images...")
    obs_e1p = create_sheared_obs('e1_positive')
    data_e1p = mp_fit_one(obs_e1p, prior, rng, psf_model=psf_model, gal_model=gal_model)
    preds_e1p = ngmix_pred(data_e1p)
    
    print("Running NGmix on e1_negative images...")
    obs_e1m = create_sheared_obs('e1_negative')
    data_e1m = mp_fit_one(obs_e1m, prior, rng, psf_model=psf_model, gal_model=gal_model)
    preds_e1m = ngmix_pred(data_e1m)
    
    print("Running NGmix on e2_positive images...")
    obs_e2p = create_sheared_obs('e2_positive')
    data_e2p = mp_fit_one(obs_e2p, prior, rng, psf_model=psf_model, gal_model=gal_model)
    preds_e2p = ngmix_pred(data_e2p)
    
    print("Running NGmix on e2_negative images...")
    obs_e2m = create_sheared_obs('e2_negative')
    data_e2m = mp_fit_one(obs_e2m, prior, rng, psf_model=psf_model, gal_model=gal_model)
    preds_e2m = ngmix_pred(data_e2m)
    
    # Calculate response matrix
    R_11 = (preds_e1p[:, 0] - preds_e1m[:, 0]) / (2 * h)
    R_12 = (preds_e2p[:, 0] - preds_e2m[:, 0]) / (2 * h)
    R_21 = (preds_e1p[:, 1] - preds_e1m[:, 1]) / (2 * h)
    R_22 = (preds_e2p[:, 1] - preds_e2m[:, 1]) / (2 * h)
    
    # Remove NaNs
    valid_mask = ~(np.isnan(R_11) | np.isnan(R_12) | np.isnan(R_21) | np.isnan(R_22))
    n_valid = np.sum(valid_mask)
    
    print(f"Valid galaxies after removing NaNs: {n_valid}/{n_gal}")
    
    R_11 = R_11[valid_mask]
    R_12 = R_12[valid_mask]
    R_21 = R_21[valid_mask]
    R_22 = R_22[valid_mask]
    
    # Stack into matrices
    R_per_galaxy = np.stack([
        np.stack([R_11, R_12], axis=1),
        np.stack([R_21, R_22], axis=1)
    ], axis=1)
    
    R = np.mean(R_per_galaxy, axis=0)
    
    # Print results
    print(f"\n{BOLD}NGmix Response Matrix (averaged over {n_valid} galaxies):{END}")
    print(f"{CYAN}R = [[{R[0,0]:.6f}, {R[0,1]:.6f}],{END}")
    print(f"{CYAN}     [{R[1,0]:.6f}, {R[1,1]:.6f}]]{END}")
    
    return R, R_per_galaxy

def calculate_multiplicative_bias(state, obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
                                  true_shear_step=0.02, batch_size=32, h=0.01,
                                  model_type='standard', 
                                  psf_g1_pos=None, psf_g1_neg=None, 
                                  psf_g2_pos=None, psf_g2_neg=None):
    """
    Calculate multiplicative (m) and additive (c) bias for both shear components.
    
    Uses the two-dataset method:
    - Generate datasets at γ_true = ±shear_step for each component
    - Calculate response matrix and mean ellipticity for each
    - Solve: γ_est = (1+m)γ_true + c
    - m = [γ_est(+) - γ_est(-)] / (2*shear_step) - 1
    - c = [γ_est(+) + γ_est(-)] / 2
    
    Args:
        state: Model state
        obs_g1_pos: Observations with base_shear_g1 = +shear_step
        obs_g1_neg: Observations with base_shear_g1 = -shear_step  
        obs_g2_pos: Observations with base_shear_g2 = +shear_step
        obs_g2_neg: Observations with base_shear_g2 = -shear_step
        true_shear_step: Applied shear magnitude (default 0.02)
        batch_size: Batch size for evaluation
        h: Response matrix perturbation size (default 0.01)
        model_type: 'standard' or 'fork'
        psf_*: PSF images for fork models
    
    Returns:
        dict: {
            'm1': float, 'c1': float,
            'm2': float, 'c2': float,
            'gamma_est_g1_pos': float, 'gamma_est_g1_neg': float,
            'gamma_est_g2_pos': float, 'gamma_est_g2_neg': float,
            'R_g1_pos': array, 'R_g1_neg': array,
            'R_g2_pos': array, 'R_g2_neg': array
        }
    """
    import jax.numpy as jnp
    
    print(f"\n{BOLD}{'='*70}")
    print("CALCULATING MULTIPLICATIVE AND ADDITIVE BIAS")
    print(f"{'='*70}{END}")
    print(f"True shear: ±{true_shear_step}")
    print(f"Response perturbation: ±{h}")
    
    # ========== Component 1 (g1) ==========
    print(f"\n{BOLD}{CYAN}--- Component 1 (g1) ---{END}")
    
    # Calculate response matrices
    print(f"\nDataset A (g1 = +{true_shear_step}):")
    R_g1_pos, _ = calculate_response_matrix(
        state, obs_g1_pos, batch_size=batch_size, h=h,
        model_type=model_type, psf_images=psf_g1_pos
    )
    
    print(f"\nDataset B (g1 = -{true_shear_step}):")
    R_g1_neg, _ = calculate_response_matrix(
        state, obs_g1_neg, batch_size=batch_size, h=h,
        model_type=model_type, psf_images=psf_g1_neg
    )
    
    # Get mean ellipticities and calculate estimated shears
    def get_gamma_est(observations, response_matrix, component_idx, psf_imgs=None):
        """Calculate γ_est = ⟨e⟩ / ⟨R_ii⟩"""
        n_gal = len(observations)
        images = np.array([obs.image for obs in observations])
        
        preds = []
        for i in range(0, n_gal, batch_size):
            batch_slice = slice(i, min(i + batch_size, n_gal))
            if model_type == 'fork':
                batch_pred = state.apply_fn(
                    state.params, images[batch_slice], psf_imgs[batch_slice], deterministic=True
                )
            else:
                batch_pred = state.apply_fn(
                    state.params, images[batch_slice], deterministic=True
                )
            preds.append(batch_pred)
        
        preds = jnp.concatenate(preds)
        mean_e = jnp.mean(preds[:, component_idx])
        R_diag = response_matrix[component_idx, component_idx]
        return float(mean_e / R_diag), float(mean_e), float(R_diag)
    
    gamma_est_g1_pos, mean_e1_pos, R11_pos = get_gamma_est(obs_g1_pos, R_g1_pos, 0, psf_g1_pos)
    gamma_est_g1_neg, mean_e1_neg, R11_neg = get_gamma_est(obs_g1_neg, R_g1_neg, 0, psf_g1_neg)
    
    # Calculate m1 and c1
    m1 = (gamma_est_g1_pos - gamma_est_g1_neg) / (2 * true_shear_step) - 1
    c1 = (gamma_est_g1_pos + gamma_est_g1_neg) / 2
    
    print(f"\n{CYAN}Summary for g1:{END}")
    print(f"  Dataset A: ⟨e₁⟩ = {mean_e1_pos:+.6f}, ⟨R₁₁⟩ = {R11_pos:.6f}, γ₁ᵉˢᵗ = {gamma_est_g1_pos:+.6f}")
    print(f"  Dataset B: ⟨e₁⟩ = {mean_e1_neg:+.6f}, ⟨R₁₁⟩ = {R11_neg:.6f}, γ₁ᵉˢᵗ = {gamma_est_g1_neg:+.6f}")
    
    # ========== Component 2 (g2) ==========
    print(f"\n{BOLD}{CYAN}--- Component 2 (g2) ---{END}")
    
    # Calculate response matrices
    print(f"\nDataset C (g2 = +{true_shear_step}):")
    R_g2_pos, _ = calculate_response_matrix(
        state, obs_g2_pos, batch_size=batch_size, h=h,
        model_type=model_type, psf_images=psf_g2_pos
    )
    
    print(f"\nDataset D (g2 = -{true_shear_step}):")
    R_g2_neg, _ = calculate_response_matrix(
        state, obs_g2_neg, batch_size=batch_size, h=h,
        model_type=model_type, psf_images=psf_g2_neg
    )
    
    gamma_est_g2_pos, mean_e2_pos, R22_pos = get_gamma_est(obs_g2_pos, R_g2_pos, 1, psf_g2_pos)
    gamma_est_g2_neg, mean_e2_neg, R22_neg = get_gamma_est(obs_g2_neg, R_g2_neg, 1, psf_g2_neg)
    
    # Calculate m2 and c2
    m2 = (gamma_est_g2_pos - gamma_est_g2_neg) / (2 * true_shear_step) - 1
    c2 = (gamma_est_g2_pos + gamma_est_g2_neg) / 2
    
    print(f"\n{CYAN}Summary for g2:{END}")
    print(f"  Dataset C: ⟨e₂⟩ = {mean_e2_pos:+.6f}, ⟨R₂₂⟩ = {R22_pos:.6f}, γ₂ᵉˢᵗ = {gamma_est_g2_pos:+.6f}")
    print(f"  Dataset D: ⟨e₂⟩ = {mean_e2_neg:+.6f}, ⟨R₂₂⟩ = {R22_neg:.6f}, γ₂ᵉˢᵗ = {gamma_est_g2_neg:+.6f}")
    
    # ========== Final Results ==========
    print(f"\n{BOLD}{'='*70}")
    print("BIAS CALIBRATION RESULTS")
    print(f"{'='*70}{END}")
    print(f"\n{BOLD}{YELLOW}Component 1:{END}")
    print(f"{BOLD}  m₁ = {m1:+.6f}  ({m1*100:+.2f}%){END}")
    print(f"{BOLD}  c₁ = {c1:+.6f}{END}")
    print(f"\n{BOLD}{YELLOW}Component 2:{END}")
    print(f"{BOLD}  m₂ = {m2:+.6f}  ({m2*100:+.2f}%){END}")
    print(f"{BOLD}  c₂ = {c2:+.6f}{END}")
    print(f"{BOLD}{'='*70}{END}\n")
    
    return {
        'm1': m1, 'c1': c1,
        'm2': m2, 'c2': c2,
        'gamma_est_g1_pos': gamma_est_g1_pos,
        'gamma_est_g1_neg': gamma_est_g1_neg,
        'gamma_est_g2_pos': gamma_est_g2_pos,
        'gamma_est_g2_neg': gamma_est_g2_neg,
        'R_g1_pos': R_g1_pos, 'R_g1_neg': R_g1_neg,
        'R_g2_pos': R_g2_pos, 'R_g2_neg': R_g2_neg
    }

def save_bias_results(bias_results, config, bias_ngmix=None):
    """
    Save multiplicative and additive bias results to file.
    
    Args:
        bias_results: Dict from calculate_multiplicative_bias() containing
                     m1, c1, m2, c2, gamma estimates, and response matrices
        config: Configuration dict with 'plot_path' and 'model_name'
        bias_ngmix: Optional dict with NGmix bias results (same structure)
    """
    plot_path = os.path.join(config['plot_path'], config['model_name'])
    os.makedirs(plot_path, exist_ok=True)
    
    bias_path = os.path.join(plot_path, "bias_calibration.npz")
    
    # Prepare data to save
    save_dict = {
        # Multiplicative and additive biases
        'm1': bias_results['m1'],
        'c1': bias_results['c1'],
        'm2': bias_results['m2'],
        'c2': bias_results['c2'],
        
        # Estimated shears
        'gamma_est_g1_pos': bias_results['gamma_est_g1_pos'],
        'gamma_est_g1_neg': bias_results['gamma_est_g1_neg'],
        'gamma_est_g2_pos': bias_results['gamma_est_g2_pos'],
        'gamma_est_g2_neg': bias_results['gamma_est_g2_neg'],
        
        # Response matrices for each dataset
        'R_g1_pos': bias_results['R_g1_pos'],
        'R_g1_neg': bias_results['R_g1_neg'],
        'R_g2_pos': bias_results['R_g2_pos'],
        'R_g2_neg': bias_results['R_g2_neg'],
    }
    
    # Add NGmix results if provided
    if bias_ngmix is not None:
        save_dict.update({
            'm1_ngmix': bias_ngmix['m1'],
            'c1_ngmix': bias_ngmix['c1'],
            'm2_ngmix': bias_ngmix['m2'],
            'c2_ngmix': bias_ngmix['c2'],
            'gamma_est_g1_pos_ngmix': bias_ngmix['gamma_est_g1_pos'],
            'gamma_est_g1_neg_ngmix': bias_ngmix['gamma_est_g1_neg'],
            'gamma_est_g2_pos_ngmix': bias_ngmix['gamma_est_g2_pos'],
            'gamma_est_g2_neg_ngmix': bias_ngmix['gamma_est_g2_neg'],
            'R_g1_pos_ngmix': bias_ngmix['R_g1_pos'],
            'R_g1_neg_ngmix': bias_ngmix['R_g1_neg'],
            'R_g2_pos_ngmix': bias_ngmix['R_g2_pos'],
            'R_g2_neg_ngmix': bias_ngmix['R_g2_neg'],
        })
    
    # Save to file
    np.savez(bias_path, **save_dict)
    
    print(f"\n{GREEN}Bias calibration results saved to: {bias_path}{END}")
    
    # Print summary of what was saved
    print(f"{CYAN}Saved parameters:{END}")
    print(f"  • Multiplicative biases: m₁ = {bias_results['m1']:+.6f}, m₂ = {bias_results['m2']:+.6f}")
    print(f"  • Additive biases: c₁ = {bias_results['c1']:+.6f}, c₂ = {bias_results['c2']:+.6f}")
    if bias_ngmix is not None:
        print(f"  • NGmix multiplicative biases: m₁ = {bias_ngmix['m1']:+.6f}, m₂ = {bias_ngmix['m2']:+.6f}")
        print(f"  • NGmix additive biases: c₁ = {bias_ngmix['c1']:+.6f}, c₂ = {bias_ngmix['c2']:+.6f}")

def calculate_multiplicative_bias_ngmix(obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
                                       true_shear_step=0.02, h=0.01,
                                       seed=1234, psf_model='gauss', gal_model='gauss'):
    """
    Calculate multiplicative (m) and additive (c) bias for NGmix estimator.
    
    Uses the two-dataset method with NGmix predictions:
    - Run NGmix on datasets at γ_true = ±shear_step for each component
    - Calculate response matrix and mean ellipticity for each
    - Solve: γ_est = (1+m)γ_true + c
    
    Args:
        obs_g1_pos: Observations with base_shear_g1 = +shear_step
        obs_g1_neg: Observations with base_shear_g1 = -shear_step  
        obs_g2_pos: Observations with base_shear_g2 = +shear_step
        obs_g2_neg: Observations with base_shear_g2 = -shear_step
        true_shear_step: Applied shear magnitude (default 0.02)
        h: Response matrix perturbation size (default 0.01)
        seed: Random seed for NGmix
        psf_model: PSF model for NGmix
        gal_model: Galaxy model for NGmix
    
    Returns:
        dict: Same structure as calculate_multiplicative_bias
    """
    from ..methods.ngmix import _get_priors, mp_fit_one, ngmix_pred
    
    print(f"\n{BOLD}{'='*70}")
    print("CALCULATING MULTIPLICATIVE AND ADDITIVE BIAS (NGMIX)")
    print(f"{'='*70}{END}")
    print(f"True shear: ±{true_shear_step}")
    print(f"Response perturbation: ±{h}")
    
    prior = _get_priors(seed)
    rng = np.random.RandomState(seed)
    
    # ========== Component 1 (g1) ==========
    print(f"\n{BOLD}{CYAN}--- Component 1 (g1) ---{END}")
    
    # Calculate response matrices
    print(f"\nDataset A (g1 = +{true_shear_step}):")
    R_g1_pos, _ = calculate_ngmix_response_matrix(
        obs_g1_pos, None, h=h, seed=seed,
        psf_model=psf_model, gal_model=gal_model
    )
    
    print(f"\nDataset B (g1 = -{true_shear_step}):")
    R_g1_neg, _ = calculate_ngmix_response_matrix(
        obs_g1_neg, None, h=h, seed=seed,
        psf_model=psf_model, gal_model=gal_model
    )
    
    # Get mean ellipticities for each dataset using NGmix
    def get_gamma_est_ngmix(observations, response_matrix, component_idx):
        """Calculate γ_est = ⟨e⟩ / ⟨R_ii⟩ using NGmix predictions"""
        # Run NGmix on original (non-perturbed) images
        data = mp_fit_one(observations, prior, rng, psf_model=psf_model, gal_model=gal_model)
        preds = ngmix_pred(data)
        
        # Remove NaNs
        valid_mask = ~np.isnan(preds[:, component_idx])
        preds_valid = preds[valid_mask]
        
        if len(preds_valid) == 0:
            print(f"{RED}Warning: All predictions are NaN for component {component_idx+1}{END}")
            return 0.0, 0.0, 1.0
        
        mean_e = np.mean(preds_valid[:, component_idx])
        R_diag = response_matrix[component_idx, component_idx]
        return float(mean_e / R_diag), float(mean_e), float(R_diag)
    
    gamma_est_g1_pos, mean_e1_pos, R11_pos = get_gamma_est_ngmix(obs_g1_pos, R_g1_pos, 0)
    gamma_est_g1_neg, mean_e1_neg, R11_neg = get_gamma_est_ngmix(obs_g1_neg, R_g1_neg, 0)
    
    # Calculate m1 and c1
    m1 = (gamma_est_g1_pos - gamma_est_g1_neg) / (2 * true_shear_step) - 1
    c1 = (gamma_est_g1_pos + gamma_est_g1_neg) / 2
    
    print(f"\n{CYAN}Summary for g1:{END}")
    print(f"  Dataset A: ⟨e₁⟩ = {mean_e1_pos:+.6f}, ⟨R₁₁⟩ = {R11_pos:.6f}, γ₁ᵉˢᵗ = {gamma_est_g1_pos:+.6f}")
    print(f"  Dataset B: ⟨e₁⟩ = {mean_e1_neg:+.6f}, ⟨R₁₁⟩ = {R11_neg:.6f}, γ₁ᵉˢᵗ = {gamma_est_g1_neg:+.6f}")
    
    # ========== Component 2 (g2) ==========
    print(f"\n{BOLD}{CYAN}--- Component 2 (g2) ---{END}")
    
    # Calculate response matrices
    print(f"\nDataset C (g2 = +{true_shear_step}):")
    R_g2_pos, _ = calculate_ngmix_response_matrix(
        obs_g2_pos, None, h=h, seed=seed,
        psf_model=psf_model, gal_model=gal_model
    )
    
    print(f"\nDataset D (g2 = -{true_shear_step}):")
    R_g2_neg, _ = calculate_ngmix_response_matrix(
        obs_g2_neg, None, h=h, seed=seed,
        psf_model=psf_model, gal_model=gal_model
    )
    
    gamma_est_g2_pos, mean_e2_pos, R22_pos = get_gamma_est_ngmix(obs_g2_pos, R_g2_pos, 1)
    gamma_est_g2_neg, mean_e2_neg, R22_neg = get_gamma_est_ngmix(obs_g2_neg, R_g2_neg, 1)
    
    # Calculate m2 and c2
    m2 = (gamma_est_g2_pos - gamma_est_g2_neg) / (2 * true_shear_step) - 1
    c2 = (gamma_est_g2_pos + gamma_est_g2_neg) / 2
    
    print(f"\n{CYAN}Summary for g2:{END}")
    print(f"  Dataset C: ⟨e₂⟩ = {mean_e2_pos:+.6f}, ⟨R₂₂⟩ = {R22_pos:.6f}, γ₂ᵉˢᵗ = {gamma_est_g2_pos:+.6f}")
    print(f"  Dataset D: ⟨e₂⟩ = {mean_e2_neg:+.6f}, ⟨R₂₂⟩ = {R22_neg:.6f}, γ₂ᵉˢᵗ = {gamma_est_g2_neg:+.6f}")
    
    # ========== Final Results ==========
    print(f"\n{BOLD}{'='*70}")
    print("BIAS CALIBRATION RESULTS (NGMIX)")
    print(f"{'='*70}{END}")
    print(f"\n{BOLD}{YELLOW}Component 1:{END}")
    print(f"{BOLD}  m₁ = {m1:+.6f}  ({m1*100:+.2f}%){END}")
    print(f"{BOLD}  c₁ = {c1:+.6f}{END}")
    print(f"\n{BOLD}{YELLOW}Component 2:{END}")
    print(f"{BOLD}  m₂ = {m2:+.6f}  ({m2*100:+.2f}%){END}")
    print(f"{BOLD}  c₂ = {c2:+.6f}{END}")
    print(f"{BOLD}{'='*70}{END}\n")
    
    return {
        'm1': m1, 'c1': c1,
        'm2': m2, 'c2': c2,
        'gamma_est_g1_pos': gamma_est_g1_pos,
        'gamma_est_g1_neg': gamma_est_g1_neg,
        'gamma_est_g2_pos': gamma_est_g2_pos,
        'gamma_est_g2_neg': gamma_est_g2_neg,
        'R_g1_pos': R_g1_pos, 'R_g1_neg': R_g1_neg,
        'R_g2_pos': R_g2_pos, 'R_g2_neg': R_g2_neg
    }