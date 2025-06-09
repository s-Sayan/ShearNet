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
    
    # Combined loss
    loss = optax.l2_loss(preds, labels).mean()
    
    # Per-label losses
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], labels[:, 1]).mean(),
        'g1g2_combined': optax.l2_loss(preds[:, :2], labels[:, :2]).mean(),  # Combined g1,g2
        'sigma': optax.l2_loss(preds[:, 2], labels[:, 2]).mean(),
        'flux': optax.l2_loss(preds[:, 3], labels[:, 3]).mean()
    }

    return loss, preds, loss_per_label


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

    loss, preds, loss_per_label = loss_fn_ngmix(test_obs, test_labels, seed, 
                                                psf_model=psf_model, gal_model=gal_model)
    
    # Combined metrics
    bias = (preds - test_labels).mean()
    
    # Per-label biases
    bias_per_label = {
        'g1': (preds[:, 0] - test_labels[:, 0]).mean(),
        'g2': (preds[:, 1] - test_labels[:, 1]).mean(),
        'g1g2_combined': (preds[:, :2] - test_labels[:, :2]).mean(),  # Average bias for g1,g2
        'sigma': (preds[:, 2] - test_labels[:, 2]).mean(),
        'flux': (preds[:, 3] - test_labels[:, 3]).mean()
    }
    
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