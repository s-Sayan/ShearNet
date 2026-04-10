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
import ngmix
import galsim
from ngmix.shape import e1e2_to_g1g2

# ANSI color codes for pretty printing
BOLD = '\033[1m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
GREEN = '\033[92m'
END = '\033[0m'

def remove_nan_preds_multi(pred1: np.ndarray, pred2: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = ~np.any(np.isnan(pred1) | np.isnan(pred2), axis=1)
    num_removed = np.sum(~mask)
    if num_removed > 0:
        print(f"[NaN Filter] Removed {num_removed} rows with NaNs in predictions.")
    return pred1[mask], pred2[mask], labels[mask]

def remove_nan_preds(preds: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = ~np.any(np.isnan(preds), axis=1)
    num_removed = np.sum(~mask)
    if num_removed > 0:
        print(f"[NaN Filter] Removed {num_removed} rows with NaNs in predictions.")
    return preds[mask], labels[mask]

def loss_fn_mcal(images, labels, psf_fwhm):
    from ..methods.metacal import mcal_preds
    preds = mcal_preds(images, psf_fwhm)
    loss = optax.l2_loss(preds[:, :2], labels[:, :2]).mean()
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], labels[:, 1]).mean(),
        'g1g2_combined': loss
    }
    return loss, preds, loss_per_label


def loss_fn_ngmix(obs_list, labels, seed=1234, psf_model='gauss', gal_model='gauss'):
    from ..methods.ngmix import _get_priors, mp_fit_one, ngmix_pred
    prior = _get_priors(seed)
    rng = np.random.RandomState(seed)
    datalist = mp_fit_one(obs_list, prior, rng, psf_model=psf_model, gal_model=gal_model)
    preds = ngmix_pred(datalist)
    pred_filtered, labels_filtered = remove_nan_preds(preds, labels)
    pred_filtered = pred_filtered[:, 0:2]
    loss = optax.l2_loss(pred_filtered, labels_filtered).mean()
    bias = (pred_filtered - labels_filtered).mean()
    loss_per_label = {
        'g1': optax.l2_loss(pred_filtered[:, 0], labels_filtered[:, 0]).mean(),
        'g2': optax.l2_loss(pred_filtered[:, 1], labels_filtered[:, 1]).mean(),
        'g1g2_combined': optax.l2_loss(pred_filtered[:, :2], labels_filtered[:, :2]).mean(),
    }
    bias_per_label = {
        'g1': (pred_filtered[:, 0] - labels_filtered[:, 0]).mean(),
        'g2': (pred_filtered[:, 1] - labels_filtered[:, 1]).mean(),
        'g1g2_combined': (pred_filtered[:, :2] - labels_filtered[:, :2]).mean(),
    }
    return loss, preds, loss_per_label, bias, bias_per_label


def eval_ngmix(test_obs, test_labels, seed=1234, psf_model='gauss', gal_model='gauss') -> Dict[str, Any]:
    from ..methods.ngmix import _get_priors, mp_fit_one, ngmix_pred, response_calculation
    start_time = time.time()
    prior = _get_priors(seed)
    rng = np.random.RandomState(seed)
    datalist = mp_fit_one(test_obs, prior, rng, psf_model=psf_model, gal_model=gal_model)
    preds = ngmix_pred(datalist)
    r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list = response_calculation(
        datalist, mcal_shear=0.01
    )
    r11_array = np.array(r11_list)
    r22_array = np.array(r22_list)
    r12_array = np.array(r12_list)
    r21_array = np.array(r21_list)
    valid_mask = np.isfinite(r11_array) & np.isfinite(r22_array) & np.isfinite(r12_array) & np.isfinite(r21_array)
    R = np.array([
        [np.mean(r11_array[valid_mask]), np.mean(r12_array[valid_mask])],
        [np.mean(r21_array[valid_mask]), np.mean(r22_array[valid_mask])]
    ])
    R_per_gal = np.stack([
        np.stack([r11_array, r12_array], axis=1),
        np.stack([r21_array, r22_array], axis=1)
    ], axis=1)
    pred_filtered, labels_filtered = remove_nan_preds(preds, test_labels)
    pred_filtered = pred_filtered[:, 0:2]
    loss = optax.l2_loss(pred_filtered, labels_filtered).mean()
    bias = (pred_filtered - labels_filtered).mean()
    loss_per_label = {
        'g1': optax.l2_loss(pred_filtered[:, 0], labels_filtered[:, 0]).mean(),
        'g2': optax.l2_loss(pred_filtered[:, 1], labels_filtered[:, 1]).mean(),
        'g1g2_combined': optax.l2_loss(pred_filtered[:, :2], labels_filtered[:, :2]).mean(),
    }
    bias_per_label = {
        'g1': (pred_filtered[:, 0] - labels_filtered[:, 0]).mean(),
        'g2': (pred_filtered[:, 1] - labels_filtered[:, 1]).mean(),
        'g1g2_combined': (pred_filtered[:, :2] - labels_filtered[:, :2]).mean(),
    }
    total_time = time.time() - start_time
    print(f"\n{BOLD}=== Combined Metrics (NGmix) ==={END}")
    print(f"Mean Squared Error (MSE): {BOLD}{YELLOW}{loss:.6e}{END}")
    print(f"Average Bias: {BOLD}{YELLOW}{bias:.6e}{END}")
    print(f"Time taken: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    print(f"\n{BOLD}Response Matrix (from metacalibration):{END}")
    print(f"{CYAN}R = [[{R[0,0]:.6f}, {R[0,1]:.6f}],{END}")
    print(f"{CYAN}     [{R[1,0]:.6f}, {R[1,1]:.6f}]]{END}")
    print("\n=== Per-Label Metrics ===")
    for label in ['g1', 'g2', 'g1g2_combined']:
        print(f"{label:>15}: MSE = {loss_per_label[label]:.6e}, Bias = {bias_per_label[label]:+.6e}")
    print()
    return {
        'loss': loss,
        'bias': bias,
        'loss_per_label': loss_per_label,
        'bias_per_label': bias_per_label,
        'preds': preds,
        'time_taken': total_time,
        'R': R,
        'R_per_gal': R_per_gal,
        'datalist': datalist
    }


@jax.jit
def eval_step(state, images, labels):
    preds = state.apply_fn(state.params, images, deterministic=True)
    loss = optax.l2_loss(preds, labels).mean()
    return loss, preds

@jax.jit
def fork_eval_step(state, images, psf_images, labels):
    preds = state.apply_fn(state.params, images, psf_images, deterministic=True)
    loss = optax.l2_loss(preds, labels).mean()
    return loss, preds


def _per_label_metrics(preds, labels, output_keys):
    _ok = list(output_keys)
    loss_per_label = {}
    bias_per_label = {}
    for k in _ok:
        idx = _ok.index(k)
        loss_per_label[k] = float(optax.l2_loss(preds[:, idx], labels[:, idx]).mean())
        bias_per_label[k] = float((preds[:, idx] - labels[:, idx]).mean())
    if 'g1' in _ok and 'g2' in _ok:
        g_idx = [_ok.index('g1'), _ok.index('g2')]
        loss_per_label['g1g2_combined'] = float(optax.l2_loss(preds[:, g_idx], labels[:, g_idx]).mean())
        bias_per_label['g1g2_combined'] = float((preds[:, g_idx] - labels[:, g_idx]).mean())
    return loss_per_label, bias_per_label


def eval_model(state, test_images, test_labels, output_keys=("g1", "g2"), batch_size=32) -> Dict[str, Any]:
    start_time = time.time()
    total_loss = 0
    total_samples = 0
    total_bias = 0
    _label_keys = list(output_keys) + (['g1g2_combined'] if 'g1' in output_keys and 'g2' in output_keys else [])
    total_loss_per_label = {k: 0 for k in _label_keys}
    total_bias_per_label = {k: 0 for k in _label_keys}
    all_preds = []

    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        loss, preds = eval_step(state, batch_images, batch_labels)
        loss_per_label, bias_per_label = _per_label_metrics(preds, batch_labels, output_keys)
        all_preds.append(preds)
        batch_size_actual = len(batch_images)
        total_loss += loss * batch_size_actual
        total_bias += float((preds - batch_labels).mean()) * batch_size_actual
        total_samples += batch_size_actual
        for label in total_loss_per_label:
            total_loss_per_label[label] += loss_per_label[label] * batch_size_actual
            total_bias_per_label[label] += bias_per_label[label] * batch_size_actual

    avg_loss = total_loss / total_samples
    avg_bias = total_bias / total_samples
    avg_loss_per_label = {label: total / total_samples for label, total in total_loss_per_label.items()}
    avg_bias_per_label = {label: total / total_samples for label, total in total_bias_per_label.items()}
    total_time = time.time() - start_time

    print(f"\n{BOLD}=== Combined Metrics (ShearNet) ==={END}")
    print(f"Mean Squared Error (MSE) from ShearNet: {BOLD}{YELLOW}{avg_loss:.6e}{END}")
    print(f"Average Bias from ShearNet: {BOLD}{YELLOW}{avg_bias:.6e}{END}")
    print(f"Time taken: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    print("\n=== Per-Label Metrics ===")
    for label in _label_keys:
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

def fork_eval_model(state, test_images, test_psf_images, test_labels, output_keys=("g1", "g2"), batch_size=32) -> Dict[str, Any]:
    start_time = time.time()
    total_loss = 0
    total_samples = 0
    total_bias = 0
    _label_keys = list(output_keys) + (['g1g2_combined'] if 'g1' in output_keys and 'g2' in output_keys else [])
    total_loss_per_label = {k: 0 for k in _label_keys}
    total_bias_per_label = {k: 0 for k in _label_keys}
    all_preds = []

    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_psf_images = test_psf_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        loss, preds = fork_eval_step(state, batch_images, batch_psf_images, batch_labels)
        loss_per_label, bias_per_label = _per_label_metrics(preds, batch_labels, output_keys)
        all_preds.append(preds)
        batch_size_actual = len(batch_images)
        total_loss += loss * batch_size_actual
        total_bias += float((preds - batch_labels).mean()) * batch_size_actual
        total_samples += batch_size_actual
        for label in total_loss_per_label:
            total_loss_per_label[label] += loss_per_label[label] * batch_size_actual
            total_bias_per_label[label] += bias_per_label[label] * batch_size_actual

    avg_loss = total_loss / total_samples
    avg_bias = total_bias / total_samples
    avg_loss_per_label = {label: total / total_samples for label, total in total_loss_per_label.items()}
    avg_bias_per_label = {label: total / total_samples for label, total in total_bias_per_label.items()}
    total_time = time.time() - start_time

    print(f"\n{BOLD}=== Combined Metrics (ShearNet) ==={END}")
    print(f"Mean Squared Error (MSE) from ShearNet: {BOLD}{YELLOW}{avg_loss:.6e}{END}")
    print(f"Average Bias from ShearNet: {BOLD}{YELLOW}{avg_bias:.6e}{END}")
    print(f"Time taken: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    print("\n=== Per-Label Metrics ===")
    for label in _label_keys:
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
    import jax.numpy as jnp
    print(f"\n{BOLD}Calculating Response Matrix...{END}")
    n_gal = len(observations)
    e1_positive_images = np.array([obs.meta['e1_positive'] for obs in observations])
    e1_negative_images = np.array([obs.meta['e1_negative'] for obs in observations])
    e2_positive_images = np.array([obs.meta['e2_positive'] for obs in observations])
    e2_negative_images = np.array([obs.meta['e2_negative'] for obs in observations])
    print(f"Extracted {n_gal} galaxies with sheared images")
    print(f"Shear step: ±{h}")

    def get_predictions(images, psf_imgs=None):
        preds = []
        for i in range(0, n_gal, batch_size):
            batch_slice = slice(i, min(i + batch_size, n_gal))
            if model_type == 'fork':
                batch_pred = state.apply_fn(state.params, images[batch_slice], psf_imgs[batch_slice], deterministic=True)
            else:
                batch_pred = state.apply_fn(state.params, images[batch_slice], deterministic=True)
            preds.append(batch_pred)
        return jnp.concatenate(preds)

    preds_e1p = get_predictions(e1_positive_images, psf_images)
    preds_e1m = get_predictions(e1_negative_images, psf_images)
    preds_e2p = get_predictions(e2_positive_images, psf_images)
    preds_e2m = get_predictions(e2_negative_images, psf_images)

    R_11 = (preds_e1p[:, 0] - preds_e1m[:, 0]) / (2 * h)
    R_12 = (preds_e2p[:, 0] - preds_e2m[:, 0]) / (2 * h)
    R_21 = (preds_e1p[:, 1] - preds_e1m[:, 1]) / (2 * h)
    R_22 = (preds_e2p[:, 1] - preds_e2m[:, 1]) / (2 * h)

    R_per_galaxy = jnp.stack([
        jnp.stack([R_11, R_12], axis=1),
        jnp.stack([R_21, R_22], axis=1)
    ], axis=1)
    R = jnp.mean(R_per_galaxy, axis=0)

    print(f"\n{BOLD}Response Matrix (averaged over {n_gal} galaxies):{END}")
    print(f"{CYAN}R = [[{R[0,0]:.6f}, {R[0,1]:.6f}],{END}")
    print(f"{CYAN}     [{R[1,0]:.6f}, {R[1,1]:.6f}]]{END}")
    print(f"\nResponse matrix statistics:")
    print(f"  R_11: mean={float(jnp.mean(R_11)):.6f}, std={float(jnp.std(R_11)):.6f}")
    print(f"  R_22: mean={float(jnp.mean(R_22)):.6f}, std={float(jnp.std(R_22)):.6f}")
    print(f"  R_12: mean={float(jnp.mean(R_12)):.6f}, std={float(jnp.std(R_12)):.6f}")
    print(f"  R_21: mean={float(jnp.mean(R_21)):.6f}, std={float(jnp.std(R_21)):.6f}")

    return R, R_per_galaxy


def calculate_multiplicative_bias(state, obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
                                  true_shear_step=0.02, batch_size=32, h=0.01,
                                  model_type='standard',
                                  psf_g1_pos=None, psf_g1_neg=None,
                                  psf_g2_pos=None, psf_g2_neg=None,
                                  R=None):
    import jax.numpy as jnp
    print(f"\n{BOLD}{'='*70}")
    print("CALCULATING MULTIPLICATIVE AND ADDITIVE BIAS")
    print(f"{'='*70}{END}")
    print(f"True shear: ±{true_shear_step}")
    print(f"Response perturbation: ±{h}")
    print(f"\n{BOLD}{CYAN}--- Component 1 (g1) ---{END}")
    print(f"\nDataset A (g1 = +{true_shear_step}):")
    if R is None:
        R, _ = calculate_response_matrix(state, obs_g1_pos, batch_size=batch_size, h=h, model_type=model_type, psf_images=psf_g1_pos)
    print(f"\nDataset B (g1 = -{true_shear_step}):")

    def get_gamma_est(observations, response_matrix, component_idx, psf_imgs=None):
        n_gal = len(observations)
        images = np.array([obs.image for obs in observations])
        preds = []
        for i in range(0, n_gal, batch_size):
            batch_slice = slice(i, min(i + batch_size, n_gal))
            if model_type == 'fork':
                batch_pred = state.apply_fn(state.params, images[batch_slice], psf_imgs[batch_slice], deterministic=True)
            else:
                batch_pred = state.apply_fn(state.params, images[batch_slice], deterministic=True)
            preds.append(batch_pred)
        preds = jnp.concatenate(preds)
        mean_e = jnp.mean(preds[:, component_idx])
        R_diag = response_matrix[component_idx, component_idx]
        return float(mean_e / R_diag), float(mean_e), float(R_diag)

    gamma_est_g1_pos, mean_e1_pos, R11_pos = get_gamma_est(obs_g1_pos, R, 0, psf_g1_pos)
    gamma_est_g1_neg, mean_e1_neg, R11_neg = get_gamma_est(obs_g1_neg, R, 0, psf_g1_neg)
    m1 = (gamma_est_g1_pos - gamma_est_g1_neg) / (2 * true_shear_step) - 1
    c1 = (gamma_est_g1_pos + gamma_est_g1_neg) / 2
    print(f"\n{CYAN}Summary for g1:{END}")
    print(f"  Dataset A: ⟨e₁⟩ = {mean_e1_pos:+.6f}, ⟨R₁₁⟩ = {R11_pos:.6f}, γ₁ᵉˢᵗ = {gamma_est_g1_pos:+.6f}")
    print(f"  Dataset B: ⟨e₁⟩ = {mean_e1_neg:+.6f}, ⟨R₁₁⟩ = {R11_neg:.6f}, γ₁ᵉˢᵗ = {gamma_est_g1_neg:+.6f}")
    print(f"\n{BOLD}{CYAN}--- Component 2 (g2) ---{END}")
    print(f"\nDataset C (g2 = +{true_shear_step}):")
    print(f"\nDataset D (g2 = -{true_shear_step}):")
    gamma_est_g2_pos, mean_e2_pos, R22_pos = get_gamma_est(obs_g2_pos, R, 1, psf_g2_pos)
    gamma_est_g2_neg, mean_e2_neg, R22_neg = get_gamma_est(obs_g2_neg, R, 1, psf_g2_neg)
    m2 = (gamma_est_g2_pos - gamma_est_g2_neg) / (2 * true_shear_step) - 1
    c2 = (gamma_est_g2_pos + gamma_est_g2_neg) / 2
    print(f"\n{CYAN}Summary for g2:{END}")
    print(f"  Dataset C: ⟨e₂⟩ = {mean_e2_pos:+.6f}, ⟨R₂₂⟩ = {R22_pos:.6f}, γ₂ᵉˢᵗ = {gamma_est_g2_pos:+.6f}")
    print(f"  Dataset D: ⟨e₂⟩ = {mean_e2_neg:+.6f}, ⟨R₂₂⟩ = {R22_neg:.6f}, γ₂ᵉˢᵗ = {gamma_est_g2_neg:+.6f}")
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
        'R': R
    }


def calculate_multiplicative_bias_ngmix(obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
                                       true_shear_step=0.02, h=0.01,
                                       seed=1234, psf_model='gauss', gal_model='gauss'):
    from ..methods.ngmix import _get_priors, mp_fit_one, ngmix_pred, response_calculation
    print(f"\n{BOLD}{'='*70}")
    print("CALCULATING MULTIPLICATIVE AND ADDITIVE BIAS (NGMIX)")
    print(f"{'='*70}{END}")
    print(f"True shear: ±{true_shear_step}")
    print(f"Response perturbation: ±{h}")
    prior = _get_priors(seed)
    rng = np.random.RandomState(seed)

    def run_ngmix_with_response(observations, component_idx):
        datalist = mp_fit_one(observations, prior, rng, psf_model=psf_model, gal_model=gal_model)
        preds = ngmix_pred(datalist)
        r11_list, r22_list, r12_list, r21_list, _, _, _, _ = response_calculation(datalist, mcal_shear=h)
        r11_array = np.array(r11_list)
        r22_array = np.array(r22_list)
        r12_array = np.array(r12_list)
        r21_array = np.array(r21_list)
        valid_mask = np.isfinite(r11_array) & np.isfinite(r22_array) & np.isfinite(r12_array) & np.isfinite(r21_array)
        R = np.array([
            [np.mean(r11_array[valid_mask]), np.mean(r12_array[valid_mask])],
            [np.mean(r21_array[valid_mask]), np.mean(r22_array[valid_mask])]
        ])
        valid_preds = ~np.isnan(preds[:, component_idx])
        mean_e = np.mean(preds[valid_preds, component_idx])
        R_diag = R[component_idx, component_idx]
        gamma_est = mean_e / R_diag
        return gamma_est, mean_e, R_diag, R

    print(f"\n{BOLD}{CYAN}--- Component 1 (g1) ---{END}")
    print(f"Dataset A (g1 = +{true_shear_step}):")
    gamma_est_g1_pos, mean_e1_pos, R11_pos, R_pos = run_ngmix_with_response(obs_g1_pos, 0)
    print(f"Dataset B (g1 = -{true_shear_step}):")
    gamma_est_g1_neg, mean_e1_neg, R11_neg, R_neg = run_ngmix_with_response(obs_g1_neg, 0)
    R = 0.5 * (R_pos + R_neg)
    m1 = (gamma_est_g1_pos - gamma_est_g1_neg) / (2 * true_shear_step) - 1
    c1 = (gamma_est_g1_pos + gamma_est_g1_neg) / 2
    print(f"\n{CYAN}Summary for g1:{END}")
    print(f"  Dataset A: ⟨e₁⟩ = {mean_e1_pos:+.6f}, ⟨R₁₁⟩ = {R11_pos:.6f}, γ₁ᵉˢᵗ = {gamma_est_g1_pos:+.6f}")
    print(f"  Dataset B: ⟨e₁⟩ = {mean_e1_neg:+.6f}, ⟨R₁₁⟩ = {R11_neg:.6f}, γ₁ᵉˢᵗ = {gamma_est_g1_neg:+.6f}")
    print(f"\n{BOLD}{CYAN}--- Component 2 (g2) ---{END}")
    print(f"Dataset C (g2 = +{true_shear_step}):")
    gamma_est_g2_pos, mean_e2_pos, R22_pos, _ = run_ngmix_with_response(obs_g2_pos, 1)
    print(f"Dataset D (g2 = -{true_shear_step}):")
    gamma_est_g2_neg, mean_e2_neg, R22_neg, _ = run_ngmix_with_response(obs_g2_neg, 1)
    m2 = (gamma_est_g2_pos - gamma_est_g2_neg) / (2 * true_shear_step) - 1
    c2 = (gamma_est_g2_pos + gamma_est_g2_neg) / 2
    print(f"\n{CYAN}Summary for g2:{END}")
    print(f"  Dataset C: ⟨e₂⟩ = {mean_e2_pos:+.6f}, ⟨R₂₂⟩ = {R22_pos:.6f}, γ₂ᵉˢᵗ = {gamma_est_g2_pos:+.6f}")
    print(f"  Dataset D: ⟨e₂⟩ = {mean_e2_neg:+.6f}, ⟨R₂₂⟩ = {R22_neg:.6f}, γ₂ᵉˢᵗ = {gamma_est_g2_neg:+.6f}")
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
        'R': R,
    }

def get_admoms_ngmix_fit(obs: "ngmix.Observation", reduced: bool = True) -> dict:
    jac = obs._jacobian
    scale = jac.get_scale()
    image = obs.image
    norm = np.sum(image[image > 0])
    if norm <= 0:
        return {"e1": np.nan, "e2": np.nan, "T": np.nan, "flags": 1}
    obs_norm = ngmix.Observation(image=image / norm, jacobian=jac)
    am = ngmix.admom.AdmomFitter()
    res = am.go(obs_norm, guess=0.5)
    e1, e2, T_ngmix = res["e1"], res["e2"], res["T"]
    gal_image = galsim.Image(image / norm, scale=scale)
    admoms = galsim.hsm.FindAdaptiveMom(gal_image)
    sigma = admoms.moments_sigma * scale
    T_galsim = 2 * sigma**2
    flag = 0 if (admoms.moments_status == 0 and res["flags"] == 0) else 1
    if reduced:
        e1, e2 = e1e2_to_g1g2(e1, e2)
    return {"e1": e1, "e2": e2, "T": T_galsim, "flags": flag}