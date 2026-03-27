"""Command-line interface for evaluating trained ShearNet models."""

import os
import argparse
import jax.random as random
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.table import Table
import optax
import time
from flax.training import checkpoints, train_state
from datetime import datetime

from ..config.config_handler import Config
from ..core.dataset import generate_dataset, split_combined_images
from ..core.models import (
    SimpleGalaxyNN, EnhancedGalaxyNN, GalaxyResNet, 
    ResearchBackedGalaxyResNet, ForkLensPSFNet, ForkLike
)
from ..utils.metrics import (
    eval_model, fork_eval_model, eval_ngmix, 
    remove_nan_preds_multi, calculate_response_matrix
)
from ..utils.plot_helpers import (
    plot_residuals, 
    visualize_galaxy_samples,
    visualize_psf_samples, 
    plot_true_vs_predicted, 
    animate_model_epochs,
    plot_psf_systematics_from_eval
)

# ANSI color codes
BOLD = '\033[1m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'


def create_parser():
    """Create argument parser for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained galaxy shear estimation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate using saved model config
  shearnet-eval --model_name cnn6
  
  # Override test samples
  shearnet-eval --model_name cnn6 --test_samples 5000
  
  # Enable comparison methods and plotting
  shearnet-eval --model_name cnn6 --mcal --plot
  
  # Override random seed for different test set
  shearnet-eval --model_name cnn6 --seed 123 --plot
        """
    )
    
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Name of the model to load.')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for test data generation (overrides config).')
    parser.add_argument('--test_samples', type=int, default=None, 
                       help='Number of test samples (overrides config).')
    parser.add_argument('--mcal', action='store_true', 
                       help='Compare with metacalibration and NGmix')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate all plots')
    parser.add_argument('--plot_animation', action='store_true', 
                       help='Plot animation of scatter plot.')
    parser.add_argument('--process_psf', action='store_const', const=True, default=None, 
                       help='Process psf images on separate CNN branch.')
    parser.add_argument('--galaxy_type', type=str, default=None, 
                       help='Galaxy model type for fork-like models')
    parser.add_argument('--psf_type', type=str, default=None,
                       help='PSF model type for fork-like models')
    parser.add_argument('--apply_psf_shear', action='store_const', const=True, default=None,
                       help='Apply random shear to PSF images')
    parser.add_argument('--psf_shear_range', type=float, default=None,
                       help='Maximum absolute shear value for PSF (default: 0.05)')
    
    return parser


def load_config(args):
    """Load and validate configuration."""
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')

    os.makedirs(default_save_path, exist_ok=True)
    os.makedirs(default_plot_path, exist_ok=True)
    
    model_name = args.model_name
    model_config_path = os.path.join(default_plot_path, model_name, 'training_config.yaml')
    
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"No training config found at {model_config_path}")
    
    print(f"\n{BOLD}Loading model config from: {model_config_path}{END}")
    config = Config(model_config_path)
    config.print_eval_config()
    
    # Get values from config with command-line overrides
    eval_config = {
        'seed': args.seed if args.seed is not None else config.get('evaluation.seed', config.get('dataset.seed')),
        'test_samples': args.test_samples if args.test_samples is not None else config.get('evaluation.test_samples'),
        'process_psf': args.process_psf if args.process_psf is not None else config.get('model.process_psf'),
        'nn': config.get('model.type'),
        'galaxy_type': args.galaxy_type if args.galaxy_type else config.get('model.galaxy.type'),
        'psf_type': args.psf_type if args.psf_type else config.get('model.psf.type'),
        'psf_sigma': config.get('dataset.psf_sigma'),
        'nse_sd': config.get('dataset.nse_sd'),
        'exp': config.get('dataset.exp'),
        'stamp_size': config.get('dataset.stamp_size'),
        'pixel_size': config.get('dataset.pixel_size'),
        'apply_psf_shear': args.apply_psf_shear if args.apply_psf_shear is not None else config.get('dataset.apply_psf_shear', False),
        'psf_shear_range': args.psf_shear_range if args.psf_shear_range is not None else config.get('dataset.psf_shear_range', 0.05),
        'mcal': args.mcal or config.get('comparison.ngmix', False),
        'plot': args.plot or config.get('plotting.plot', False),
        'plot_animation': args.plot_animation or config.get('plotting.animation', False),
        'model_name': model_name,
        'save_path': default_save_path,
        'plot_path': default_plot_path,
        'dataset_type': config.get('dataset.type', 'gauss'),
        'psf_model': config.get('comparison.psf_model', 'gauss'),
    }
    
    return eval_config


def generate_test_data(config):
    """Generate test dataset."""
    print(f"\n{BOLD}{'='*50}")
    print("Generating Test Dataset")
    print(f"{'='*50}{END}")
    
    test_images, test_labels, test_obs = generate_dataset(
        config['test_samples'], 
        config['psf_sigma'], 
        exp=config['exp'], 
        seed=config['seed'], 
        nse_sd=config['nse_sd'], 
        npix=config['stamp_size'], 
        scale=config['pixel_size'], 
        return_psf=config['process_psf'], 
        return_obs=True,
        apply_psf_shear=config['apply_psf_shear'], 
        psf_shear_range=config['psf_shear_range']
    )
    
    # Extract SNR values
    snr_values = np.array([obs.meta.get('snr', obs.get_s2n()) for obs in test_obs])
    
    # Split images if using fork model
    if config['process_psf']:
        test_galaxy_images, test_psf_images = split_combined_images(
            test_images, has_psf=True, has_clean=False
        )
        print(f"Shape of test galaxy images: {test_galaxy_images.shape}")
        print(f"Shape of test PSF images: {test_psf_images.shape}")
    else:
        test_galaxy_images = test_images
        test_psf_images = None
        print(f"Shape of test images: {test_images.shape}")
    
    print(f"Shape of test labels: {test_labels.shape}")
    print(f"SNR range: [{snr_values.min():.2f}, {snr_values.max():.2f}], mean: {snr_values.mean():.2f}")
    
    return test_galaxy_images, test_psf_images, test_labels, test_obs, snr_values


def initialize_model(config, test_galaxy_images, test_psf_images):
    """Initialize model and load checkpoint."""
    print(f"\n{BOLD}{'='*50}")
    print("Loading Model")
    print(f"{'='*50}{END}")
    
    rng_key = random.PRNGKey(config['seed'])
    
    # Select model
    nn = config['nn']
    if nn == "mlp":
        model = SimpleGalaxyNN()
    elif nn == "cnn":
        model = EnhancedGalaxyNN()
    elif nn == "resnet":
        model = GalaxyResNet()
    elif nn == "research_backed":
        model = ResearchBackedGalaxyResNet()
    elif nn == "forklens_psfnet":
        model = ForkLensPSFNet()
    elif nn == "fork-like":
        model = ForkLike(galaxy_model_type=config['galaxy_type'], psf_model_type=config['psf_type'])
    else:
        raise ValueError(f"Invalid model type specified: {nn}")
    
    # Initialize parameters
    if config['process_psf']:
        init_params = model.init(rng_key, jnp.ones_like(test_galaxy_images[0]), jnp.ones_like(test_psf_images[0]))
    else:
        init_params = model.init(rng_key, jnp.ones_like(test_galaxy_images[0]))

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adam(1e-3)
    )

    # Find checkpoint
    load_path = os.path.abspath(config['save_path'])
    matching_dirs = [
        d for d in os.listdir(load_path) 
        if os.path.isdir(os.path.join(load_path, d)) and d.startswith(config['model_name'])
    ]

    if not matching_dirs:
        raise FileNotFoundError(f"No directory found in {load_path} starting with '{config['model_name']}'.")

    print(f"Found {len(matching_dirs)} matching checkpoint(s):")
    for idx, directory in enumerate(matching_dirs, start=1):
        print(f"  {idx}. {directory}")
    
    # Use most recent checkpoint
    if len(matching_dirs) == 1:
        model_dir = os.path.join(load_path, matching_dirs[0])
    else:
        model_dir = os.path.join(load_path, sorted(matching_dirs)[-1])

    print(f"\n{GREEN}Loading checkpoint from: {model_dir}{END}")
    state = checkpoints.restore_checkpoint(ckpt_dir=model_dir, target=state)
    print(f"{GREEN}✓ Model checkpoint loaded successfully{END}")
    
    return state


def evaluate_neural_network(state, test_galaxy_images, test_psf_images, test_labels, config):
    """Evaluate neural network model."""
    print(f"\n{BOLD}{'='*50}")
    print("Evaluating Neural Network")
    print(f"{'='*50}{END}")
    
    if config['process_psf']:
        nn_results = fork_eval_model(state, test_galaxy_images, test_psf_images, test_labels)
    else:
        nn_results = eval_model(state, test_galaxy_images, test_labels)
    
    return nn_results


def evaluate_comparison_methods(test_obs, test_galaxy_images, test_labels, config):
    """Evaluate NGmix and metacalibration if requested."""
    ngmix_results = None
    
    if config['mcal']:
        print(f"\n{BOLD}{'='*50}")
        print("Evaluating Comparison Methods")
        print(f"{'='*50}{END}")

        # Get model types from config
        dataset_type = config.get('dataset_type', 'gauss')
        psf_model = config.get('psf_model', 'gauss')
        
        # NGmix
        ngmix_results = eval_ngmix(
            test_obs, test_labels, seed=config['seed'], 
            psf_model=psf_model, gal_model=dataset_type
        )
    
    return ngmix_results


def calculate_response_matrices(state, test_obs, test_galaxy_images, test_psf_images, test_labels, config, ngmix_results):
    """Calculate shear response matrices for calibration."""
    print(f"\n{BOLD}{'='*50}")
    print("Response Matrices")
    print(f"{'='*50}{END}")
    
    # Neural network response matrix
    if config['process_psf']:
        R_nn, R_per_gal_nn = calculate_response_matrix(
            state, test_obs, batch_size=32, h=0.01,
            model_type='fork', psf_images=test_psf_images
        )
    else:
        R_nn, R_per_gal_nn = calculate_response_matrix(
            state, test_obs, batch_size=32, h=0.01,
            model_type='standard'
        )
    
    # Extract NGmix response matrix (already calculated in eval_ngmix!)
    R_ngmix = None
    R_per_gal_ngmix = None
    
    if ngmix_results is not None:
        R_ngmix = ngmix_results.get('R')
        R_per_gal_ngmix = ngmix_results.get('R_per_gal')
        
        print(f"\n{BOLD}Response Matrix from NGmix (already computed):{END}")
        print(f"{CYAN}R_ngmix = [[{R_ngmix[0,0]:.6f}, {R_ngmix[0,1]:.6f}],{END}")
        print(f"{CYAN}          [{R_ngmix[1,0]:.6f}, {R_ngmix[1,1]:.6f}]]{END}")
        
        # Compare response matrices
        print(f"\n{BOLD}Response Matrix Comparison:{END}")
        print(f"\n{CYAN}ShearNet R matrix:{END}")
        print(f"  [[{R_nn[0,0]:+.6f}, {R_nn[0,1]:+.6f}],")
        print(f"   [{R_nn[1,0]:+.6f}, {R_nn[1,1]:+.6f}]]")
        print(f"\n{CYAN}NGmix R matrix:{END}")
        print(f"  [[{R_ngmix[0,0]:+.6f}, {R_ngmix[0,1]:+.6f}],")
        print(f"   [{R_ngmix[1,0]:+.6f}, {R_ngmix[1,1]:+.6f}]]")
        
        # Calculate differences
        R_diff = R_nn - R_ngmix
        print(f"\n{YELLOW}Difference (ShearNet - NGmix):{END}")
        print(f"  [[{R_diff[0,0]:+.6f}, {R_diff[0,1]:+.6f}],")
        print(f"   [{R_diff[1,0]:+.6f}, {R_diff[1,1]:+.6f}]]")
    
    return R_nn, R_per_gal_nn, R_ngmix, R_per_gal_ngmix


def generate_bias_datasets_and_evaluate(state, config):
    """
    Generate four bias calibration datasets and get predictions + responses for both NN and NGmix.
    
    Returns:
        tuple: (bias_datasets, shear_step) where bias_datasets has structure:
            {
                'g1_pos': {
                    'obs': ..., 
                    'nn_preds': ..., 
                    'nn_R_per_gal': ...,
                    'ngmix_preds': ...,
                    'ngmix_R_per_gal': ...,
                    'psf_images': ...
                },
                ...
            }
    """
    from ..methods.ngmix import _get_priors, mp_fit_one, ngmix_pred, response_calculation
    
    print(f"\n{BOLD}{'='*50}")
    print("Generating Bias Calibration Datasets")
    print(f"{'='*50}{END}")
    
    shear_step = 0.02
    print(f"Shear step: ±{shear_step}")
    print(f"Samples per dataset: {config['test_samples']}")
    
    # Get NGmix parameters
    dataset_type = config.get('dataset_type', 'gauss')
    psf_model = config.get('psf_model', 'gauss')
    prior = _get_priors(config['seed'])
    
    bias_datasets = {}
    
    # Helper function to evaluate both NN and NGmix
    def evaluate_dataset(images, obs, psf_images, model_type, batch_size=128):
        """Evaluate dataset with both NN and NGmix."""
        # NN evaluation
        if model_type == 'fork':
            _, R_per_gal_nn = calculate_response_matrix(
                state, obs, batch_size=32, h=0.01,
                model_type='fork', psf_images=psf_images
            )
            preds_list = []
            for i in range(0, len(images), batch_size):
                batch_preds = state.apply_fn(
                    state.params,
                    images[i:i+batch_size],
                    psf_images[i:i+batch_size],
                    deterministic=True,
                )
                preds_list.append(batch_preds)
            preds_nn = jnp.concatenate(preds_list, axis=0)
        else:
            _, R_per_gal_nn = calculate_response_matrix(
                state, obs, batch_size=32, h=0.01, model_type='standard'
            )
            preds_list = []
            for i in range(0, len(images), batch_size):
                batch_preds = state.apply_fn(
                    state.params,
                    images[i:i+batch_size],
                    deterministic=True,
                )
                preds_list.append(batch_preds)
            preds_nn = jnp.concatenate(preds_list, axis=0)
        
        # NGmix evaluation
        rng_local = np.random.RandomState(config['seed'])
        datalist = mp_fit_one(obs, prior, rng_local, psf_model=psf_model, gal_model=dataset_type)
        preds_ngmix = ngmix_pred(datalist)
        
        # Extract NGmix response
        r11_list, r22_list, r12_list, r21_list, _, _, _, _ = response_calculation(
            datalist, mcal_shear=0.01
        )
        
        r11_array = np.array(r11_list)
        r22_array = np.array(r22_list)
        r12_array = np.array(r12_list)
        r21_array = np.array(r21_list)
        
        R_per_gal_ngmix = np.stack([
            np.stack([r11_array, r12_array], axis=1),
            np.stack([r21_array, r22_array], axis=1)
        ], axis=1)
        
        return preds_nn, R_per_gal_nn, preds_ngmix, R_per_gal_ngmix
    
    # Dataset A: g1 = +0.02
    print(f"\n{CYAN}Dataset A: g1 = +{shear_step}{END}")
    images_g1_pos, labels_g1_pos, obs_g1_pos = generate_dataset(
        config['test_samples'], config['psf_sigma'],
        exp=config['exp'], seed=config['seed'],
        nse_sd=config['nse_sd'], npix=config['stamp_size'],
        scale=config['pixel_size'], return_psf=config['process_psf'],
        return_obs=True, apply_psf_shear=config['apply_psf_shear'],
        psf_shear_range=config['psf_shear_range'],
        base_shear_g1=shear_step, base_shear_g2=0.0
    )
    
    if config['process_psf']:
        gal_images_g1_pos, psf_images_g1_pos = split_combined_images(images_g1_pos, has_psf=True, has_clean=False)
        preds_nn, R_nn, preds_ngmix, R_ngmix = evaluate_dataset(
            gal_images_g1_pos, obs_g1_pos, psf_images_g1_pos, 'fork'
        )
    else:
        gal_images_g1_pos = images_g1_pos
        psf_images_g1_pos = None
        preds_nn, R_nn, preds_ngmix, R_ngmix = evaluate_dataset(
            gal_images_g1_pos, obs_g1_pos, None, 'standard'
        )
    
    bias_datasets['g1_pos'] = {
        'obs': obs_g1_pos,
        'nn_preds': preds_nn,
        'nn_R_per_gal': R_nn,
        'ngmix_preds': preds_ngmix,
        'ngmix_R_per_gal': R_ngmix,
        'psf_images': psf_images_g1_pos
    }
    
    # Dataset B: g1 = -0.02
    print(f"\n{CYAN}Dataset B: g1 = -{shear_step}{END}")
    images_g1_neg, labels_g1_neg, obs_g1_neg = generate_dataset(
        config['test_samples'], config['psf_sigma'],
        exp=config['exp'], seed=config['seed'],
        nse_sd=config['nse_sd'], npix=config['stamp_size'],
        scale=config['pixel_size'], return_psf=config['process_psf'],
        return_obs=True, apply_psf_shear=config['apply_psf_shear'],
        psf_shear_range=config['psf_shear_range'],
        base_shear_g1=-shear_step, base_shear_g2=0.0
    )
    
    if config['process_psf']:
        gal_images_g1_neg, psf_images_g1_neg = split_combined_images(images_g1_neg, has_psf=True, has_clean=False)
        preds_nn, R_nn, preds_ngmix, R_ngmix = evaluate_dataset(
            gal_images_g1_neg, obs_g1_neg, psf_images_g1_neg, 'fork'
        )
    else:
        gal_images_g1_neg = images_g1_neg
        psf_images_g1_neg = None
        preds_nn, R_nn, preds_ngmix, R_ngmix = evaluate_dataset(
            gal_images_g1_neg, obs_g1_neg, None, 'standard'
        )
    
    bias_datasets['g1_neg'] = {
        'obs': obs_g1_neg,
        'nn_preds': preds_nn,
        'nn_R_per_gal': R_nn,
        'ngmix_preds': preds_ngmix,
        'ngmix_R_per_gal': R_ngmix,
        'psf_images': psf_images_g1_neg
    }
    
    # Dataset C: g2 = +0.02
    print(f"\n{CYAN}Dataset C: g2 = +{shear_step}{END}")
    images_g2_pos, labels_g2_pos, obs_g2_pos = generate_dataset(
        config['test_samples'], config['psf_sigma'],
        exp=config['exp'], seed=config['seed'],
        nse_sd=config['nse_sd'], npix=config['stamp_size'],
        scale=config['pixel_size'], return_psf=config['process_psf'],
        return_obs=True, apply_psf_shear=config['apply_psf_shear'],
        psf_shear_range=config['psf_shear_range'],
        base_shear_g1=0.0, base_shear_g2=shear_step
    )
    
    if config['process_psf']:
        gal_images_g2_pos, psf_images_g2_pos = split_combined_images(images_g2_pos, has_psf=True, has_clean=False)
        preds_nn, R_nn, preds_ngmix, R_ngmix = evaluate_dataset(
            gal_images_g2_pos, obs_g2_pos, psf_images_g2_pos, 'fork'
        )
    else:
        gal_images_g2_pos = images_g2_pos
        psf_images_g2_pos = None
        preds_nn, R_nn, preds_ngmix, R_ngmix = evaluate_dataset(
            gal_images_g2_pos, obs_g2_pos, None, 'standard'
        )
    
    bias_datasets['g2_pos'] = {
        'obs': obs_g2_pos,
        'nn_preds': preds_nn,
        'nn_R_per_gal': R_nn,
        'ngmix_preds': preds_ngmix,
        'ngmix_R_per_gal': R_ngmix,
        'psf_images': psf_images_g2_pos
    }
    
    # Dataset D: g2 = -0.02
    print(f"\n{CYAN}Dataset D: g2 = -{shear_step}{END}")
    images_g2_neg, labels_g2_neg, obs_g2_neg = generate_dataset(
        config['test_samples'], config['psf_sigma'],
        exp=config['exp'], seed=config['seed'],
        nse_sd=config['nse_sd'], npix=config['stamp_size'],
        scale=config['pixel_size'], return_psf=config['process_psf'],
        return_obs=True, apply_psf_shear=config['apply_psf_shear'],
        psf_shear_range=config['psf_shear_range'],
        base_shear_g1=0.0, base_shear_g2=-shear_step
    )
    
    if config['process_psf']:
        gal_images_g2_neg, psf_images_g2_neg = split_combined_images(images_g2_neg, has_psf=True, has_clean=False)
        preds_nn, R_nn, preds_ngmix, R_ngmix = evaluate_dataset(
            gal_images_g2_neg, obs_g2_neg, psf_images_g2_neg, 'fork'
        )
    else:
        gal_images_g2_neg = images_g2_neg
        psf_images_g2_neg = None
        preds_nn, R_nn, preds_ngmix, R_ngmix = evaluate_dataset(
            gal_images_g2_neg, obs_g2_neg, None, 'standard'
        )
    
    bias_datasets['g2_neg'] = {
        'obs': obs_g2_neg,
        'nn_preds': preds_nn,
        'nn_R_per_gal': R_nn,
        'ngmix_preds': preds_ngmix,
        'ngmix_R_per_gal': R_ngmix,
        'psf_images': psf_images_g2_neg
    }
    
    print(f"\n{GREEN}✓ Generated 4 bias datasets with NN and NGmix predictions/responses{END}")
    
    return bias_datasets, shear_step


def print_summary(config, nn_results, ngmix_results):
    """Print evaluation summary."""
    print(f"\n{BOLD}{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}{END}")
    
    print(f"\n{BOLD}Configuration:{END}")
    print(f"  Model: {config['model_name']}")
    print(f"  Architecture: {config['nn']}")
    print(f"  Test samples: {config['test_samples']}")
    print(f"  Seed: {config['seed']}")
    
    print(f"\n{BOLD}ShearNet Performance:{END}")
    print(f"  MSE: {nn_results['loss']:.6e}")
    print(f"  Bias: {nn_results['bias']:.6e}")
    print(f"  Time: {nn_results['time_taken']:.2f}s")
    
    if ngmix_results:
        print(f"\n{BOLD}NGmix Performance:{END}")
        print(f"  MSE: {ngmix_results['loss']:.6e}")
        print(f"  Bias: {ngmix_results['bias']:.6e}")
        print(f"  Time: {ngmix_results['time_taken']:.2f}s")
        
        # Speedup
        speedup = ngmix_results['time_taken'] / nn_results['time_taken']
        print(f"\n{BOLD}Speedup:{END} {CYAN}{speedup:.2f}x{END}")
    
    print(f"\n{GREEN}{'='*70}")
    print("✓ EVALUATION COMPLETE")
    print(f"{'='*70}{END}\n")


def save_evaluation_to_fits(
    test_galaxy_images, test_psf_images, test_labels, test_obs,
    nn_predictions, ngmix_predictions, R_per_gal_nn, R_per_gal_ngmix,
    bias_datasets, shear_step,
    config, output_path, include_images=True
):
    """
    Save complete evaluation dataset to FITS file.
    
    Args:
        include_images: If True, include image HDUs; if False, catalog only
        
    Structure:
    - HDU 0 (Primary): Metadata header with SHRSTEP
    - HDU 1 (CATALOG): Scalar measurements table with:
        * Ground truth (TRUE_G1, TRUE_G2, TRUE_SIGMA, TRUE_FLUX)
        * Base dataset: NN and NGmix predictions (uncalibrated and calibrated) with response matrices
        * Bias datasets: NN and NGmix predictions for g1_pos, g1_neg, g2_pos, g2_neg with response matrices
        * PSF properties (PSF_E1, PSF_E2, PSF_T, PSF_FLAG)
        * Galaxy size (T_GAL, SIZE_RATIO)
        * Observation properties (SNR, WEIGHT, FLAGS)
    - HDU 2 (CLEAN_IMAGES): True galaxy images without PSF/noise (N, H, W) [if include_images]
    - HDU 3 (GALAXY_IMAGES): Observed galaxy images (N, H, W) [if include_images]
    - HDU 4 (PSF_IMAGES): PSF images (N, H, W) [if include_images]
    """
    
    n_samples = len(test_labels)
    npix = test_galaxy_images.shape[1] if include_images else config['stamp_size']
    
    file_type = "with images" if include_images else "catalog only"
    print(f"\n{BOLD}{CYAN}Creating FITS file ({file_type})...{END}")
    
    # ========== HDU 0: Primary Header with Metadata ==========
    primary_hdu = fits.PrimaryHDU()
    
    # Dataset parameters
    primary_hdu.header['NSAMPLES'] = (n_samples, 'Number of galaxies')
    primary_hdu.header['NPIX'] = (npix, 'Image stamp size (pixels)')
    primary_hdu.header['SCALE'] = (config['pixel_size'], 'Pixel scale (arcsec/pixel)')
    
    # Simulation parameters
    primary_hdu.header['PSF_SIG'] = (config['psf_sigma'], 'PSF sigma')
    primary_hdu.header['NSE_SD'] = (config['nse_sd'], 'Noise standard deviation')
    primary_hdu.header['EXP'] = (config['exp'], 'Experiment type')
    primary_hdu.header['SEED'] = (config['seed'], 'Random seed')
    
    # Bias calibration shear step
    primary_hdu.header['SHRSTEP'] = (shear_step, 'Shear step for bias calibration datasets')
    
    # Model information
    primary_hdu.header['MODELNAM'] = (config['model_name'], 'Model name')
    primary_hdu.header['MODELTYP'] = (config['nn'], 'Model architecture type')
    
    # PSF shear information (if applicable)
    if config.get('apply_psf_shear', False):
        primary_hdu.header['PSFSHEAR'] = (True, 'PSF shear applied')
        primary_hdu.header['PSFSHRNG'] = (config['psf_shear_range'], 'PSF shear range')
    else:
        primary_hdu.header['PSFSHEAR'] = (False, 'PSF shear not applied')
    
    # Processing flags
    primary_hdu.header['PROCPSF'] = (config['process_psf'], 'Separate PSF processing')
    primary_hdu.header['HASIMG'] = (include_images, 'Contains image HDUs')
    
    # Timestamp
    primary_hdu.header['DATE'] = (datetime.now().isoformat(), 'File creation date')
    
    primary_hdu.header['COMMENT'] = 'ShearNet evaluation dataset'
    primary_hdu.header['COMMENT'] = 'Contains bias calibration datasets for multiplicative bias estimation'
    primary_hdu.header['COMMENT'] = f'Bias datasets use shear step = +/- {shear_step}'
    
    # ========== HDU 1: Catalog Table with Scalar Data ==========
    
    # Extract observation metadata
    snr_values = np.array([obs.meta.get('snr', obs.get_s2n()) for obs in test_obs])
    
    # Extract PSF properties from PSF observation metadata (measured via adaptive moments)
    psf_e1 = np.array([obs.psf.meta.get('e1', np.nan) for obs in test_obs])
    psf_e2 = np.array([obs.psf.meta.get('e2', np.nan) for obs in test_obs])
    psf_T = np.array([obs.psf.meta.get('T', np.nan) for obs in test_obs])
    psf_flag = np.array([obs.psf.meta.get('admom_flag', 0) for obs in test_obs], dtype=np.int32)
    
    # Compute galaxy T from true sigma: T = 2 * sigma^2 (for Gaussian profile)
    true_sigma = test_labels[:, 2] if test_labels.shape[1] > 2 else np.zeros(n_samples)
    T_gal = 2.0 * true_sigma**2
    
    # Compute size ratio T_gal / T_psf
    with np.errstate(divide='ignore', invalid='ignore'):
        size_ratio = T_gal / psf_T
        size_ratio = np.where(np.isfinite(size_ratio), size_ratio, np.nan)
    
    # Extract per-galaxy NN response matrix components for BASE dataset
    # R_per_gal_nn shape: (n_samples, 2, 2)
    R11_nn = R_per_gal_nn[:, 0, 0] if R_per_gal_nn is not None else np.ones(n_samples)
    R12_nn = R_per_gal_nn[:, 0, 1] if R_per_gal_nn is not None else np.zeros(n_samples)
    R21_nn = R_per_gal_nn[:, 1, 0] if R_per_gal_nn is not None else np.zeros(n_samples)
    R22_nn = R_per_gal_nn[:, 1, 1] if R_per_gal_nn is not None else np.ones(n_samples)
    
    # Raw NN predictions (uncalibrated) for BASE dataset
    nn_g1_raw = nn_predictions['all_preds'][:, 0]
    nn_g2_raw = nn_predictions['all_preds'][:, 1]
    
    # Calibrated NN predictions: g_cal = g_raw / R
    # Using diagonal approximation: g1_cal = g1_raw / R11, g2_cal = g2_raw / R22
    with np.errstate(divide='ignore', invalid='ignore'):
        nn_g1_cal = nn_g1_raw / R11_nn
        nn_g2_cal = nn_g2_raw / R22_nn
        nn_g1_cal = np.where(np.isfinite(nn_g1_cal), nn_g1_cal, np.nan)
        nn_g2_cal = np.where(np.isfinite(nn_g2_cal), nn_g2_cal, np.nan)
    
    # Compute per-galaxy weights (inverse variance weighting based on SNR)
    # Higher SNR = higher weight; this is a simple scheme, can be refined
    weight = snr_values**2 / np.sum(snr_values**2)  # Normalized weights
    
    # Initialize flags array
    flags = np.zeros(n_samples, dtype=np.int32)
    
    # Flag galaxies with PSF measurement failures (bit 0)
    flags[psf_flag != 0] |= 1
    
    # Flag galaxies with invalid response (bit 1)
    invalid_response = ~np.isfinite(R11_nn) | ~np.isfinite(R22_nn) | (R11_nn == 0) | (R22_nn == 0)
    flags[invalid_response] |= 2
    
    # Build catalog table
    catalog_data = {
        # Galaxy ID
        'ID': np.arange(n_samples),
        
        # ===== GROUND TRUTH =====
        'TRUE_G1': test_labels[:, 0].astype(np.float64),
        'TRUE_G2': test_labels[:, 1].astype(np.float64),
        'TRUE_SIGMA': true_sigma.astype(np.float64),
        'TRUE_FLUX': test_labels[:, 3].astype(np.float64) if test_labels.shape[1] > 3 else np.zeros(n_samples, dtype=np.float64),
        
        # ===== BASE DATASET - NN PREDICTIONS =====
        'NN_G1': nn_g1_raw.astype(np.float64),
        'NN_G2': nn_g2_raw.astype(np.float64),
        'NN_G1_CAL': nn_g1_cal.astype(np.float64),
        'NN_G2_CAL': nn_g2_cal.astype(np.float64),
        
        # ===== BASE DATASET - NN RESPONSE MATRIX =====
        'NN_R11': R11_nn.astype(np.float64),
        'NN_R12': R12_nn.astype(np.float64),
        'NN_R21': R21_nn.astype(np.float64),
        'NN_R22': R22_nn.astype(np.float64),
        
        # ===== PSF PROPERTIES (from adaptive moments) =====
        'PSF_E1': psf_e1.astype(np.float64),
        'PSF_E2': psf_e2.astype(np.float64),
        'PSF_T': psf_T.astype(np.float64),
        'PSF_FLAG': psf_flag,
        
        # ===== GALAXY SIZE =====
        'T_GAL': T_gal.astype(np.float64),
        'SIZE_RATIO': size_ratio.astype(np.float64),
        
        # ===== OBSERVATION PROPERTIES =====
        'SNR': snr_values.astype(np.float64),
        'WEIGHT': weight.astype(np.float64),
        'FLAGS': flags,
    }
    
    # Add BASE dataset NGmix predictions and response if available
    if ngmix_predictions is not None and ngmix_predictions.get('preds') is not None:
        preds = ngmix_predictions['preds']
        
        # Uncalibrated NGmix predictions
        catalog_data['NGMIX_G1'] = preds[:, 0].astype(np.float64)
        catalog_data['NGMIX_G2'] = preds[:, 1].astype(np.float64)
        
        # Flag NaN predictions (bit 2)
        nan_mask = np.any(np.isnan(preds), axis=1)
        flags[nan_mask] |= 4
        
        # Add NGmix response matrix if available
        if R_per_gal_ngmix is not None:
            R11_ngmix = R_per_gal_ngmix[:, 0, 0]
            R12_ngmix = R_per_gal_ngmix[:, 0, 1]
            R21_ngmix = R_per_gal_ngmix[:, 1, 0]
            R22_ngmix = R_per_gal_ngmix[:, 1, 1]
            
            catalog_data['NGMIX_R11'] = R11_ngmix.astype(np.float64)
            catalog_data['NGMIX_R12'] = R12_ngmix.astype(np.float64)
            catalog_data['NGMIX_R21'] = R21_ngmix.astype(np.float64)
            catalog_data['NGMIX_R22'] = R22_ngmix.astype(np.float64)
            
            # Calibrated NGmix predictions
            with np.errstate(divide='ignore', invalid='ignore'):
                ngmix_g1_cal = preds[:, 0] / R11_ngmix
                ngmix_g2_cal = preds[:, 1] / R22_ngmix
                ngmix_g1_cal = np.where(np.isfinite(ngmix_g1_cal), ngmix_g1_cal, np.nan)
                ngmix_g2_cal = np.where(np.isfinite(ngmix_g2_cal), ngmix_g2_cal, np.nan)
            
            catalog_data['NGMIX_G1_CAL'] = ngmix_g1_cal.astype(np.float64)
            catalog_data['NGMIX_G2_CAL'] = ngmix_g2_cal.astype(np.float64)
    
    # ===== ADD BIAS CALIBRATION DATASETS =====
    for dataset_name, dataset_info in bias_datasets.items():
        prefix = dataset_name.upper()  # G1_POS, G1_NEG, G2_POS, G2_NEG
        
        # ===== NN DATA =====
        nn_preds = dataset_info['nn_preds']
        nn_R_per_gal = dataset_info['nn_R_per_gal']
        
        # Uncalibrated NN predictions
        catalog_data[f'{prefix}_NN_G1'] = nn_preds[:, 0].astype(np.float64)
        catalog_data[f'{prefix}_NN_G2'] = nn_preds[:, 1].astype(np.float64)
        
        # NN Response matrix
        nn_R11 = nn_R_per_gal[:, 0, 0]
        nn_R12 = nn_R_per_gal[:, 0, 1]
        nn_R21 = nn_R_per_gal[:, 1, 0]
        nn_R22 = nn_R_per_gal[:, 1, 1]
        
        catalog_data[f'{prefix}_NN_R11'] = nn_R11.astype(np.float64)
        catalog_data[f'{prefix}_NN_R12'] = nn_R12.astype(np.float64)
        catalog_data[f'{prefix}_NN_R21'] = nn_R21.astype(np.float64)
        catalog_data[f'{prefix}_NN_R22'] = nn_R22.astype(np.float64)
        
        # Calibrated NN predictions
        with np.errstate(divide='ignore', invalid='ignore'):
            nn_g1_cal = nn_preds[:, 0] / nn_R11
            nn_g2_cal = nn_preds[:, 1] / nn_R22
            nn_g1_cal = np.where(np.isfinite(nn_g1_cal), nn_g1_cal, np.nan)
            nn_g2_cal = np.where(np.isfinite(nn_g2_cal), nn_g2_cal, np.nan)
        
        catalog_data[f'{prefix}_NN_G1_CAL'] = nn_g1_cal.astype(np.float64)
        catalog_data[f'{prefix}_NN_G2_CAL'] = nn_g2_cal.astype(np.float64)
        
        # ===== NGMIX DATA =====
        ngmix_preds = dataset_info['ngmix_preds']
        ngmix_R_per_gal = dataset_info['ngmix_R_per_gal']
        
        # Uncalibrated NGmix predictions
        catalog_data[f'{prefix}_NGMIX_G1'] = ngmix_preds[:, 0].astype(np.float64)
        catalog_data[f'{prefix}_NGMIX_G2'] = ngmix_preds[:, 1].astype(np.float64)
        
        # NGmix Response matrix
        ngmix_R11 = ngmix_R_per_gal[:, 0, 0]
        ngmix_R12 = ngmix_R_per_gal[:, 0, 1]
        ngmix_R21 = ngmix_R_per_gal[:, 1, 0]
        ngmix_R22 = ngmix_R_per_gal[:, 1, 1]
        
        catalog_data[f'{prefix}_NGMIX_R11'] = ngmix_R11.astype(np.float64)
        catalog_data[f'{prefix}_NGMIX_R12'] = ngmix_R12.astype(np.float64)
        catalog_data[f'{prefix}_NGMIX_R21'] = ngmix_R21.astype(np.float64)
        catalog_data[f'{prefix}_NGMIX_R22'] = ngmix_R22.astype(np.float64)
        
        # Calibrated NGmix predictions
        with np.errstate(divide='ignore', invalid='ignore'):
            ngmix_g1_cal = ngmix_preds[:, 0] / ngmix_R11
            ngmix_g2_cal = ngmix_preds[:, 1] / ngmix_R22
            ngmix_g1_cal = np.where(np.isfinite(ngmix_g1_cal), ngmix_g1_cal, np.nan)
            ngmix_g2_cal = np.where(np.isfinite(ngmix_g2_cal), ngmix_g2_cal, np.nan)
        
        catalog_data[f'{prefix}_NGMIX_G1_CAL'] = ngmix_g1_cal.astype(np.float64)
        catalog_data[f'{prefix}_NGMIX_G2_CAL'] = ngmix_g2_cal.astype(np.float64)
    
    # Update flags in catalog
    catalog_data['FLAGS'] = flags
    
    catalog_table = Table(catalog_data)
    catalog_hdu = fits.BinTableHDU(catalog_table, name='CATALOG')
    
    # Add detailed column descriptions to header
    catalog_hdu.header['COMMENT'] = 'ShearNet evaluation catalog'
    catalog_hdu.header['COMMENT'] = f'Bias calibration shear step: +/- {shear_step}'
    catalog_hdu.header['COMMENT'] = 'Base dataset has TRUE_G1, TRUE_G2 (no added shear)'
    catalog_hdu.header['COMMENT'] = 'Bias datasets: G1_POS (gamma1=+0.02), G1_NEG (gamma1=-0.02)'
    catalog_hdu.header['COMMENT'] = '                G2_POS (gamma2=+0.02), G2_NEG (gamma2=-0.02)'
    
    # Document flag bits
    catalog_hdu.header['FLAG0'] = ('PSF_FAIL', 'Bit 0: PSF adaptive moments failed')
    catalog_hdu.header['FLAG1'] = ('R_INVALID', 'Bit 1: Invalid response matrix')
    catalog_hdu.header['FLAG2'] = ('NGMIX_NAN', 'Bit 2: NGmix returned NaN')
    catalog_hdu.header['COMMENT'] = 'Use (FLAGS & N) to check bit N'
    
    # ========== HDU List ==========
    hdu_list = [primary_hdu, catalog_hdu]
    
    # ========== Add Image Extensions (if requested) ==========
    if include_images:
        # Clean galaxy images (no PSF, no noise) - THE TRUTH
        clean_images = np.array([obs.meta['clean_image'] for obs in test_obs])
        clean_hdu = fits.ImageHDU(clean_images, name='CLEAN_IMAGES')
        clean_hdu.header['EXTNAME'] = 'CLEAN_IMAGES'
        clean_hdu.header['COMMENT'] = 'True galaxy images (no PSF, no noise)'
        clean_hdu.header['BUNIT'] = 'counts'
        
        # Observed galaxy images (with PSF and noise) - WHAT WE MEASURE
        galaxy_hdu = fits.ImageHDU(np.asarray(test_galaxy_images), name='GALAXY_IMAGES')
        galaxy_hdu.header['EXTNAME'] = 'GALAXY_IMAGES'
        galaxy_hdu.header['COMMENT'] = 'Observed galaxy images (PSF-convolved + noise)'
        galaxy_hdu.header['BUNIT'] = 'counts'
        
        # PSF images
        if test_psf_images is not None:
            psf_hdu = fits.ImageHDU(np.asarray(test_psf_images), name='PSF_IMAGES')
        else:
            # Extract from observations
            psf_images = np.array([obs.psf.image for obs in test_obs])
            psf_hdu = fits.ImageHDU(psf_images, name='PSF_IMAGES')
        psf_hdu.header['EXTNAME'] = 'PSF_IMAGES'
        psf_hdu.header['COMMENT'] = 'Point Spread Function images'
        psf_hdu.header['BUNIT'] = 'normalized'
        
        hdu_list.extend([clean_hdu, galaxy_hdu, psf_hdu])
    
    # ========== Construct and Write HDU List ==========
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(output_path, overwrite=True)
    
    # Print summary
    print(f"\n{GREEN}✓ Evaluation data saved to: {output_path}{END}")
    print(f"\n{CYAN}FITS structure:{END}")
    print(f"  HDU 0 (Primary): Metadata header (SHRSTEP={shear_step})")
    print(f"  HDU 1 (CATALOG): {n_samples} galaxies with measurements")
    print(f"    - Ground truth: TRUE_G1, TRUE_G2, TRUE_SIGMA, TRUE_FLUX")
    print(f"    - Base dataset NN: NN_G1, NN_G2, NN_G1_CAL, NN_G2_CAL, NN_R11-R22")
    if ngmix_predictions is not None:
        print(f"    - Base dataset NGmix: NGMIX_G1, NGMIX_G2, NGMIX_G1_CAL, NGMIX_G2_CAL, NGMIX_R11-R22")
    print(f"    - Bias datasets (G1_POS, G1_NEG, G2_POS, G2_NEG):")
    print(f"      * NN: {{PREFIX}}_NN_G1, {{PREFIX}}_NN_G2, {{PREFIX}}_NN_G1_CAL, {{PREFIX}}_NN_G2_CAL")
    print(f"      * NN response: {{PREFIX}}_NN_R11, {{PREFIX}}_NN_R12, {{PREFIX}}_NN_R21, {{PREFIX}}_NN_R22")
    print(f"      * NGmix: {{PREFIX}}_NGMIX_G1, {{PREFIX}}_NGMIX_G2, {{PREFIX}}_NGMIX_G1_CAL, {{PREFIX}}_NGMIX_G2_CAL")
    print(f"      * NGmix response: {{PREFIX}}_NGMIX_R11-R22")
    print(f"    - PSF: PSF_E1, PSF_E2, PSF_T, PSF_FLAG")
    print(f"    - Size: T_GAL, SIZE_RATIO")
    print(f"    - Quality: SNR, WEIGHT, FLAGS")
    if include_images:
        print(f"  HDU 2 (CLEAN_IMAGES): True images {clean_images.shape}")
        print(f"  HDU 3 (GALAXY_IMAGES): Observed images {np.asarray(test_galaxy_images).shape}")
        print(f"  HDU 4 (PSF_IMAGES): PSF images")
    print(f"\n{CYAN}Quality summary:{END}")
    print(f"  Galaxies with PSF measurement issues: {np.sum(flags & 1 > 0)}")
    print(f"  Galaxies with invalid response: {np.sum(flags & 2 > 0)}")
    if ngmix_predictions is not None:
        print(f"  Galaxies with NGmix failures: {np.sum(flags & 4 > 0)}")
    print(f"\n{CYAN}File size: {os.path.getsize(output_path) / 1024**2:.2f} MB{END}")


def main():
    """Main evaluation function."""
    # Parse arguments and load config
    parser = create_parser()
    args = parser.parse_args()
    config = load_config(args)
    
    # Generate test data
    test_galaxy_images, test_psf_images, test_labels, test_obs, snr_values = generate_test_data(config)
    
    # Initialize and load model
    state = initialize_model(config, test_galaxy_images, test_psf_images)
    
    # Evaluate neural network
    nn_results = evaluate_neural_network(state, test_galaxy_images, test_psf_images, test_labels, config)
    
    # Evaluate comparison methods (if requested)
    ngmix_results = evaluate_comparison_methods(test_obs, test_galaxy_images, test_labels, config)

    # Calculate response matrices
    R_nn, R_per_gal_nn, R_ngmix, R_per_gal_ngmix = calculate_response_matrices(
        state, test_obs, test_galaxy_images, test_psf_images, 
        test_labels, config, ngmix_results
    )
    
    # Generate bias calibration datasets and evaluate
    bias_datasets, shear_step = generate_bias_datasets_and_evaluate(state, config)

    # Save catalog-only FITS
    catalog_path = os.path.join(config['plot_path'], config['model_name'], 'evaluation_catalog.fits')
    save_evaluation_to_fits(
        test_galaxy_images, test_psf_images, test_labels, test_obs,
        nn_results, ngmix_results if ngmix_results else None, 
        R_per_gal_nn, R_per_gal_ngmix if R_per_gal_ngmix is not None else None,
        bias_datasets, shear_step,
        config, catalog_path,
        include_images=False
    )
    
    # Save FITS with images
    full_path = os.path.join(config['plot_path'], config['model_name'], 'evaluation_full.fits')
    save_evaluation_to_fits(
        test_galaxy_images, test_psf_images, test_labels, test_obs,
        nn_results, ngmix_results if ngmix_results else None, 
        R_per_gal_nn, R_per_gal_ngmix if R_per_gal_ngmix is not None else None,
        bias_datasets, shear_step,
        config, full_path,
        include_images=True
    )
    
    # Print summary
    print_summary(config, nn_results, ngmix_results)


if __name__ == "__main__":
    main()
