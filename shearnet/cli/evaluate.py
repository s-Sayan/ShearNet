"""Command-line interface for evaluating trained ShearNet models."""

import os
import argparse
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
import time
from flax.training import checkpoints, train_state

from ..config.config_handler import Config
from ..core.dataset import generate_dataset, split_combined_images
from ..core.models import (
    SimpleGalaxyNN, EnhancedGalaxyNN, GalaxyResNet, 
    ResearchBackedGalaxyResNet, ForkLensPSFNet, ForkLike
)
from ..utils.metrics import (
    eval_model, fork_eval_model, eval_ngmix, eval_mcal, 
    remove_nan_preds_multi, calculate_response_matrix, 
    calculate_ngmix_response_matrix,
    calculate_multiplicative_bias,
    calculate_multiplicative_bias_ngmix
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
    mcal_results = None
    
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
        
        # Metacalibration
        mcal_results = eval_mcal(test_galaxy_images, test_labels, config['psf_sigma'])
    
    return ngmix_results, mcal_results


def calculate_response_matrices(state, test_obs, test_galaxy_images, test_psf_images, test_labels, config, ngmix_results):
    """Calculate shear response matrices for calibration."""
    print(f"\n{BOLD}{'='*50}")
    print("Calculating Response Matrices")
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
    
    R_ngmix = None
    R_per_gal_ngmix = None
    
    # NGmix response matrix (if comparison is enabled)
    if config['mcal'] and ngmix_results is not None:
        dataset_type = config.get('dataset_type', 'gauss')
        psf_model = config.get('psf_model', 'gauss')
        
        R_ngmix, R_per_gal_ngmix = calculate_ngmix_response_matrix(
            test_obs, test_labels, h=0.01, seed=config['seed'],
            psf_model=psf_model, gal_model=dataset_type
        )
        
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


def save_response_matrices(R_nn, R_per_gal_nn, R_ngmix, R_per_gal_ngmix, config):
    """Save response matrices to file."""
    plot_path = os.path.join(config['plot_path'], config['model_name'])
    os.makedirs(plot_path, exist_ok=True)
    
    response_path = os.path.join(plot_path, "response_matrices.npz")
    
    if R_ngmix is not None:
        np.savez(response_path, 
                R_nn=R_nn, R_per_gal_nn=R_per_gal_nn,
                R_ngmix=R_ngmix, R_per_gal_ngmix=R_per_gal_ngmix)
    else:
        np.savez(response_path, 
                R_nn=R_nn, R_per_gal_nn=R_per_gal_nn)
    
    print(f"\n{GREEN}Response matrices saved to: {response_path}{END}")


def generate_bias_calibration_datasets(config):
    """
    Generate four datasets for bias calibration.
    
    Returns:
        tuple: (obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
                psf_g1_pos, psf_g1_neg, psf_g2_pos, psf_g2_neg)
    """
    print(f"\n{BOLD}{'='*50}")
    print("Generating Bias Calibration Datasets")
    print(f"{'='*50}{END}")
    
    shear_step = 0.02
    print(f"Shear step: ±{shear_step}")
    print(f"Samples per dataset: {config['test_samples']}")
    
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
    
    # Dataset B: g1 = -0.02
    print(f"\n{CYAN}Dataset B: g1 = -{shear_step}{END}")
    images_g1_neg, labels_g1_neg, obs_g1_neg = generate_dataset(
        config['test_samples'], config['psf_sigma'],
        exp=config['exp'], seed=config['seed'] + 1,
        nse_sd=config['nse_sd'], npix=config['stamp_size'],
        scale=config['pixel_size'], return_psf=config['process_psf'],
        return_obs=True, apply_psf_shear=config['apply_psf_shear'],
        psf_shear_range=config['psf_shear_range'],
        base_shear_g1=-shear_step, base_shear_g2=0.0
    )
    
    # Dataset C: g2 = +0.02
    print(f"\n{CYAN}Dataset C: g2 = +{shear_step}{END}")
    images_g2_pos, labels_g2_pos, obs_g2_pos = generate_dataset(
        config['test_samples'], config['psf_sigma'],
        exp=config['exp'], seed=config['seed'] + 2,
        nse_sd=config['nse_sd'], npix=config['stamp_size'],
        scale=config['pixel_size'], return_psf=config['process_psf'],
        return_obs=True, apply_psf_shear=config['apply_psf_shear'],
        psf_shear_range=config['psf_shear_range'],
        base_shear_g1=0.0, base_shear_g2=shear_step
    )
    
    # Dataset D: g2 = -0.02
    print(f"\n{CYAN}Dataset D: g2 = -{shear_step}{END}")
    images_g2_neg, labels_g2_neg, obs_g2_neg = generate_dataset(
        config['test_samples'], config['psf_sigma'],
        exp=config['exp'], seed=config['seed'] + 3,
        nse_sd=config['nse_sd'], npix=config['stamp_size'],
        scale=config['pixel_size'], return_psf=config['process_psf'],
        return_obs=True, apply_psf_shear=config['apply_psf_shear'],
        psf_shear_range=config['psf_shear_range'],
        base_shear_g1=0.0, base_shear_g2=-shear_step
    )
    
    # Split PSF images if needed
    psf_g1_pos = psf_g1_neg = psf_g2_pos = psf_g2_neg = None
    if config['process_psf']:
        _, psf_g1_pos = split_combined_images(images_g1_pos, has_psf=True, has_clean=False)
        _, psf_g1_neg = split_combined_images(images_g1_neg, has_psf=True, has_clean=False)
        _, psf_g2_pos = split_combined_images(images_g2_pos, has_psf=True, has_clean=False)
        _, psf_g2_neg = split_combined_images(images_g2_neg, has_psf=True, has_clean=False)
        print(f"\n{GREEN}✓ Generated 4 bias datasets with PSF images{END}")
    else:
        print(f"\n{GREEN}✓ Generated 4 bias datasets{END}")
    
    return (obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
            psf_g1_pos, psf_g1_neg, psf_g2_pos, psf_g2_neg)


def calculate_bias_calibration(state, obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
                               psf_g1_pos, psf_g1_neg, psf_g2_pos, psf_g2_neg,
                               config, calculate_ngmix_bias=False):
    """
    Calculate multiplicative and additive bias for ShearNet (and optionally NGmix).
    
    Args:
        state: Model state
        obs_g1_pos, obs_g1_neg: Observations for g1 bias datasets
        obs_g2_pos, obs_g2_neg: Observations for g2 bias datasets  
        psf_g1_pos, psf_g1_neg, psf_g2_pos, psf_g2_neg: PSF images (if fork model)
        config: Configuration dict
        calculate_ngmix_bias: Whether to also calculate NGmix bias
    
    Returns:
        tuple: (bias_results_nn, bias_results_ngmix, time_nn, time_ngmix)
    """
    print(f"\n{BOLD}{'='*50}")
    print("Calculating Bias Calibration")
    print(f"{'='*50}{END}")
    
    # Calculate ShearNet bias
    print(f"\n{BOLD}{YELLOW}ShearNet Bias Calibration{END}")
    time_start_nn = time.time()
    bias_results_nn = calculate_multiplicative_bias(
        state, obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
        true_shear_step=0.02,
        batch_size=32,
        h=0.01,
        model_type='fork' if config['process_psf'] else 'standard',
        psf_g1_pos=psf_g1_pos, psf_g1_neg=psf_g1_neg,
        psf_g2_pos=psf_g2_pos, psf_g2_neg=psf_g2_neg
    )
    time_nn = time.time() - time_start_nn
    
    # Calculate NGmix bias if requested
    bias_results_ngmix = None
    time_ngmix = None
    if calculate_ngmix_bias:
        print(f"\n{BOLD}{YELLOW}NGmix Bias Calibration{END}")
        # Get model types from config
        dataset_type = config.get('dataset_type', 'gauss')
        psf_model_type = config.get('psf_model', 'gauss')
        
        time_start_ngmix = time.time()
        bias_results_ngmix = calculate_multiplicative_bias_ngmix(
            obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
            true_shear_step=0.02,
            h=0.01,
            seed=config['seed'],
            psf_model=psf_model_type,
            gal_model=dataset_type
        )
        time_ngmix = time.time() - time_start_ngmix
        
        # Print comparison
        print(f"\n{BOLD}{'='*70}")
        print("BIAS CALIBRATION COMPARISON")
        print(f"{'='*70}{END}")
        print(f"\n{CYAN}ShearNet:{END}")
        print(f"  m₁ = {bias_results_nn['m1']:+.6f} ({bias_results_nn['m1']*100:+.2f}%)")
        print(f"  m₂ = {bias_results_nn['m2']:+.6f} ({bias_results_nn['m2']*100:+.2f}%)")
        print(f"  c₁ = {bias_results_nn['c1']:+.6f}")
        print(f"  c₂ = {bias_results_nn['c2']:+.6f}")
        print(f"  Time: {time_nn:.2f}s")
        
        print(f"\n{CYAN}NGmix:{END}")
        print(f"  m₁ = {bias_results_ngmix['m1']:+.6f} ({bias_results_ngmix['m1']*100:+.2f}%)")
        print(f"  m₂ = {bias_results_ngmix['m2']:+.6f} ({bias_results_ngmix['m2']*100:+.2f}%)")
        print(f"  c₁ = {bias_results_ngmix['c1']:+.6f}")
        print(f"  c₂ = {bias_results_ngmix['c2']:+.6f}")
        print(f"  Time: {time_ngmix:.2f}s")
        
        # Differences
        print(f"\n{YELLOW}Difference (ShearNet - NGmix):{END}")
        print(f"  Δm₁ = {bias_results_nn['m1'] - bias_results_ngmix['m1']:+.6f}")
        print(f"  Δm₂ = {bias_results_nn['m2'] - bias_results_ngmix['m2']:+.6f}")
        print(f"  Δc₁ = {bias_results_nn['c1'] - bias_results_ngmix['c1']:+.6f}")
        print(f"  Δc₂ = {bias_results_nn['c2'] - bias_results_ngmix['c2']:+.6f}")
        
        # Speedup
        speedup = time_ngmix / time_nn
        print(f"\n{BOLD}Speedup:{END} {CYAN}{speedup:.2f}x{END}")
        print(f"{BOLD}{'='*70}{END}\n")
    
    return bias_results_nn, bias_results_ngmix, time_nn, time_ngmix


def save_bias_results(bias_results_nn, bias_results_ngmix, time_nn, time_ngmix, config):
    """Save bias calibration results to file."""
    plot_path = os.path.join(config['plot_path'], config['model_name'])
    os.makedirs(plot_path, exist_ok=True)
    
    bias_path = os.path.join(plot_path, "bias_calibration.npz")
    
    # Prepare save dict
    save_dict = {
        # ShearNet biases
        'm1': bias_results_nn['m1'],
        'c1': bias_results_nn['c1'],
        'm2': bias_results_nn['m2'],
        'c2': bias_results_nn['c2'],
        'gamma_est_g1_pos': bias_results_nn['gamma_est_g1_pos'],
        'gamma_est_g1_neg': bias_results_nn['gamma_est_g1_neg'],
        'gamma_est_g2_pos': bias_results_nn['gamma_est_g2_pos'],
        'gamma_est_g2_neg': bias_results_nn['gamma_est_g2_neg'],
        'R_g1_pos': bias_results_nn['R_g1_pos'],
        'R_g1_neg': bias_results_nn['R_g1_neg'],
        'R_g2_pos': bias_results_nn['R_g2_pos'],
        'R_g2_neg': bias_results_nn['R_g2_neg'],
        'time_shearnet': time_nn,
    }
    
    # Add NGmix results if available
    if bias_results_ngmix is not None:
        save_dict.update({
            'm1_ngmix': bias_results_ngmix['m1'],
            'c1_ngmix': bias_results_ngmix['c1'],
            'm2_ngmix': bias_results_ngmix['m2'],
            'c2_ngmix': bias_results_ngmix['c2'],
            'gamma_est_g1_pos_ngmix': bias_results_ngmix['gamma_est_g1_pos'],
            'gamma_est_g1_neg_ngmix': bias_results_ngmix['gamma_est_g1_neg'],
            'gamma_est_g2_pos_ngmix': bias_results_ngmix['gamma_est_g2_pos'],
            'gamma_est_g2_neg_ngmix': bias_results_ngmix['gamma_est_g2_neg'],
            'R_g1_pos_ngmix': bias_results_ngmix['R_g1_pos'],
            'R_g1_neg_ngmix': bias_results_ngmix['R_g1_neg'],
            'R_g2_pos_ngmix': bias_results_ngmix['R_g2_pos'],
            'R_g2_neg_ngmix': bias_results_ngmix['R_g2_neg'],
            'time_ngmix': time_ngmix,
            'speedup': time_ngmix / time_nn if time_ngmix else None,
        })
    
    np.savez(bias_path, **save_dict)
    
    print(f"\n{GREEN}Bias calibration results saved to: {bias_path}{END}")
    print(f"{CYAN}Saved parameters:{END}")
    print(f"  • m₁ = {bias_results_nn['m1']:+.6f}, c₁ = {bias_results_nn['c1']:+.6f}")
    print(f"  • m₂ = {bias_results_nn['m2']:+.6f}, c₂ = {bias_results_nn['c2']:+.6f}")
    print(f"  • Time (ShearNet): {time_nn:.2f}s")
    if bias_results_ngmix:
        print(f"  • m₁ (NGmix) = {bias_results_ngmix['m1']:+.6f}, c₁ = {bias_results_ngmix['c1']:+.6f}")
        print(f"  • m₂ (NGmix) = {bias_results_ngmix['m2']:+.6f}, c₂ = {bias_results_ngmix['c2']:+.6f}")
        print(f"  • Time (NGmix): {time_ngmix:.2f}s")
        print(f"  • Speedup: {time_ngmix / time_nn:.2f}x")


def generate_plots(state, test_obs, test_galaxy_images, test_psf_images, test_labels, snr_values, ngmix_results, R_nn, R_ngmix, config):
    """Generate evaluation plots."""
    print(f"\n{BOLD}{'='*50}")
    print("Generating Plots")
    print(f"{'='*50}{END}")
    
    # Get predictions
    if config['process_psf']:
        predicted_labels = state.apply_fn(
            state.params, test_galaxy_images, test_psf_images, deterministic=True
        )
    else:
        predicted_labels = state.apply_fn(
            state.params, test_galaxy_images, deterministic=True
        )
    
    # Remove NaNs if comparing with NGmix
    ngmix_preds = None
    if config['mcal'] and ngmix_results:
        ngmix_preds = ngmix_results['preds']
        predicted_labels, ngmix_preds, test_labels = remove_nan_preds_multi(
            predicted_labels, ngmix_preds, test_labels
        )

    df_plot_path = os.path.join(config['plot_path'], config['model_name'])
    os.makedirs(df_plot_path, exist_ok=True)
    
    # Residuals
    print("  → Plotting residuals...")
    residuals_path = os.path.join(df_plot_path, "residuals")
    plot_residuals(
        test_labels, predicted_labels, path=residuals_path,
        mcal=config['mcal'], preds_ngmix=ngmix_preds
    )

    # Sample visualizations
    if config['process_psf']:
        print("  → Plotting galaxy samples...")
        samples_galaxy_path = os.path.join(df_plot_path, "samples_galaxy_plot.png")
        visualize_galaxy_samples(
            test_galaxy_images, test_labels, predicted_labels, 
            snr_values, path=samples_galaxy_path
        )
        
        print("  → Plotting PSF samples...")
        samples_psf_path = os.path.join(df_plot_path, "samples_psf_plot.png")
        visualize_psf_samples(test_psf_images, path=samples_psf_path)
    else:
        print("  → Plotting samples...")
        samples_path = os.path.join(df_plot_path, "samples_plot.png")
        visualize_galaxy_samples(
            test_galaxy_images, test_labels, predicted_labels, 
            snr_values, path=samples_path
        )

    # Scatter plots
    print("  → Plotting scatter plots...")
    scatter_path = os.path.join(df_plot_path, "scatters")
    plot_true_vs_predicted(
        test_labels, predicted_labels, path=scatter_path, 
        mcal=config['mcal'], preds_mcal=ngmix_preds
    )

    # PSF systematics plots
    print("  → Plotting psf systematics plots...")
    psf_systematics_path = os.path.join(df_plot_path, "psf_systematics")
    fig, results = plot_psf_systematics_from_eval(
        test_obs, 
        predicted_labels,
        response_matrix=R_nn,
        ngmix_preds=ngmix_preds,
        ngmix_response=R_ngmix,
        path=psf_systematics_path,
        n_bins=20
    )
    
    print(f"\n{GREEN}✓ All plots saved to: {df_plot_path}{END}")


def generate_animation(state, test_labels, ngmix_results, config):
    """Generate animation of model predictions over epochs (if requested)."""
    if not config['plot_animation']:
        return
    
    print(f"\n{BOLD}{'='*50}")
    print("Generating Animation")
    print(f"{'='*50}{END}")
    
    df_plot_path = os.path.join(config['plot_path'], config['model_name'])
    animation_path = os.path.join(df_plot_path, "animation_plot")
    epochs = np.arange(1, 101)  # Assuming 100 epochs
    
    ngmix_preds = ngmix_results['preds'] if ngmix_results else None
    
    animate_model_epochs(
        test_labels, config['save_path'], config['plot_path'], epochs, 
        state=state, model_name=config['model_name'], 
        mcal=config['mcal'], preds_mcal=ngmix_preds
    )
    
    print(f"{GREEN}✓ Animation saved to: {animation_path}{END}")


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
    ngmix_results, mcal_results = evaluate_comparison_methods(test_obs, test_galaxy_images, test_labels, config)
    
    # Calculate response matrices
    R_nn, R_per_gal_nn, R_ngmix, R_per_gal_ngmix = calculate_response_matrices(
        state, test_obs, test_galaxy_images, test_psf_images, 
        test_labels, config, ngmix_results
    )
    
    # Save response matrices
    save_response_matrices(R_nn, R_per_gal_nn, R_ngmix, R_per_gal_ngmix, config)
    
    # Generate bias calibration datasets
    (obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
     psf_g1_pos, psf_g1_neg, psf_g2_pos, psf_g2_neg) = generate_bias_calibration_datasets(config)
    
    # Calculate multiplicative and additive bias with timing
    bias_results_nn, bias_results_ngmix, time_nn, time_ngmix = calculate_bias_calibration(
        state, obs_g1_pos, obs_g1_neg, obs_g2_pos, obs_g2_neg,
        psf_g1_pos, psf_g1_neg, psf_g2_pos, psf_g2_neg,
        config, calculate_ngmix_bias=config['mcal']
    )
    
    # Save bias results with timing
    save_bias_results(bias_results_nn, bias_results_ngmix, time_nn, time_ngmix, config)
    
    # Generate plots (if requested)
    if config['plot']:
        generate_plots(state, test_obs, test_galaxy_images, test_psf_images, 
                      test_labels, snr_values, ngmix_results, R_nn, R_ngmix, config)
    
    # Generate animation (if requested)
    generate_animation(state, test_labels, ngmix_results, config)
    
    # Print summary
    print_summary(config, nn_results, ngmix_results)


if __name__ == "__main__":
    main()