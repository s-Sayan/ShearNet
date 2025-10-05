"""Command-line interface for evaluating trained PSF deconvolution models."""

import os
import argparse
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state
import matplotlib.pyplot as plt

from ...config.config_handler import Config
from ...core.dataset import generate_dataset, split_combined_images
from ..core.models import PSFDeconvolutionNet, EnhancedPSFDeconvNet, ResearchBackedPSFDeconvolutionUNet
from ..methods.galsim_deconv import deconvolve, deconvolve_single
from ..utils.metrics import eval_model, eval_galsim_deconv, compare_methods, _eval_batch_jit, normalize_data, inverse_normalized_data
import time
from ..utils.plot_helpers import plot_comparison, plot_spatial_residuals

# ANSI color codes for pretty printing
BOLD = '\033[1m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'

def create_parser():
    """Create argument parser for deconv evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PSF deconvolution model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate and plot comparison
  shearnet-eval-deconv --model_name deconv_unet --plot
  
  # Override test samples
  shearnet-eval-deconv --model_name deconv_unet --test_samples 1000 --plot
        """
    )
    
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Name of the model to load.')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for test data generation (overrides config).')
    parser.add_argument('--normalized', action='store_const', const=True, 
                       help='normalized data')
    parser.add_argument('--test_samples', type=int, default=None, 
                       help='Number of test samples (overrides config).')
    parser.add_argument('--plot', action='store_const', const=True, 
                       help='Generate evaluation plots')
    parser.add_argument('--num_plot_samples', type=int, default=5,
                       help='Number of samples to show in plots')

    return parser

def warm_up_model(state, galaxy_shape, psf_shape):
    """Warm up the model to trigger JIT compilation."""
    print("Warming up model...")
    dummy_galaxy = jnp.ones((1,) + galaxy_shape)
    dummy_psf = jnp.ones((1,) + psf_shape)
    
    # Trigger compilation
    _ = state.apply_fn(state.params, dummy_galaxy, dummy_psf, deterministic=True)
    print("Model warm-up complete")

def generate_neural_predictions(state, galaxy_images, psf_images, batch_size=64):
    """Generate predictions using the neural network."""
    # Pre-compile the function
    sample_galaxy = galaxy_images[:1]
    sample_psf = psf_images[:1]
    _ = _eval_batch_jit(state, sample_galaxy, sample_psf)  # Trigger compilation
    
    predictions = []
    for i in range(0, len(galaxy_images), batch_size):
        batch_galaxy = galaxy_images[i:i + batch_size]
        batch_psf = psf_images[i:i + batch_size]
        
        batch_preds = _eval_batch_jit(state, batch_galaxy, batch_psf)
        predictions.append(batch_preds)
    
    return jnp.concatenate(predictions, axis=0)

def main():
    """Main function for deconv model evaluation."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Get paths
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')

    os.makedirs(default_save_path, exist_ok=True)
    os.makedirs(default_plot_path, exist_ok=True)
    
    model_name = args.model_name
    model_config_path = os.path.join(default_plot_path, model_name, 'training_config.yaml')
    
    if os.path.exists(model_config_path):
        print(f"\nLoading model config from: {model_config_path}")
        config = Config(model_config_path)
        config.print_eval_config()
        
        # Get values from config with overrides
        seed = args.seed if args.seed is not None else config.get('evaluation.seed', config.get('dataset.seed'))
        test_samples = args.test_samples if args.test_samples is not None else config.get('evaluation.test_samples', 1000)
        model_type = config.get('deconv.model_type', 'unet')
        psf_sigma = config.get('dataset.psf_sigma')
        nse_sd = config.get('dataset.nse_sd')
        normalized = config.get('dataset.normalized')
        exp = config.get('dataset.exp')
        stamp_size = config.get('dataset.stamp_size')
        pixel_size = config.get('dataset.pixel_size')
        apply_psf_shear = config.get('dataset.apply_psf_shear', False)
        psf_shear_range = config.get('dataset.psf_shear_range', 0.05)
        plot = args.plot if args.plot is not None else config.get('plotting.plot', True)
    else:
        raise FileNotFoundError(f"No training config found at {model_config_path}")
    
    load_path = os.path.abspath(default_save_path)
    plot_path = os.path.abspath(default_plot_path)

    print(f"\nGenerating test dataset...")
    
    combined_images, labels, obs = generate_dataset(
        test_samples, psf_sigma, exp=exp, seed=seed, npix=stamp_size, 
        scale=pixel_size, nse_sd=nse_sd, return_clean=True, return_psf=True, return_obs=True, 
        apply_psf_shear=apply_psf_shear, psf_shear_range=psf_shear_range
    )
    
    # Split into galaxy, psf, and clean (target) images
    test_galaxy_images, test_psf_images, test_target_images = split_combined_images(combined_images, has_psf=True, has_clean=True)

    (test_galaxy_images, test_psf_images, test_target_images), (means, stds) = normalize_data(test_galaxy_images, test_psf_images, test_target_images)
    
    print(f"Test galaxy images shape: {test_galaxy_images.shape}")
    print(f"Test PSF images shape: {test_psf_images.shape}")
    print(f"Test target images shape: {test_target_images.shape}")

    # Initialize and load model
    rng_key = random.PRNGKey(seed)
    
    if model_type == "base":
        model = PSFDeconvolutionNet()
    elif model_type == "enhanced":
        model = EnhancedPSFDeconvNet()
    elif model_type == "research_backed":
        model = ResearchBackedPSFDeconvolutionUNet()
    else:
        raise ValueError(f"Invalid model type specified: {model_type}")

    init_params = model.init(rng_key, jnp.ones_like(test_galaxy_images[0][..., None]), jnp.ones_like(test_psf_images[0][..., None]), deterministic=True)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adam(1e-3)
    )

    # Find and load checkpoint
    all_items = os.listdir(load_path)
    candidate_dirs = [
        os.path.join(load_path, item) for item in all_items
        if os.path.isdir(os.path.join(load_path, item)) and model_name in item
    ]
    
    if not candidate_dirs:
        raise FileNotFoundError(f"No checkpoint directories found for model '{model_name}'")
    
    def extract_step(dir_path):
        """Extract step number from directory name"""
        dir_name = os.path.basename(dir_path)
        try:
            if '_' in dir_name:
                last_part = dir_name.split('_')[-1]
                if last_part.isdigit():
                    return int(last_part)
            numbers = [int(s) for s in dir_name.split('_') if s.isdigit()]
            if numbers:
                return max(numbers)
            return -1
        except Exception:
            return -1

    candidate_dirs.sort(key=extract_step, reverse=True)
    chosen_dir = candidate_dirs[0]

    print(f"Loading checkpoint from directory: {chosen_dir}")
    state = checkpoints.restore_checkpoint(
        ckpt_dir=chosen_dir,
        target=state,
        prefix=model_name
    )

    warm_up_model(state, test_galaxy_images[0].shape, test_psf_images[0].shape)

    if state.params is None:
        state = checkpoints.restore_checkpoint(ckpt_dir=chosen_dir, target=state)

    if state.params is None:
        raise ValueError(f"Failed to load checkpoint from {chosen_dir}")

    print(f"Successfully loaded checkpoint from {chosen_dir}")

    # Evaluate the neural deconv model
    print(f"\n{'='*50}")
    print("Evaluating Neural Deconvolution Model")
    print(f"{'='*50}")
    
    if normalized:
        neural_results = eval_model(
            state, test_galaxy_images, test_psf_images, test_target_images, means, stds, normalized
        )
    else: 
        neural_results = eval_model(
            state, test_galaxy_images, test_psf_images, test_target_images, normalized
        )

    # Always evaluate GalSim for comparison
    print(f"\n{'='*50}")
    print("Evaluating GalSim Deconvolution")
    print(f"{'='*50}")
    
    galsim_results = eval_galsim_deconv(
        observations=obs,
        target_images=test_target_images
    )

    # Compare methods
    print(f"\n{'='*50}")
    print("Method Comparison")
    print(f"{'='*50}")
    compare_methods(neural_results, galsim_results)

    # Generate plots if requested
    if args.plot or plot:
        print(f"\nGenerating evaluation plots...")
        df_plot_path = os.path.join(plot_path, model_name)
        os.makedirs(df_plot_path, exist_ok=True)
        
        # Generate neural network predictions
        print("Generating neural network predictions...")
        neural_predictions = generate_neural_predictions(
            state, test_galaxy_images, test_psf_images
        )
        
        # Get galsim predictions from results
        galsim_predictions = galsim_results['predictions']
        
        # Ensure predictions have correct shape
        if neural_predictions.ndim == 4:
            neural_predictions = neural_predictions.squeeze(-1)
        if galsim_predictions.ndim == 4:
            galsim_predictions = galsim_predictions.squeeze(-1)
        if test_target_images.ndim == 4:
            test_target_images = test_target_images.squeeze(-1)
        
        print(f"Neural predictions shape: {neural_predictions.shape}")
        print(f"GalSim predictions shape: {galsim_predictions.shape}")
        print(f"Target images shape: {test_target_images.shape}")
        
        # Plot comparison
        print("Creating comparison plot...")
        comparison_path = os.path.join(df_plot_path, "comparison.png")
        plot_comparison(
            test_target_images[:args.num_plot_samples],
            neural_predictions[:args.num_plot_samples],
            galsim_predictions[:args.num_plot_samples],
            num_samples=args.num_plot_samples,
            save_path=comparison_path
        )

        # Plot spatial residuals
        print("Creating spatial residuals heat map...")
        spatial_residuals_path = os.path.join(df_plot_path, "spatial_residual.png")
        plot_spatial_residuals(
            test_target_images,
            neural_predictions,
            galsim_predictions,
            path=spatial_residuals_path,
            title="Spatial Deconvolution Bias Analysis"
        )
        
        print(f"Plots saved to: {df_plot_path}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()