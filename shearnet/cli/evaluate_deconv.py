"""Command-line interface for evaluating trained PSF deconvolution models."""

import os
import argparse
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state

from ..config.config_handler import Config
from ..core.dataset import generate_dataset, split_combined_images
from ..deconv.models import PSFDeconvolutionNet, SimplePSFDeconvNet, create_deconv_net
from ..deconv.train import generate_deconv_predictions
from ..methods.fft_deconv import fourier_deconvolve, wiener_deconvolve, richardson_lucy_deconvolve
from ..utils.deconv_metrics import eval_deconv_model, eval_fft_deconv, compare_deconv_methods
from ..utils.deconv_plots import (
    plot_deconv_samples,
    plot_deconv_metrics,
    plot_deconv_comparison,
    plot_deconv_compare_samples
)
import time

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
  # Evaluate using saved model config
  shearnet-eval-deconv --model_name deconv_unet
  
  # Override test samples and compare with FFT
  shearnet-eval-deconv --model_name deconv_unet --test_samples 1000 --compare_fft
  
  # Enable plotting
  shearnet-eval-deconv --model_name deconv_unet --plot
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
    parser.add_argument('--compare_fft', action='store_true', 
                       help='Compare with FFT deconvolution baseline')
    parser.add_argument('--fft_epsilon', type=float, default=1e-3,
                       help='Regularization parameter for FFT deconvolution')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate evaluation plots')
    parser.add_argument('--num_plot_samples', type=int, default=5,
                       help='Number of samples to show in plots')

    return parser


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
        exp = config.get('dataset.exp')
        stamp_size = config.get('dataset.stamp_size')
        pixel_size = config.get('dataset.pixel_size')
        apply_psf_shear = config.get('dataset.apply_psf_shear', False)
        psf_shear_range = config.get('dataset.psf_shear_range', 0.05)
    else:
        raise FileNotFoundError(f"No training config found at {model_config_path}")
    
    load_path = os.path.abspath(default_save_path)
    plot_path = os.path.abspath(default_plot_path)

    print(f"\nGenerating test dataset...")
    print(f"Test samples: {test_samples}")
    print(f"Model type: {model_type}")
    
    # Generate test data with neural_metacal=True to get clean targets
    combined_images, labels = generate_dataset(
        test_samples, psf_sigma, exp=exp, seed=seed, nse_sd=nse_sd, 
        npix=stamp_size, scale=pixel_size, neural_metacal=True,
        apply_psf_shear=apply_psf_shear, psf_shear_range=psf_shear_range
    )
    
    # Split into galaxy and clean images
    test_galaxy_images, test_target_images = split_combined_images(
        combined_images, has_psf=False, has_clean=True
    )
    
    # Generate PSF images separately
    psf_combined_images, _ = generate_dataset(
        test_samples, psf_sigma, exp=exp, seed=seed, nse_sd=nse_sd,
        npix=stamp_size, scale=pixel_size, process_psf=True,
        apply_psf_shear=apply_psf_shear, psf_shear_range=psf_shear_range
    )
    _, test_psf_images = split_combined_images(psf_combined_images, has_psf=True, has_clean=False)
    
    print(f"Test galaxy images shape: {test_galaxy_images.shape}")
    print(f"Test PSF images shape: {test_psf_images.shape}")
    print(f"Test target images shape: {test_target_images.shape}")

    # Initialize and load model
    rng_key = random.PRNGKey(seed)
    
    if model_type == "unet":
        model = PSFDeconvolutionNet()
    elif model_type == "simple":
        model = SimplePSFDeconvNet()
    else:
        raise ValueError(f"Invalid model type specified: {model_type}")

    print("galaxy shape"+str(jnp.ones_like(test_galaxy_images[0][..., None]).shape))
    print("psf shape"+str(jnp.ones_like(test_psf_images[0][..., None]).shape))

    init_params = model.init(rng_key, jnp.ones_like(test_galaxy_images[0][..., None]), jnp.ones_like(test_psf_images[0][..., None]), training=False)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adam(1e-3)
    )

    import pickle

    # Find checkpoint directories instead of files
    all_items = os.listdir(load_path)
    print(f"All items in checkpoint directory: {all_items}")

    # Look for directories that might be checkpoints
    candidate_dirs = []
    for item in all_items:
        full_path = os.path.join(load_path, item)
        if os.path.isdir(full_path):
            # Check if directory name matches our pattern
            if model_name in item:
                candidate_dirs.append(full_path)

    print(f"Candidate directories for model '{model_name}': {candidate_dirs}")

    if not candidate_dirs:
        # Try looking for best/final versions
        for suffix in ['_best', '_final']:
            prefixed_name = f"{model_name}{suffix}"
            for item in all_items:
                full_path = os.path.join(load_path, item)
                if os.path.isdir(full_path) and prefixed_name in item:
                    candidate_dirs.append(full_path)
        
        print(f"Directories after checking suffixes: {candidate_dirs}")

    if not candidate_dirs:
        # Last resort: look for any directory containing the model name
        for item in all_items:
            full_path = os.path.join(load_path, item)
            if os.path.isdir(full_path) and model_name in item:
                candidate_dirs.append(full_path)
        
        print(f"Directories after broad search: {candidate_dirs}")
        if not candidate_dirs:
            raise FileNotFoundError(
                f"No checkpoint directories found in {load_path} for model '{model_name}'. "
                f"Items present: {all_items}"
            )

    # Extract step numbers from directory names
    def extract_step(dir_path):
        """Extract step number from directory name"""
        dir_name = os.path.basename(dir_path)
        try:
            # Strategy 1: Look for numbers at the end of directory name
            if '_' in dir_name:
                last_part = dir_name.split('_')[-1]
                if last_part.isdigit():
                    return int(last_part)
            
            # Strategy 2: Extract all numbers and take the last one
            numbers = [int(s) for s in dir_name.split('_') if s.isdigit()]
            if numbers:
                return max(numbers)
                
            return -1  # Couldn't determine step number
        except Exception as e:
            print(f"Warning: Error extracting step from {dir_name}: {str(e)}")
            return -1

    # Sort directories by step number (highest first)
    candidate_dirs.sort(key=extract_step, reverse=True)
    chosen_dir = candidate_dirs[0]

    # Use Flax's built-in checkpoint restoration
    print(f"Loading checkpoint from directory: {chosen_dir}")
    state = checkpoints.restore_checkpoint(
        ckpt_dir=chosen_dir,
        target=state,
        prefix=model_name  # This should match the prefix used when saving
    )

    # If the above doesn't work, try without prefix
    if state.params is None:
        print("Trying without prefix...")
        state = checkpoints.restore_checkpoint(
            ckpt_dir=chosen_dir,
            target=state
        )

    if state.params is None:
        raise ValueError(f"Failed to load checkpoint from {chosen_dir}")

    print(f"Successfully loaded checkpoint from {chosen_dir}")

    # Evaluate the neural deconv model
    print(f"\n{'='*50}")
    print("Evaluating Neural Deconvolution Model")
    print(f"{'='*50}")
    
    neural_results = eval_deconv_model(
        state, test_galaxy_images, test_psf_images, test_target_images
    )

    # Compare with FFT deconvolution if requested
    if args.compare_fft:
        print(f"\n{'='*50}")
        print("Evaluating FFT Deconvolution Baseline")
        print(f"{'='*50}")
        
        fft_results = eval_fft_deconv(
            test_galaxy_images, test_psf_images, test_target_images, 
            epsilon=args.fft_epsilon
        )
        
        print(f"\n{'='*50}")
        print("Comparison Summary")
        print(f"{'='*50}")
        
        compare_deconv_methods(neural_results, fft_results)

    # After neural evaluation
    if args.plot:
        print(f"\nGenerating evaluation plots...")
        df_plot_path = os.path.join(plot_path, model_name)
        os.makedirs(df_plot_path, exist_ok=True)
        
        # Generate neural network predictions
        neural_predictions = generate_deconv_predictions(
            state, test_galaxy_images, test_psf_images
        )
        
        # generate FFT predictions for comparison
        fft_start = time.time()
        fft_predictions = fourier_deconvolve(test_galaxy_images, test_psf_images, 
                                            args.fft_epsilon)
        fft_time = time.time() - fft_start
        
        # Calculate FFT metrics
        fft_mse = float(jnp.mean((fft_predictions - test_target_images) ** 2))
        fft_mae = float(jnp.mean(jnp.abs(fft_predictions - test_target_images)))
        
        # generate Weiner predictions for comparison
        weiner_start = time.time()
        weiner_predictions = wiener_deconvolve(test_galaxy_images, test_psf_images, nse_sd)
        weiner_time = time.time() - weiner_start
        
        # Calculate Weiner metrics
        weiner_mse = float(jnp.mean((weiner_predictions - test_target_images) ** 2))
        weiner_mae = float(jnp.mean(jnp.abs(weiner_predictions - test_target_images)))

        # generate RL predictions for comparison
        richardson_lucy_start = time.time()
        richardson_lucy_predictions = richardson_lucy_deconvolve(test_galaxy_images, test_psf_images)
        richardson_lucy_time = time.time() - richardson_lucy_start
        
        # Calculate RL metrics
        richardson_lucy_mse = float(jnp.mean((richardson_lucy_predictions - test_target_images) ** 2))
        richardson_lucy_mae = float(jnp.mean(jnp.abs(richardson_lucy_predictions - test_target_images)))
        
        # Print FFT metrics
        print(f"\n{BOLD}=== FFT Deconvolution Results ==={END}")
        print(f"Evaluation Time: {BOLD}{CYAN}{fft_time:.2f} seconds{END}")
        print(f"Mean Squared Error (MSE): {BOLD}{YELLOW}{fft_mse:.6e}{END}")
        print(f"Mean Absolute Error (MAE): {BOLD}{YELLOW}{fft_mae:.6e}{END}")

        # Print weiner metrics
        print(f"\n{BOLD}=== Weiner Deconvolution Results ==={END}")
        print(f"Evaluation Time: {BOLD}{CYAN}{weiner_time:.2f} seconds{END}")
        print(f"Mean Squared Error (MSE): {BOLD}{YELLOW}{weiner_mse:.6e}{END}")
        print(f"Mean Absolute Error (MAE): {BOLD}{YELLOW}{weiner_mae:.6e}{END}")

        # Print RL metrics
        print(f"\n{BOLD}=== RL Deconvolution Results ==={END}")
        print(f"Evaluation Time: {BOLD}{CYAN}{richardson_lucy_time:.2f} seconds{END}")
        print(f"Mean Squared Error (MSE): {BOLD}{YELLOW}{richardson_lucy_mse:.6e}{END}")
        print(f"Mean Absolute Error (MAE): {BOLD}{YELLOW}{richardson_lucy_mae:.6e}{END}")
        
        # Plot sample results
        print("Plotting sample deconvolutions...")
        samples_path = os.path.join(df_plot_path, "deconv_samples.png")
        plot_deconv_samples(
            test_galaxy_images[:args.num_plot_samples],
            test_psf_images[:args.num_plot_samples], 
            test_target_images[:args.num_plot_samples],
            neural_predictions[:args.num_plot_samples],
            num_samples=args.num_plot_samples,
            path=samples_path
        )

        # Plot sample results
        print("Plotting comparison deconvolutions...")
        samples_path = os.path.join(df_plot_path, "deconv_comparison_samples.png")
        plot_deconv_compare_samples(
            test_target_images[:args.num_plot_samples],
            neural_predictions[:args.num_plot_samples],
            fft_predictions[:args.num_plot_samples],
            weiner_predictions[:args.num_plot_samples],
            richardson_lucy_predictions[:args.num_plot_samples],
            num_samples=args.num_plot_samples,
            path=samples_path
        )
        
        # Plot metrics (updated to include FFT)
        print("Plotting deconvolution metrics...")
        metrics_path = os.path.join(df_plot_path, "deconv_metrics.png")
        plot_deconv_metrics(
            test_target_images, 
            neural_predictions,
            fft_predictions,  # ADDED FFT predictions
            path=metrics_path, 
            title="Deconvolution Performance Comparison"
        )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()