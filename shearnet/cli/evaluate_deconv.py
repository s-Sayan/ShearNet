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
from ..methods.ngmix_deconv import ngmix_parametric_deconvolve, ngmix_exp_deconvolve, ngmix_gauss_deconvolve, ngmix_dev_deconvolve
from ..utils.deconv_metrics import eval_deconv_model, eval_ngmix_deconv, compare_deconv_methods
from ..utils.deconv_plots import (
    plot_deconv_samples,
    plot_deconv_metrics,
    plot_deconv_comparison,
    plot_deconv_compare_samples
)
import time
from ..deconv.research_backed_unet import create_research_backed_deconv_unet

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
    
    parser.add_argument('--compare_ngmix', action='store_const', const=True, default=None, help='Compare with NGmix deconvolution baseline')
    
    parser.add_argument('--fft_epsilon', type=float, default=1e-3,
                       help='Regularization parameter for FFT deconvolution')
    parser.add_argument('--plot', action='store_true', 
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
    _ = state.apply_fn(state.params, dummy_galaxy, dummy_psf, training=False)
    print("Model warm-up complete")

def evaluate_ngmix_deconv_methods(galaxy_images, psf_images, target_images):
    """Evaluate NGmix deconvolution methods."""
    results = {}
    
    methods = [
        ('exp', 'Exponential', ngmix_exp_deconvolve),
        ('gauss', 'Gaussian', ngmix_gauss_deconvolve), 
        ('dev', 'de Vaucouleurs', ngmix_dev_deconvolve)
    ]
    
    for model_key, model_name, method_func in methods:
        print(f"\n{BOLD}=== NGmix {model_name} Deconvolution ==={END}")
        
        start_time = time.time()
        
        try:
            predictions = method_func(galaxy_images, psf_images, n_jobs=4)
            elapsed_time = time.time() - start_time
            
            # Calculate metrics
            mse = float(jnp.mean((predictions - target_images) ** 2))
            mae = float(jnp.mean(jnp.abs(predictions - target_images)))
            
            # PSNR
            if mse > 0:
                max_val = float(jnp.max(target_images))
                psnr = 20 * jnp.log10(max_val / jnp.sqrt(mse))
            else:
                psnr = float('inf')
            
            # Bias
            bias = float(jnp.mean(predictions - target_images))
            
            results[f'ngmix_{model_key}'] = {
                'method': f'NGmix {model_name}',
                'predictions': predictions,
                'mse': mse,
                'mae': mae,
                'psnr': psnr,
                'bias': bias,
                'time_taken': elapsed_time
            }
            
            print(f"Evaluation Time: {BOLD}{CYAN}{elapsed_time:.2f} seconds{END}")
            print(f"Mean Squared Error (MSE): {BOLD}{YELLOW}{mse:.6e}{END}")
            print(f"Mean Absolute Error (MAE): {BOLD}{YELLOW}{mae:.6e}{END}")
            print(f"Peak Signal-to-Noise Ratio (PSNR): {BOLD}{CYAN}{psnr:.2f} dB{END}")
            print(f"Bias: {BOLD}{bias:+.6e}{END}")
            
        except Exception as e:
            print(f"{RED}Error with NGmix {model_name}: {e}{END}")
            results[f'ngmix_{model_key}'] = {'error': str(e)}
    
    return results

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
    elif model_type == "research_backed":
        # Extract research-backed parameters from config
        architecture = config.get('deconv.architecture', 'full')
        encoder_features = config.get('deconv.encoder_features')
        use_transformers = config.get('deconv.use_transformers', True)
        use_physics_informed = config.get('deconv.use_physics_informed', True)
        use_dense_connections = config.get('deconv.use_dense_connections', True)
        use_pyramid_fusion = config.get('deconv.use_pyramid_fusion', True)
        window_size = config.get('deconv.window_size', 8)
        num_heads = config.get('deconv.num_heads', 8)
        dropout_rate = config.get('deconv.dropout_rate', 0.1)
        
        model = create_research_backed_deconv_unet(
            architecture=architecture,
            encoder_features=encoder_features,
            use_transformers=use_transformers,
            use_physics_informed=use_physics_informed,
            use_dense_connections=use_dense_connections,
            use_pyramid_fusion=use_pyramid_fusion,
            window_size=window_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Invalid model type specified: {model_type}")

    print("galaxy shape"+str(jnp.ones_like(test_galaxy_images[0][..., None]).shape))
    print("psf shape"+str(jnp.ones_like(test_psf_images[0][..., None]).shape))

    init_params = model.init(rng_key, jnp.ones_like(test_galaxy_images[0][..., None]), jnp.ones_like(test_psf_images[0][..., None]), training=False)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adam(1e-3)
    )

    import pickle

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
    
    neural_results = eval_deconv_model(
        state, test_galaxy_images, test_psf_images, test_target_images
    )

    # Compare with NGmix deconvolution if requested
    ngmix_results = {}
    if args.compare_ngmix:
        print(f"\n{'='*50}")
        print("Evaluating NGmix Deconvolution Methods")
        print(f"{'='*50}")
        
        ngmix_results = evaluate_ngmix_deconv_methods(
            test_galaxy_images, test_psf_images, test_target_images
        )
        
        print(f"\n{'='*50}")
        print("Comparison Summary")
        print(f"{'='*50}")
        
        # Compare with best NGmix method
        best_ngmix_key = None
        best_ngmix_mse = float('inf')
        
        for key, result in ngmix_results.items():
            if 'error' not in result and result['mse'] < best_ngmix_mse:
                best_ngmix_mse = result['mse']
                best_ngmix_key = key
        
        if best_ngmix_key:
            best_ngmix_result = ngmix_results[best_ngmix_key]
            #compare_deconv_methods(neural_results, best_ngmix_result)

    # Generate plots if requested
    if args.plot:
        print(f"\nGenerating evaluation plots...")
        df_plot_path = os.path.join(plot_path, model_name)
        os.makedirs(df_plot_path, exist_ok=True)
        
        # Generate neural network predictions
        neural_predictions = generate_deconv_predictions(
            state, test_galaxy_images, test_psf_images
        )
        
        # Get NGmix predictions
        ngmix_predictions = {}
        for key, result in ngmix_results.items():
            if 'error' not in result:
                ngmix_predictions[key] = result['predictions']
        
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

        # Plot comparison between methods if we have NGmix results
        if ngmix_predictions:
            # Use the best NGmix method for comparison
            best_key = min(ngmix_predictions.keys(), 
                          key=lambda k: ngmix_results[k]['mse'])
            best_ngmix_preds = ngmix_predictions[best_key]
            
            print("Plotting method comparison...")
            comparison_path = os.path.join(df_plot_path, "deconv_method_comparison.png")
            plot_deconv_comparison(
                test_galaxy_images[:args.num_plot_samples],
                test_target_images[:args.num_plot_samples],
                neural_predictions[:args.num_plot_samples],
                best_ngmix_preds[:args.num_plot_samples],
                num_samples=args.num_plot_samples,
                path=comparison_path
            )
            
            # Plot comparison with all NGmix methods
            if len(ngmix_predictions) >= 2:
                print("Plotting comprehensive comparison...")
                comp_samples_path = os.path.join(df_plot_path, "deconv_all_methods_comparison.png")
                
                # Get predictions from different methods
                exp_preds = ngmix_predictions.get('ngmix_exp', neural_predictions)
                gauss_preds = ngmix_predictions.get('ngmix_gauss', neural_predictions)
                dev_preds = ngmix_predictions.get('ngmix_dev', neural_predictions)
                
                plot_deconv_compare_samples(
                    test_target_images[:args.num_plot_samples],
                    neural_predictions[:args.num_plot_samples],
                    exp_preds[:args.num_plot_samples],
                    gauss_preds[:args.num_plot_samples],
                    dev_preds[:args.num_plot_samples],
                    num_samples=args.num_plot_samples,
                    path=comp_samples_path
                )
        
        # Plot metrics comparison
        if ngmix_predictions:
            print("Plotting deconvolution metrics...")
            metrics_path = os.path.join(df_plot_path, "deconv_metrics.png")
            plot_deconv_metrics(
                test_target_images, 
                neural_predictions,
                best_ngmix_preds,
                path=metrics_path, 
                title="Neural vs NGmix Deconvolution Performance"
            )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()