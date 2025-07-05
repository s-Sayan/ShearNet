"""Command-line interface for evaluating trained ShearNet models."""

import os
import argparse
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state

from ..config.config_handler import Config
from ..core.dataset import generate_dataset
from ..core.models import OriginalGalaxyNN, EnhancedGalaxyNN, OriginalGalaxyResNet, GalaxyResNet, ResearchBackedGalaxyResNet
from ..utils.metrics import eval_model, eval_ngmix, eval_mcal, remove_nan_preds_multi
from ..utils.plot_helpers import (
    plot_residuals, 
    visualize_samples, 
    plot_true_vs_predicted, 
    animate_model_epochs
)

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
    
    return parser


def main():
    """Main function for model evaluation."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Get the SHEARNET_DATA_PATH environment variable
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
    
    # Set default paths
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')

    # Ensure the directories exist
    os.makedirs(default_save_path, exist_ok=True)
    os.makedirs(default_plot_path, exist_ok=True)
    
    if args.model_name is None:
        if args.config is not None:
            model_name = Config(args.config).get('output.model_name')
            if model_name is None:
                raise ValueError("Model name not found in the configuration file.")
        else:
            raise ValueError("Either --model_name or --config must be provided.")
    else:
        model_name = args.model_name

    model_config_path = os.path.join(default_plot_path, model_name, 'training_config.yaml')
    
    if os.path.exists(model_config_path):
        print(f"\nLoading model config from: {model_config_path}")
        config = Config(model_config_path)
        config.print_eval_config()
        
        # Get values from config
        seed = args.seed if args.seed is not None else config.get('evaluation.seed')
        test_samples = args.test_samples if args.test_samples is not None else config.get('evaluation.test_samples')
        nn = config.get('model.type')
        psf_sigma = config.get('dataset.psf_sigma')
        nse_sd = config.get('dataset.nse_sd')
        exp = config.get('dataset.exp')
        mcal = args.mcal or config.get('comparison.ngmix', False)
        plot = args.plot or config.get('plotting.plot', False)
        plot_animation = args.plot_animation or config.get('plotting.animation', False)
    else:
        raise FileNotFoundError(f"No training config found at {model_config_path}")
    
    load_path = os.path.abspath(default_save_path)
    plot_path = os.path.abspath(default_plot_path)

    # Generate test data
    test_images, test_labels, test_obs = generate_dataset(
        test_samples, psf_sigma, exp=exp, seed=seed, nse_sd=nse_sd, return_obs=True
    )
    print(f"Shape of test images: {test_images.shape}")
    print(f"Shape of test labels: {test_labels.shape}")

    # Extract PSF images and create 2-channel input for all models
    print("Creating 2-channel input (galaxy + PSF)...")
    psf_images = np.array([obs.psf.image for obs in test_obs])
    test_images_2channel = np.stack([test_images, psf_images], axis=-1)
    print(f"Shape of 2-channel test images: {test_images_2channel.shape}")
    model_input = test_images_2channel

    # Initialize the model and its parameters
    rng_key = random.PRNGKey(seed)
    
    # Model selection
    if nn == "cnn":
        model = OriginalGalaxyNN()
    elif nn == "dev_cnn":
        model = EnhancedGalaxyNN()
    elif nn == "resnet":
        model = OriginalGalaxyResNet()
    elif nn == "dev_resnet":
        model = GalaxyResNet()
    elif nn == "research_backed":
        model = ResearchBackedGalaxyResNet()
    else:
        raise ValueError("Invalid model type specified.")
        
    init_params = model.init(rng_key, jnp.ones_like(model_input[0]))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adam(1e-3)
    )

    # Check for directories that start with model_name
    matching_dirs = [
        d for d in os.listdir(load_path) 
        if os.path.isdir(os.path.join(load_path, d)) and d.startswith(model_name)
    ]

    # Print the number of matching directories
    print(f"Number of matching directories found: {len(matching_dirs)}")

    # Handle the case when no matching directories are found
    if not matching_dirs:
        raise FileNotFoundError(f"No directory found in {load_path} starting with '{model_name}'.")

    # Print the list of matching directories
    for idx, directory in enumerate(matching_dirs, start=1):
        print(f"Matching directory {idx}: {directory}")
        
    # If there's only one matching directory, use it
    if len(matching_dirs) == 1:
        model_dir = os.path.join(load_path, matching_dirs[0])
    else:
        # If multiple directories, you might want to choose the latest one
        # This assumes directory names include timestamps or are sortable
        model_dir = os.path.join(load_path, sorted(matching_dirs)[-1])

    state = checkpoints.restore_checkpoint(ckpt_dir=model_dir, target=state)
    print("Model checkpoint loaded successfully.")

    # Evaluate the model
    nn_results = eval_model(state, model_input, test_labels)

    # Compare with other methods if requested
    ngmix_results = None
    if mcal:
        ngmix_results = eval_ngmix(test_obs, test_labels, seed=1234, psf_model='gauss', gal_model='gauss')
        mcal_results = eval_mcal(test_images, test_labels, psf_sigma)
        ngmix_preds = ngmix_results['preds']

    # Generate plots if requested
    if plot:
        predicted_labels = state.apply_fn(state.params, model_input, deterministic=True)
        predicted_labels, ngmix_preds, test_labels = remove_nan_preds_multi(predicted_labels, ngmix_preds, test_labels)

        df_plot_path = os.path.join(plot_path, model_name)
        os.makedirs(df_plot_path, exist_ok=True)
        
        print("\nGenerating plots...")
        print("Plotting residuals...")
        residuals_path = os.path.join(df_plot_path, "residuals")
        plot_residuals(
            test_labels,
            predicted_labels,
            path=residuals_path,
            mcal=mcal,
            preds_ngmix=ngmix_preds if ngmix_results else None
        )

        print("Plotting samples...")
        samples_path = os.path.join(df_plot_path, "samples_plot.png")
        # Use original test_images for visualization (single channel)
        visualize_samples(test_images, test_labels, predicted_labels, path=samples_path)

        print("Plotting scatter plots...")
        scatter_path = os.path.join(df_plot_path, "scatters")
        preds_ngmix = ngmix_preds if ngmix_results else None
        plot_true_vs_predicted(
            test_labels, predicted_labels, path=scatter_path, 
            mcal=mcal, preds_mcal=preds_ngmix
        )

    if plot_animation:
        print("\nGenerating animation...")
        animation_path = os.path.join(df_plot_path, "animation_plot")
        epochs = np.arange(1, 101)  # Assuming 100 epochs
        animate_model_epochs(
            test_labels, load_path, plot_path, epochs, 
            state=state, model_name=model_name, 
            mcal=mcal, preds_mcal=ngmix_results['preds'] if ngmix_results else None
        )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()