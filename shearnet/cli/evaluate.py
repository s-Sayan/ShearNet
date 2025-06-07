"""Command-line interface for evaluating trained ShearNet models."""

import os
import argparse
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state

from ..core.dataset import generate_dataset
from ..core.models import SimpleGalaxyNN, EnhancedGalaxyNN, GalaxyResNet
from ..utils.metrics import eval_model, eval_ngmix, eval_mcal
from ..utils.plot_helpers import (
    plot_residuals, 
    visualize_samples, 
    plot_true_vs_predicted, 
    animate_model_epochs
)


def main():
    """Main function for model evaluation."""
    # Get the SHEARNET_DATA_PATH environment variable
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
    
    # Set default save_path and plot_path
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')

    # Ensure the directories exist
    os.makedirs(default_save_path, exist_ok=True)
    os.makedirs(default_plot_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Evaluate a trained galaxy shear estimation model.")
    parser.add_argument('--seed', type=int, default=58, help='Random seed for reproducibility.')
    parser.add_argument('--load_path', type=str, default=default_save_path, help='Path to load the model parameters.')
    parser.add_argument('--nn', type=str, default='mlp', choices=['mlp', 'cnn', 'resnet'], 
                        help='Neural network architecture to use.')
    parser.add_argument('--model_name', type=str, default='my_model', help='Name of the model to load.')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples.')
    parser.add_argument('--psf_sigma', type=float, default=1.0, help='PSF FWHM for simulation.')
    parser.add_argument('--exp', type=str, default='ideal', help='Which experiment to run')
    parser.add_argument('--mcal', action='store_true', help='Compare with metacalibration and NGmix')
    parser.add_argument('--plot', action='store_true', help='Generate all plots')
    parser.add_argument('--plot_path', type=str, default=default_plot_path, help='Path to save plots.')
    parser.add_argument('--plot_animation', action='store_true', help='Plot animation of scatter plot.')
    
    args = parser.parse_args()
    load_path = os.path.abspath(args.load_path)

    # Generate test data
    test_images, test_labels, test_obs = generate_dataset(
        args.test_samples, args.psf_sigma, exp=args.exp, seed=args.seed, return_obs=True
    )
    print(f"Shape of test images: {test_images.shape}")
    print(f"Shape of test labels: {test_labels.shape}")

    # Initialize the model and its parameters
    rng_key = random.PRNGKey(args.seed)
    
    # Model selection
    if args.nn == "mlp":
        model = SimpleGalaxyNN()
    elif args.nn == "cnn":
        model = EnhancedGalaxyNN()
    elif args.nn == "resnet":
        model = GalaxyResNet() 
    else:
        raise ValueError("Invalid model type specified.")
        
    init_params = model.init(rng_key, jnp.ones_like(test_images[0]))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=init_params, tx=optax.adam(1e-3)
    )

    # Check for directories that start with args.model_name
    matching_dirs = [
        d for d in os.listdir(load_path) 
        if os.path.isdir(os.path.join(load_path, d)) and d.startswith(args.model_name)
    ]

    # Print the number of matching directories
    print(f"Number of matching directories found: {len(matching_dirs)}")

    # Handle the case when no matching directories are found
    if not matching_dirs:
        raise FileNotFoundError(f"No directory found in {load_path} starting with '{args.model_name}'.")

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
    nn_results = eval_model(state, test_images, test_labels)

    # Compare with other methods if requested
    if args.mcal:
        ngmix_results = eval_ngmix(test_obs, test_labels, seed=1234, psf_model='gauss', gal_model='gauss')
        mcal_results = eval_mcal(test_images, test_labels, args.psf_sigma)

    # Generate plots if requested
    if args.plot:
        predicted_labels = state.apply_fn(state.params, test_images, deterministic=True)
        
        df_plot_path = os.path.join(args.plot_path, args.model_name)
        os.makedirs(df_plot_path, exist_ok=True)
        
        print("Plotting residuals...")
        residuals_path = os.path.join(df_plot_path, "residuals") if args.plot_path else None
        plot_residuals(
            test_labels,
            predicted_labels,
            path=residuals_path,
            mcal=args.mcal,
            preds_ngmix=ngmix_results['preds']
        )

        print("Plotting samples...")
        samples_path = os.path.join(df_plot_path, "samples_plot.png") if args.plot_path else None
        visualize_samples(test_images, test_labels, predicted_labels, path=samples_path)

        print("Plotting scatter plots...")
        scatter_path = os.path.join(df_plot_path, "scatters") if args.plot_path else None
        preds_ngmix = ngmix_results['preds'] if args.mcal else None
        plot_true_vs_predicted(
            test_labels, predicted_labels, path=scatter_path, 
            mcal=args.mcal, preds_mcal=preds_ngmix
        )

    if args.plot_animation:
        # Under development
        animation_path = os.path.join(df_plot_path, "animation_plot") if args.plot_path else None
        epochs = np.arange(1, 101)  # Assuming 100 epochs
        animate_model_epochs(
            test_labels, load_path, args.plot_path, epochs, 
            state=state, model_name=args.model_name, 
            mcal=args.mcal, preds_mcal=preds_ngmix if args.mcal else None
        )


if __name__ == "__main__":
    main()