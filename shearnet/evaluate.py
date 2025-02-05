import argparse
import os
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state
from shearnet.train import loss_fn
from shearnet.dataset import generate_dataset
from shearnet.models import SimpleGalaxyNN
from shearnet.mcal import mcal_preds
from shearnet.plot_helpers import plot_residuals, visualize_samples, plot_true_vs_predicted, animate_model_epochs
import jax
import matplotlib.pyplot as plt


def loss_fn_mcal(images, labels, psf_fwhm):
    """Calculate the loss for the MCAL model."""

    preds = mcal_preds(images, psf_fwhm)
    loss = optax.l2_loss(preds, labels).mean()

    return loss, preds

def eval_mcal(test_images, test_labels, psf_fwhm, batch_size=32):
    """Evaluate the model on the entire test set."""
    total_loss = 0
    total_samples = 0
    total_bias = 0


    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        loss, preds = loss_fn_mcal(batch_images, batch_labels, psf_fwhm)
        
        batch_bias = (preds - batch_labels).mean()
        batch_size_actual = len(batch_images)
        total_loss += loss * batch_size_actual
        total_bias += batch_bias * batch_size_actual
        total_samples += batch_size_actual

    avg_loss = total_loss / total_samples
    avg_bias = total_bias / total_samples
    print(f"Mean Squared Error (MSE) from Metacalibration: {avg_loss}")
    print(f"Average Bias from Metacalibration: {avg_bias}")

@jax.jit
def eval_step(state, images, labels):
    """Evaluate the model on a single batch."""
    loss = loss_fn(state, state.params, images, labels)  # Reuse the training loss function
    preds = state.apply_fn(state.params, images)
    return loss, preds

def eval_model(state, test_images, test_labels, batch_size=32):
    """Evaluate the model on the entire test set."""
    total_loss = 0
    total_samples = 0
    total_bias = 0

    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        loss, preds = eval_step(state, batch_images, batch_labels)
        
        batch_bias = (preds - batch_labels).mean()
        batch_size_actual = len(batch_images)
        total_loss += loss * batch_size_actual
        total_bias += batch_bias * batch_size_actual
        total_samples += batch_size_actual

    avg_loss = total_loss / total_samples
    avg_bias = total_bias / total_samples
    print(f"Mean Squared Error (MSE) from NN: {avg_loss}")
    print(f"Average Bias from NN: {avg_bias}")

def main():
    # Get the SHEARNET_DATA_PATH environment variable
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))  # Default to current directory if not set

    # Set default save_path and plot_path
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')

    # Ensure the directories exist
    os.makedirs(default_save_path, exist_ok=True)
    os.makedirs(default_plot_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Evaluate a trained galaxy shear estimation model.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--load_path', type=str, default=default_save_path, help='Path to load the model parameters.')
    parser.add_argument('--model_name', type=str, default='my_model', help='Path to load the model parameters.')
    parser.add_argument('--test_samples', type=int, default=1000, help='Number of test samples.')
    parser.add_argument('--psf_fwhm', type=float, default=1.0, help='PSF FWHM for simulation.')
    parser.add_argument('--exp', type=str, default='ideal', help='Which experiment to run')
    parser.add_argument('--mcal', action='store_true', help='If you want mcal MSE')
    parser.add_argument('--plot', action='store_true', help='Flag to plot')
    parser.add_argument('--plot_residuals', action='store_true', help='Plot residuals.')
    parser.add_argument('--plot_samples', action='store_true', help='Plot samples.')
    parser.add_argument('--plot_scatter', action='store_true', help='Plot true vs. predicted scatterplots.')
    parser.add_argument('--plot_animation', action='store_true', help='Plot animation of scatter plot.')
    parser.add_argument('--combined_residuals', action='store_true', help='Plot combined residuals for e1 and e2.')
    parser.add_argument('--plot_path', type=str, default=default_plot_path, help='Path to save plots.')
    args = parser.parse_args()

    load_path = os.path.abspath(args.load_path)

    # Generate test data
    test_images, test_labels = generate_dataset(args.samples, args.psf_fwhm, exp=args.exp)
    print(f"Shape of test images: {test_images.shape}")
    print(f"Shape of test labels: {test_labels.shape}")

    # Initialize the model and its parameters
    rng_key = random.PRNGKey(args.seed)
    model = SimpleGalaxyNN()
    init_params = model.init(rng_key, jnp.ones_like(test_images[0]))
    state = train_state.TrainState.create(apply_fn=model.apply, params=init_params, tx=optax.adam(1e-3))

    # Check for directories that start with args.model_name
    matching_dirs = [d for d in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, d)) and d.startswith(args.model_name)]

    # Print the number of matching directories
    print(f"Number of matching directories found: {len(matching_dirs)}")

    # Handle the case when no matching directories are found
    if not matching_dirs:
        raise FileNotFoundError(f"No directory found in {load_path} starting with '{args.model_name}'.")

    # Print the list of matching directories
    for idx, directory in enumerate(matching_dirs, start=1):
        print(f"Matching directory {idx}: {directory}")


    # Load the trained model
    state = checkpoints.restore_checkpoint(ckpt_dir=load_path, target=state, prefix=args.model_name)
    print("Model checkpoint loaded successfully.")

    # Evaluate the model
    eval_model(state, test_images, test_labels)

    if args.mcal:
        eval_mcal(test_images, test_labels, args.psf_fwhm)

    #if args.plot_residuals:
    if args.plot:
        predicted_labels = state.apply_fn(state.params, test_images)
        #import pdb; pdb.set_trace()

        df_plot_path = os.path.join(args.plot_path, args.model_name)
        os.makedirs(df_plot_path, exist_ok=True)
        print("Plotting residuals...")
        residuals_path = os.path.join(df_plot_path, "residuals_plot") if args.plot_path else None
        plot_residuals(
            test_images,
            test_labels,
            predicted_labels,
            path=residuals_path,
            mcal=args.mcal,
            psf_fwhm=args.psf_fwhm,
            combined=args.combined_residuals
        )

    #if args.plot_samples:
        print("Plotting samples...")
        samples_path = os.path.join(df_plot_path, "samples_plot.png") if args.plot_path else None
        visualize_samples(test_images, test_labels, predicted_labels, path=samples_path)

    #if args.plot_scatter:
        print("Plotting scatter plots...")
        scatter_path = os.path.join(df_plot_path, "scatter_plot") if args.plot_path else None
        preds_mcal = mcal_preds(test_images, args.psf_fwhm) if args.mcal else None
        plot_true_vs_predicted(test_labels, predicted_labels, path=scatter_path, mcal=args.mcal, preds_mcal=preds_mcal)

    if args.plot_animation:
        pass # Under development
        animation_path = os.path.join(df_plot_path, "animation_plot") if args.plot_path else None
        epochs = np.arange(1, 101)  # Assuming 100 epochs
        animate_model_epochs(test_labels, load_path, args.plot_path, epochs, state=state, model_name=args.model_name, mcal=args.mcal, preds_mcal=preds_mcal)


if __name__ == "__main__":
    main()
