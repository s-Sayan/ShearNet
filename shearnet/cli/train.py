"""Command-line interface for training ShearNet models."""

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import argparse
import os
import jax.random as random
import jax.numpy as jnp

from ..core.train import train_model
from ..core.dataset import generate_dataset
from ..utils.device import get_device
from ..utils.plot_helpers import plot_learning_curve


def main():
    """Main function for model training."""
    # Get the SHEARNET_DATA_PATH environment variable
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))  # Default to current directory if not set

    # Set default save_path and plot_path
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')

    # Ensure the directories exist
    os.makedirs(default_save_path, exist_ok=True)
    os.makedirs(default_plot_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a galaxy shear estimation model.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--psf_fwhm', type=float, default=1.0, help='PSF FWHM for simulation.')
    parser.add_argument('--exp', type=str, default='ideal', help='Which experiment to run')
    parser.add_argument('--nn', type=str, default='mlp', help='Which model to use')
    parser.add_argument('--save', action='store_true', help='Flag to save a model')
    parser.add_argument('--save_path', type=str, default=default_save_path, help='Path to save the model parameters.')
    parser.add_argument('--model_name', type=str, default='my_model', help='Name of the model.')
    parser.add_argument('--plot', action='store_true', help='Flag to plot learning curve')
    parser.add_argument('--plot_training_loss', action='store_true', help='Flag to plot learning curve')
    parser.add_argument('--save_loss', action='store_true', help='Flag to save training and validation loss')
    parser.add_argument('--plot_path', type=str, default=default_plot_path, help='Path to save the learning curve plot.')
    args = parser.parse_args()

    # Ensure the save_path is absolute
    save_path = os.path.abspath(args.save_path) if args.save_path else None
    plot_path = os.path.abspath(args.plot_path) if args.plot_path else None

    # Ensure directories exist
    os.makedirs(save_path, exist_ok=True) if save_path else None
    os.makedirs(plot_path, exist_ok=True) if plot_path else None

    get_device()

    train_images, train_labels = generate_dataset(args.samples, args.psf_fwhm, exp=args.exp, seed=args.seed)
    rng_key = random.PRNGKey(args.seed)
    print(f"Shape of training images: {train_images.shape}")
    print(f"Shape of training labels: {train_labels.shape}")
    
    state, train_loss, val_loss = train_model(
                                    train_images,
                                    train_labels,
                                    rng_key,
                                    epochs=args.epochs,
                                    batch_size=args.batch_size,
                                    nn=args.nn,
                                    save_path=save_path,
                                    model_name=args.model_name,
                                    val_split=0.2, # validation split fraction
                                    eval_interval=1, 
                                    patience=args.patience #for early stopping
                                )

    print("Plotting learning curve...")
    plot_path = os.path.join(args.plot_path, args.model_name, "learning_curve.png") if args.plot_path else None
    plot_learning_curve(val_loss, train_loss, plot_path)
    #if args.plot_training_loss:
    #    print("Plotting training loss...")
    #    plot_path = os.path.join(args.plot_path, args.model_name, "training_loss.png") if args.plot_path else None
    #    plot_learning_curve(train_loss, plot_path)
    print("Saving training and validation loss...")
    loss_path = os.path.join(args.plot_path, args.model_name, f"{args.model_name}_loss.npz") if save_path else None
    jnp.savez(loss_path, train_loss=train_loss, val_loss=val_loss) if loss_path else None


if __name__ == "__main__":
    main()