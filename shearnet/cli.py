import argparse
import os
from shearnet.train import train_model
from shearnet.dataset import generate_dataset
from shearnet.utils import get_device
import jax.random as random
from shearnet.plot_helpers import plot_learning_curve

def main():
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
    parser.add_argument('--psf_fwhm', type=float, default=1.0, help='PSF FWHM for simulation.')
    parser.add_argument('--save', action='store_true', help='Flag to save a model')
    parser.add_argument('--save_path', type=str, default=default_save_path, help='Path to save the model parameters.')
    parser.add_argument('--model_name', type=str, default='my_model', help='Name of the model.')
    parser.add_argument('--plot', action='store_true', help='Flag to plot learning curve')
    parser.add_argument('--plot_path', type=str, default=default_plot_path, help='Path to save the learning curve plot.')
    args = parser.parse_args()

    # Ensure the save_path is absolute
    save_path = os.path.abspath(args.save_path) if args.save_path else None
    plot_path = os.path.abspath(args.plot_path) if args.plot_path else None

    # Ensure directories exist
    os.makedirs(save_path, exist_ok=True) if save_path else None
    os.makedirs(plot_path, exist_ok=True) if plot_path else None

    get_device()
    
    train_images, train_labels = generate_dataset(args.samples, args.psf_fwhm)
    rng_key = random.PRNGKey(args.seed)
    print(f"Shape of training images: {train_images.shape}")
    print(f"Shape of training labels: {train_labels.shape}")
    
    stae, epoch_loss = train_model(
                            train_images,
                            train_labels,
                            rng_key,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            save_path=save_path if args.save else None,
                            model_name=args.model_name
                        )

    if args.plot:
        plot_path = os.path.join(args.plot_path, args.model_name, "learning_curve.png") if args.plot_path else None
        plot_learning_curve(epoch_loss, plot_path)


if __name__ == "__main__":
    main()