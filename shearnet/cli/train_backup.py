"""Command-line interface for training ShearNet models."""

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import argparse
import os
import jax.random as random
import jax.numpy as jnp

from ..config.config_handler import Config
from ..core.train import train_model
from ..core.dataset import generate_dataset
from ..utils.device import get_device
from ..utils.plot_helpers import plot_learning_curve

def create_parser():
    """Create argument parser for training."""
    parser = argparse.ArgumentParser(
        description="Train a galaxy shear estimation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  shearnet-train
  
  # Use custom config file
  shearnet-train --config my_experiment.yaml
  
  # Override specific parameters
  shearnet-train --epochs 100 --batch_size 64
  
  # Use custom config and override some values
  shearnet-train --config base.yaml --epochs 100 --model_name experiment_v2
  
  # Save the configuration used for this run
  shearnet-train --save-config --model_name my_experiment
        """
    )
    
    # Config file argument
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (default: built-in defaults)')
    parser.add_argument('--save-config', action='store_true',
                       help='Save the configuration with the model')
    
    # Keep all existing arguments as overrides
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--samples', type=int, help='Number of training samples')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--psf_sigma', type=float, help='PSF sigma for simulation')
    parser.add_argument('--exp', type=str, help='Which experiment to run')
    parser.add_argument('--nn', type=str, default='mlp', 
                   choices=['mlp', 'cnn', 'resnet'],  # These are the actual choices
                   help='Which model to use')
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for adamw optimizer')
    parser.add_argument('--save_path', type=str, help='Path to save the model parameters')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--plot_path', type=str, help='Path to save plots')
    
    return parser

def main():
    """Main function for model training."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Update config with command-line overrides
    config.update_from_args(args)
    
    # Print final configuration
    config.print_config()
    
    # Get paths from config
    save_path = os.path.abspath(config.get('output.save_path'))
    plot_path = os.path.abspath(config.get('output.plot_path'))
    
    # Ensure directories exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)
    
    # Get device info
    get_device()

    # Generate dataset
    train_images, train_labels = generate_dataset(
        config.get('dataset.samples'),
        config.get('dataset.psf_sigma'),
        exp=config.get('dataset.exp'),
        seed=config.get('dataset.seed')
    )
    
    rng_key = random.PRNGKey(config.get('dataset.seed'))
    print(f"Shape of training images: {train_images.shape}")
    print(f"Shape of training labels: {train_labels.shape}")

    # Train model
    state, train_loss, val_loss = train_model(
        train_images,
        train_labels,
        rng_key,
        epochs=int(config.get('training.epochs')),
        batch_size=config.get('training.batch_size'),
        nn=config.get('model.type'),
        save_path=save_path,
        model_name=config.get('output.model_name'),
        val_split=config.get('training.val_split'),
        eval_interval=config.get('training.eval_interval'),
        patience=config.get('training.patience'),
        lr=config.get('training.learning_rate'),
        weight_decay=config.get('training.weight_decay')
    )
    # Save config with model if requested
    if args.save_config:
        model_dir = os.path.join(save_path, config.get('output.model_name'))
        config_path = os.path.join(model_dir, 'config.yaml')
        config.save(config_path)
        print(f"\nConfiguration saved to: {config_path}")

    print("Plotting learning curve...")
    plot_save_path = os.path.join(
        plot_path, 
        config.get('output.model_name'), 
        "learning_curve.png"
    )
    plot_learning_curve(val_loss, train_loss, plot_save_path)

    print("Saving training and validation loss...")
    loss_path = os.path.join(
        plot_path,
        config.get('output.model_name'),
        f"{config.get('output.model_name')}_loss.npz"
    )
    jnp.savez(loss_path, train_loss=train_loss, val_loss=val_loss)


if __name__ == "__main__":
    main()