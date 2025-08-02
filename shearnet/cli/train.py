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
    # Get the SHEARNET_DATA_PATH environment variable
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
    
    # Set default save_path and plot_path
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')
    
    parser = argparse.ArgumentParser(
        description="Train a galaxy shear estimation model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Your original command (still works)
  shearnet-train --epochs 10 --batch_size 64 --samples 10000 --psf_sigma 0.25 --save --model_name cnn6 --plot --nn cnn --patience 20
  
  # Use config file
  shearnet-train --config configs/cnn6_experiment.yaml
  
  # Use config file but override specific values
  shearnet-train --config configs/cnn6_experiment.yaml --samples 20000 --model_name cnn6_big
  
  # Override multiple values
  shearnet-train --config configs/base.yaml --epochs 100 --nn resnet --save --plot
        """
    )

    # Config file argument
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    
    # For config overrides, use default=None so we can detect what user actually specified
    # When not using config, we'll use the defaults from the code
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size.')
    parser.add_argument('--samples', type=int, default=None, help='Number of training samples.')
    parser.add_argument('--patience', type=int, default=None, help='Patience for early stopping.')
    parser.add_argument('--psf_sigma', type=float, default=None, help='PSF sigma for simulation.')
    parser.add_argument('--nse_sd', type=float, default=None, help='noise sd for simulation.')
    parser.add_argument('--exp', type=str, default=None, help='Which experiment to run')
    parser.add_argument('--nn', type=str, default=None, help='Which model to use')
    parser.add_argument('--learning_rate', type=float, default=None, 
                       help="Initial learning rate for the learning rate scheduler")
    parser.add_argument('--weight_decay', type=float, default=None, 
                       help="Weight decay for adamw optimizer")
    parser.add_argument('--model_name', type=str, default=None, help='Name of the model.')
    parser.add_argument('--val_split', type=float, default=None, 
                    help='Validation split fraction (default: 0.2)')
    parser.add_argument('--eval_interval', type=int, default=None,
                    help='Evaluate every N epochs (default: 1)')    
    # Keep defaults for paths since they're computed
    parser.add_argument('--save_path', type=str, default=default_save_path,
                       help='Path to save the model parameters.')
    parser.add_argument('--plot_path', type=str, default=default_plot_path,
                       help='Path to save the learning curve plot.')
    
    parser.add_argument('--plot', action='store_const', const=True, default=None,
                       help='Enable plotting (overrides config)')
    parser.add_argument('--no-plot', action='store_const', const=False, dest='plot',
                       help='Disable plotting (overrides config)')
    parser.add_argument('--stamp_size', type=int, default=None, 
                   help='Stamp size of the training data.')
    parser.add_argument('--pixel_size', type=float, default=None, 
                   help='Pixel size of the training data.')
    
    return parser


def main():
    """Main function for model training."""
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Define defaults for when not using config
    DEFAULTS = {
        'epochs': 10,
        'seed': 42,
        'batch_size': 32,
        'samples': 10000,
        'patience': 10,
        'psf_sigma': 0.25,
        'nse_sd': 1e-5,
        'exp': 'ideal',
        'nn': 'mlp',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'model_name': 'my_model',
        'plot': False,
        'val_split': 0.2,      
        'eval_interval': 1,   
        'stamp_size': 53,   
        'pixel_size': 0.141,    
    }
    config = None 
    
    if args.config:
        # Load configuration
        config = Config(args.config)
        
        # Update config with command-line overrides (only non-None values)
        config.update_from_args(args)
        
        # Print configuration
        print("\nUsing config file:", args.config)
        if any(getattr(args, k) is not None for k in DEFAULTS.keys()):
            print("With command-line overrides")
        
        
        # Get values from config (after overrides applied)
        samples = config.get('dataset.samples')
        psf_sigma = config.get('dataset.psf_sigma')
        nse_sd = config.get('dataset.nse_sd')
        exp = config.get('dataset.exp')
        seed = config.get('dataset.seed')
        stamp_size = config.get('dataset.stamp_size') 
        pixel_size = config.get('dataset.pixel_size')
        epochs = config.get('training.epochs')
        batch_size = config.get('training.batch_size')
        nn = config.get('model.type')
        patience = config.get('training.patience')
        lr = config.get('training.learning_rate')
        weight_decay = config.get('training.weight_decay')
        plot_flag = config.get('plotting.plot')
        model_name = config.get('output.model_name')
        val_split = config.get('training.val_split')
        eval_interval = config.get('training.eval_interval')
        
    else:
        # Use argparse values with defaults (original behavior)
        samples = args.samples if args.samples is not None else DEFAULTS['samples']
        psf_sigma = args.psf_sigma if args.psf_sigma is not None else DEFAULTS['psf_sigma']
        nse_sd = args.nse_sd if args.nse_sd is not None else DEFAULTS['nse_sd']
        exp = args.exp if args.exp is not None else DEFAULTS['exp']
        seed = args.seed if args.seed is not None else DEFAULTS['seed']
        epochs = args.epochs if args.epochs is not None else DEFAULTS['epochs']
        batch_size = args.batch_size if args.batch_size is not None else DEFAULTS['batch_size']
        nn = args.nn if args.nn is not None else DEFAULTS['nn']
        patience = args.patience if args.patience is not None else DEFAULTS['patience']
        lr = args.learning_rate if args.learning_rate is not None else DEFAULTS['learning_rate']
        weight_decay = args.weight_decay if args.weight_decay is not None else DEFAULTS['weight_decay']
        plot_flag = args.plot
        model_name = args.model_name if args.model_name is not None else DEFAULTS['model_name']
        val_split = args.val_split if args.val_split is not None else DEFAULTS['val_split']
        eval_interval = args.eval_interval if args.eval_interval is not None else DEFAULTS['eval_interval']
        stamp_size = args.stamp_size if args.stamp_size is not None else DEFAULTS['stamp_size']
        pixel_size = args.pixel_size if args.pixel_size is not None else DEFAULTS['pixel_size']
        
        # Always create a config object with the actual values being used
        config = Config()  # Start with defaults
        # Update with actual values being used
        config._set_nested('dataset.samples', samples)
        config._set_nested('dataset.psf_sigma', psf_sigma)
        config._set_nested('dataset.nse_sd', nse_sd)
        config._set_nested('dataset.exp', exp)
        config._set_nested('dataset.seed', seed)
        config._set_nested('dataset.stamp_size', stamp_size)
        config._set_nested('dataset.pixel_size', pixel_size)
        config._set_nested('model.type', nn)
        config._set_nested('training.epochs', epochs)
        config._set_nested('training.batch_size', batch_size)
        config._set_nested('training.learning_rate', lr)
        config._set_nested('training.weight_decay', weight_decay)
        config._set_nested('training.patience', patience)
        config._set_nested('training.val_split', val_split)
        config._set_nested('training.eval_interval', eval_interval)
        config._set_nested('output.model_name', model_name)
        config._set_nested('output.save_path', args.save_path)
        config._set_nested('output.plot_path', args.plot_path)
        config._set_nested('plotting.plot', plot_flag)

    config.print_config()    
    # Rest of the code remains the same...
    save_path = os.path.abspath(args.save_path) if args.save_path else None
    plot_path = os.path.abspath(args.plot_path) if args.plot_path else None
    
    os.makedirs(save_path, exist_ok=True) if save_path else None
    os.makedirs(plot_path, exist_ok=True) if plot_path else None
    
    get_device()

    train_images, train_labels = generate_dataset(samples, psf_sigma, exp=exp, seed=seed, npix=stamp_size, scale=pixel_size, nse_sd=nse_sd)
    rng_key = random.PRNGKey(seed)
    print(f"Shape of training images: {train_images.shape}")
    print(f"Shape of training labels: {train_labels.shape}")
    
    model_dir = os.path.join(plot_path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, 'training_config.yaml')
    config.save(config_path)
    print(f"\nTraining configuration saved to: {config_path}")

    state, train_loss, val_loss = train_model(
                                    train_images,
                                    train_labels,
                                    rng_key,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    nn=nn,
                                    save_path=save_path,
                                    model_name=model_name,
                                    val_split=val_split, # validation split fraction
                                    eval_interval=eval_interval, 
                                    patience=patience, #for early stopping
                                    lr=lr, weight_decay=weight_decay #optimizer details
                                )

    if plot_flag:
        print("Plotting learning curve...")
        plot_save_path = os.path.join(plot_path, model_name, "learning_curve.png") if plot_path else None
        plot_learning_curve(val_loss, train_loss, plot_save_path)
        
    print("Saving training and validation loss...")
    loss_path = os.path.join(plot_path, model_name, f"{model_name}_loss.npz") if plot_path else None
    jnp.savez(loss_path, train_loss=train_loss, val_loss=val_loss) if loss_path else None

if __name__ == "__main__":
    main()