import logging
logging.getLogger('absl').setLevel(logging.ERROR)

import argparse
import os
import jax.random as random
import jax.numpy as jnp
import shutil

from ..config.config_handler import Config
from ..deconv.train import train_deconv_model
from ..core.dataset import generate_dataset, split_combined_images
from ..utils.device import get_device
from ..utils.plot_helpers import plot_learning_curve

def create_parser():
    """Create argument parser for deconv training."""
    data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
    default_save_path = os.path.join(data_path, 'model_checkpoint')
    default_plot_path = os.path.join(data_path, 'plots')
    
    parser = argparse.ArgumentParser(
        description="Train a PSF deconvolution model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  shearnet-train-deconv --config configs/deconv_experiment.yaml
  
  # Override specific values
  shearnet-train-deconv --config configs/deconv_base.yaml --epochs 100 --model_type unet
        """
    )

    # Config file argument
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (optional)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size.')
    parser.add_argument('--samples', type=int, default=None, help='Number of training samples.')
    parser.add_argument('--patience', type=int, default=None, help='Patience for early stopping.')
    parser.add_argument('--psf_sigma', type=float, default=None, help='PSF sigma for simulation.')
    parser.add_argument('--nse_sd', type=float, default=None, help='Noise std deviation.')
    parser.add_argument('--exp', type=str, default=None, help='Experiment type (ideal/superbit)')
    parser.add_argument('--learning_rate', type=float, default=None, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=None, help="Weight decay")
    parser.add_argument('--model_name', type=str, default=None, help='Model name.')
    parser.add_argument('--val_split', type=float, default=None, help='Validation split fraction')
    parser.add_argument('--eval_interval', type=int, default=None, help='Evaluate every N epochs')
    parser.add_argument('--stamp_size', type=int, default=None, help='Stamp size.')
    parser.add_argument('--pixel_size', type=float, default=None, help='Pixel size.')
    
    # Deconv-specific parameters
    parser.add_argument('--model_type', type=str, default=None, 
                       choices=['unet', 'simple'], help='Deconvolution model type')
    parser.add_argument('--loss_type', type=str, default=None,
                       choices=['mse', 'perceptual'], help='Loss function type')
    
    # PSF shear parameters
    parser.add_argument('--apply_psf_shear', action='store_const', const=True, default=None,
                       help='Apply random shear to PSF images')
    parser.add_argument('--psf_shear_range', type=float, default=None,
                       help='Maximum absolute shear value for PSF')
    
    # Paths
    parser.add_argument('--save_path', type=str, default=default_save_path,
                       help='Path to save model parameters.')
    parser.add_argument('--plot_path', type=str, default=default_plot_path,
                       help='Path to save plots.')
    
    parser.add_argument('--plot', action='store_const', const=True, default=None,
                       help='Enable plotting')
    parser.add_argument('--no-plot', action='store_const', const=False, dest='plot',
                       help='Disable plotting')
    
    return parser


def main():
    """Main function for deconv model training."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Define defaults for when not using config
    DEFAULTS = {
        'epochs': 50,
        'seed': 42,
        'batch_size': 16,
        'samples': 10000,
        'patience': 10,
        'psf_sigma': 0.25,
        'nse_sd': 1e-5,
        'exp': 'ideal',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'model_name': 'deconv_model',
        'plot': False,
        'val_split': 0.2,
        'eval_interval': 1,
        'stamp_size': 53,
        'pixel_size': 0.141,
        'model_type': 'unet',
        'loss_type': 'mse',
        'apply_psf_shear': False,
        'psf_shear_range': 0.05,
    }
    
    config = None
    
    if args.config:
        # Load configuration
        config = Config(args.config)
        config.update_from_args(args)
        
        print("\nUsing config file:", args.config)
        if any(getattr(args, k) is not None for k in DEFAULTS.keys()):
            print("With command-line overrides")
        
        # Get values from config
        samples = config.get('dataset.samples')
        psf_sigma = config.get('dataset.psf_sigma')
        nse_sd = config.get('dataset.nse_sd')
        exp = config.get('dataset.exp')
        seed = config.get('dataset.seed')
        stamp_size = config.get('dataset.stamp_size')
        pixel_size = config.get('dataset.pixel_size')
        epochs = config.get('training.epochs')
        batch_size = config.get('training.batch_size')
        patience = config.get('training.patience')
        lr = config.get('training.learning_rate')
        weight_decay = config.get('training.weight_decay')
        plot_flag = config.get('plotting.plot')
        model_name = config.get('output.model_name')
        val_split = config.get('training.val_split')
        eval_interval = config.get('training.eval_interval')
        model_type = config.get('deconv.model_type')
        loss_type = config.get('deconv.loss_type')
        apply_psf_shear = config.get('dataset.apply_psf_shear')
        psf_shear_range = config.get('dataset.psf_shear_range')
        
    else:
        # Use argparse values with defaults
        samples = args.samples if args.samples is not None else DEFAULTS['samples']
        psf_sigma = args.psf_sigma if args.psf_sigma is not None else DEFAULTS['psf_sigma']
        nse_sd = args.nse_sd if args.nse_sd is not None else DEFAULTS['nse_sd']
        exp = args.exp if args.exp is not None else DEFAULTS['exp']
        seed = args.seed if args.seed is not None else DEFAULTS['seed']
        epochs = args.epochs if args.epochs is not None else DEFAULTS['epochs']
        batch_size = args.batch_size if args.batch_size is not None else DEFAULTS['batch_size']
        patience = args.patience if args.patience is not None else DEFAULTS['patience']
        lr = args.learning_rate if args.learning_rate is not None else DEFAULTS['learning_rate']
        weight_decay = args.weight_decay if args.weight_decay is not None else DEFAULTS['weight_decay']
        plot_flag = args.plot
        model_name = args.model_name if args.model_name is not None else DEFAULTS['model_name']
        val_split = args.val_split if args.val_split is not None else DEFAULTS['val_split']
        eval_interval = args.eval_interval if args.eval_interval is not None else DEFAULTS['eval_interval']
        stamp_size = args.stamp_size if args.stamp_size is not None else DEFAULTS['stamp_size']
        pixel_size = args.pixel_size if args.pixel_size is not None else DEFAULTS['pixel_size']
        model_type = args.model_type if args.model_type is not None else DEFAULTS['model_type']
        loss_type = args.loss_type if args.loss_type is not None else DEFAULTS['loss_type']
        apply_psf_shear = args.apply_psf_shear if args.apply_psf_shear is not None else DEFAULTS['apply_psf_shear']
        psf_shear_range = args.psf_shear_range if args.psf_shear_range is not None else DEFAULTS['psf_shear_range']
        
        # Create config object with actual values
        config = Config()
        config._set_nested('dataset.samples', samples)
        config._set_nested('dataset.psf_sigma', psf_sigma)
        config._set_nested('dataset.nse_sd', nse_sd)
        config._set_nested('dataset.exp', exp)
        config._set_nested('dataset.seed', seed)
        config._set_nested('dataset.stamp_size', stamp_size)
        config._set_nested('dataset.pixel_size', pixel_size)
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
        config._set_nested('deconv.model_type', model_type)
        config._set_nested('deconv.loss_type', loss_type)
        config._set_nested('dataset.apply_psf_shear', apply_psf_shear)
        config._set_nested('dataset.psf_shear_range', psf_shear_range)

    config.print_config()

    save_path = os.path.abspath(args.save_path) if args.save_path else None
    plot_path = os.path.abspath(args.plot_path) if args.plot_path else None
    
    os.makedirs(save_path, exist_ok=True) if save_path else None
    os.makedirs(plot_path, exist_ok=True) if plot_path else None
    
    get_device()

    # Generate dataset with neural_metacal=True to get clean images
    print("Generating deconvolution dataset...")
    combined_images, labels = generate_dataset(
        samples, psf_sigma, exp=exp, seed=seed, npix=stamp_size, 
        scale=pixel_size, nse_sd=nse_sd, neural_metacal=True,
        apply_psf_shear=apply_psf_shear, psf_shear_range=psf_shear_range
    )
    
    # Split into galaxy, psf, and clean (target) images
    galaxy_images, clean_images = split_combined_images(combined_images, has_psf=False, has_clean=True)
    
    # Generate PSF images separately (we need them for deconv training)
    psf_combined_images, _ = generate_dataset(
        samples, psf_sigma, exp=exp, seed=seed, npix=stamp_size,
        scale=pixel_size, nse_sd=nse_sd, process_psf=True,
        apply_psf_shear=apply_psf_shear, psf_shear_range=psf_shear_range
    )
    _, psf_images = split_combined_images(psf_combined_images, has_psf=True, has_clean=False)
    
    print(f"Galaxy images shape: {galaxy_images.shape}")
    print(f"PSF images shape: {psf_images.shape}")
    print(f"Target images shape: {clean_images.shape}")

    rng_key = random.PRNGKey(seed)
    
    # Save config
    model_dir = os.path.join(plot_path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, 'training_config.yaml')
    config.save(config_path)
    print(f"\nTraining configuration saved to: {config_path}")

    # Train the deconvolution model
    state, train_loss, val_loss = train_deconv_model(
        galaxy_images=galaxy_images,
        psf_images=psf_images,
        target_images=clean_images,
        rng_key=rng_key,
        epochs=epochs,
        batch_size=batch_size,
        model_type=model_type,
        save_path=save_path,
        model_name=model_name,
        val_split=val_split,
        eval_interval=eval_interval,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay
    )

    if plot_flag:
        print("Plotting learning curve...")
        plot_save_path = os.path.join(plot_path, model_name, "learning_curve.png")
        plot_learning_curve(val_loss, train_loss, plot_save_path)
        
    print("Saving training and validation loss...")
    loss_path = os.path.join(plot_path, model_name, f"{model_name}_loss.npz")
    jnp.savez(loss_path, train_loss=train_loss, val_loss=val_loss)

if __name__ == "__main__":
    main()