import os
import warnings
import logging

logging.getLogger('absl').setLevel(logging.ERROR)


import argparse
import os
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state
from shearnet.train import loss_fn
from shearnet.dataset import generate_dataset
from shearnet.models import *
from shearnet.mcal import mcal_preds
from shearnet.ngmix import _get_priors, mp_fit_one, ngmix_pred
from shearnet.plot_helpers import plot_residuals, visualize_samples, plot_true_vs_predicted, animate_model_epochs
import jax
import ipdb
import matplotlib.pyplot as plt
import time


# Define styles
BOLD = '\033[1m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
GREEN = '\033[92m'
END = '\033[0m'

def loss_fn_mcal(images, labels, psf_fwhm):
    """Calculate the loss for the MCAL model."""
    preds = mcal_preds(images, psf_fwhm)
    
    # Combined loss for g1, g2
    loss = optax.l2_loss(preds[:, :2], labels[:, :2]).mean()
    
    # Per-label losses (only g1, g2 for mcal)
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], labels[:, 1]).mean(),
        'g1g2_combined': loss  # Same as combined loss
    }
    
    return loss, preds, loss_per_label

def loss_fn_ngmix(obs_list, labels, seed=1234, psf_model='gauss', gal_model='gauss'):
    """Calculate the loss for the NGmix model."""
    
    prior = _get_priors(seed)
    rng = np.random.RandomState(seed)
    datalist = mp_fit_one(obs_list, prior, rng, psf_model=psf_model, gal_model=gal_model)
    preds = ngmix_pred(datalist)
    
    # Combined loss
    loss = optax.l2_loss(preds, labels).mean()
    
    # Per-label losses
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], labels[:, 1]).mean(),
        'g1g2_combined': optax.l2_loss(preds[:, :2], labels[:, :2]).mean(),  # Combined g1,g2
        'sigma': optax.l2_loss(preds[:, 2], labels[:, 2]).mean(),
        'flux': optax.l2_loss(preds[:, 3], labels[:, 3]).mean()
    }

    return loss, preds, loss_per_label

def loss_fn_eval(state, params, images, labels):
    preds = state.apply_fn(params, images)
    
    # Combined loss (assuming preds shape matches labels shape)
    loss = optax.l2_loss(preds, labels).mean()
    
    # Per-label losses
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], labels[:, 1]).mean(),
        'g1g2_combined': optax.l2_loss(preds[:, :2], labels[:, :2]).mean(),
        'sigma': optax.l2_loss(preds[:, 2], labels[:, 2]).mean(),
        'flux': optax.l2_loss(preds[:, 3], labels[:, 3]).mean()
    }
    
    return loss, loss_per_label

def eval_mcal(test_images, test_labels, psf_fwhm):
    """Evaluate metacalibration on the entire test set at once."""
    # Get all predictions
    start_time = time.time()
    preds = mcal_preds(test_images, psf_fwhm)
    
    # Combined metrics
    loss = optax.l2_loss(preds[:, :2], test_labels[:, :2]).mean()
    bias = (preds - test_labels[:, :2]).mean()
    
    # Per-label metrics
    loss_per_label = {
        'g1': optax.l2_loss(preds[:, 0], test_labels[:, 0]).mean(),
        'g2': optax.l2_loss(preds[:, 1], test_labels[:, 1]).mean(),
        'g1g2_combined': loss
    }
    
    bias_per_label = {
        'g1': (preds[:, 0] - test_labels[:, 0]).mean(),
        'g2': (preds[:, 1] - test_labels[:, 1]).mean(),
        'g1g2_combined': bias
    }
    total_time = time.time() - start_time
    # Print results
    print("\n=== Combined Metrics (Moment-Based Approach) ===")
    print(f"Mean Squared Error (MSE) from MOM: {loss:.6e}")
    print(f"Average Bias from MOM: {bias:.6e}")
    print(f"Time taken: {total_time:.2f} seconds")
    
    print("\n=== Per-Label Metrics ===")
    label_names = ['g1', 'g2', 'g1g2_combined']
    for label in label_names:
        print(f"{label:>15}: MSE = {loss_per_label[label]:.6e}, Bias = {bias_per_label[label]:+.6e}")
    print()
    return {
        'loss': loss,
        'bias': bias,
        'loss_per_label': loss_per_label,
        'bias_per_label': bias_per_label,
        'preds': preds
    }

def eval_ngmix(test_obs, test_labels, seed=1234, psf_model='gauss', gal_model='gauss'):
    """Evaluate the model using ngmix on the entire test set."""
    start_time = time.time()

    loss, preds, loss_per_label = loss_fn_ngmix(test_obs, test_labels, seed, 
                                                psf_model=psf_model, gal_model=gal_model)
    
    # Combined metrics
    bias = (preds - test_labels).mean()
    
    # Per-label biases
    bias_per_label = {
        'g1': (preds[:, 0] - test_labels[:, 0]).mean(),
        'g2': (preds[:, 1] - test_labels[:, 1]).mean(),
        'g1g2_combined': (preds[:, :2] - test_labels[:, :2]).mean(),  # Average bias for g1,g2
        'sigma': (preds[:, 2] - test_labels[:, 2]).mean(),
        'flux': (preds[:, 3] - test_labels[:, 3]).mean()
    }
    total_time = time.time() - start_time
    
    #ipdb.set_trace()
    # Print combined metrics
    print(f"\n{BOLD}=== Combined Metrics (NGmix) ==={END}")
    print(f"Mean Squared Error (MSE) from NGmix: {BOLD}{YELLOW}{loss:.6e}{END}")
    print(f"Average Bias from NGmix: {BOLD}{YELLOW}{bias:.6e}{END}")
    print(f"Time taken: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    
    # Print per-label metrics
    print("\n=== Per-Label Metrics ===")
    label_names = ['g1', 'g2', 'g1g2_combined', 'sigma', 'flux']
    for label in label_names:
        print(f"{label:>15}: MSE = {loss_per_label[label]:.6e}, Bias = {bias_per_label[label]:+.6e}")
    print()
    return {
        'loss': loss,
        'bias': bias,
        'loss_per_label': loss_per_label,
        'bias_per_label': bias_per_label,
        'preds': preds
    }

@jax.jit
def eval_step(state, images, labels):
    """Evaluate the model on a single batch."""
    loss, loss_per_label = loss_fn_eval(state, state.params, images, labels)
    preds = state.apply_fn(state.params, images, deterministic=True)
    
    # Calculate per-label biases
    bias_per_label = {
        'g1': (preds[:, 0] - labels[:, 0]).mean(),
        'g2': (preds[:, 1] - labels[:, 1]).mean(),
        'g1g2_combined': (preds[:, :2] - labels[:, :2]).mean(),
        'sigma': (preds[:, 2] - labels[:, 2]).mean(),
        'flux': (preds[:, 3] - labels[:, 3]).mean()
    }
    
    return loss, preds, loss_per_label, bias_per_label

def eval_model(state, test_images, test_labels, batch_size=32):
    """Evaluate the model on the entire test set."""
    start_time = time.time()

    total_loss = 0
    total_samples = 0
    total_bias = 0
    
    # Initialize per-label accumulators
    total_loss_per_label = {
        'g1': 0, 'g2': 0, 'g1g2_combined': 0, 'sigma': 0, 'flux': 0
    }
    total_bias_per_label = {
        'g1': 0, 'g2': 0, 'g1g2_combined': 0, 'sigma': 0, 'flux': 0
    }
    
    all_preds = []

    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i:i + batch_size]
        batch_labels = test_labels[i:i + batch_size]
        loss, preds, loss_per_label, bias_per_label = eval_step(state, batch_images, batch_labels)
        
        all_preds.append(preds)
        
        batch_bias = (preds - batch_labels).mean()
        batch_size_actual = len(batch_images)
        
        # Accumulate combined metrics
        total_loss += loss * batch_size_actual
        total_bias += batch_bias * batch_size_actual
        total_samples += batch_size_actual
        
        # Accumulate per-label metrics
        for label in total_loss_per_label:
            total_loss_per_label[label] += loss_per_label[label] * batch_size_actual
            total_bias_per_label[label] += bias_per_label[label] * batch_size_actual

    # Calculate averages
    avg_loss = total_loss / total_samples
    avg_bias = total_bias / total_samples
    
    avg_loss_per_label = {
        label: total / total_samples 
        for label, total in total_loss_per_label.items()
    }
    avg_bias_per_label = {
        label: total / total_samples 
        for label, total in total_bias_per_label.items()
    }
    total_time = time.time() - start_time
    
    # Print combined metrics
    print(f"\n{BOLD}=== Combined Metrics (ShearNet) ==={END}")
    print(f"Mean Squared Error (MSE) from ShearNet: {BOLD}{YELLOW}{avg_loss:.6e}{END}")
    print(f"Average Bias from ShearNet: {BOLD}{YELLOW}{avg_bias:.6e}{END}")
    print(f"Time taken: {BOLD}{CYAN}{total_time:.2f} seconds{END}")
    
    # Print per-label metrics
    print("\n=== Per-Label Metrics ===")
    label_names = ['g1', 'g2', 'g1g2_combined', 'sigma', 'flux']
    for label in label_names:
        print(f"{label:>15}: MSE = {avg_loss_per_label[label]:.6e}, Bias = {avg_bias_per_label[label]:+.6e}")
    print()
    return {
        'loss': avg_loss,
        'bias': avg_bias,
        'loss_per_label': avg_loss_per_label,
        'bias_per_label': avg_bias_per_label,
        'all_preds': jnp.concatenate(all_preds) if all_preds else None
    }

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
    parser.add_argument('--seed', type=int, default=58, help='Random seed for reproducibility.')
    parser.add_argument('--load_path', type=str, default=default_save_path, help='Path to load the model parameters.')
    parser.add_argument('--nn', type=str, default='enhanced', choices=['simple', 'enhanced'], help='Neural network architecture to use (simple or enhanced).')
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
    test_images, test_labels, test_obs = generate_dataset(args.test_samples, args.psf_fwhm, exp=args.exp, return_obs=True)
    print(f"Shape of test images: {test_images.shape}")
    print(f"Shape of test labels: {test_labels.shape}")

    # Initialize the model and its parameters
    rng_key = random.PRNGKey(args.seed)
    if args.nn == "simple":
        model = SimpleGalaxyNN()
    elif args.nn == "enhanced":
        model = EnhancedGalaxyNN()
    elif args.nn == "resnet":
        model = GalaxyResNet() 
    else:
        raise ValueError("Invalid model type specified.")
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
        res_ngmix = eval_ngmix(test_obs, test_labels, seed=1234)
        _ = eval_mcal(test_images, test_labels, args.psf_fwhm)

    #if args.plot_residuals:
    if args.plot:
        predicted_labels = state.apply_fn(state.params, test_images, deterministic=True)
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
        preds_mcal = res_ngmix['preds'] if args.mcal else None
        plot_true_vs_predicted(test_labels, predicted_labels, path=scatter_path, mcal=args.mcal, preds_mcal=preds_mcal)

    if args.plot_animation:
        pass # Under development
        animation_path = os.path.join(df_plot_path, "animation_plot") if args.plot_path else None
        epochs = np.arange(1, 101)  # Assuming 100 epochs
        animate_model_epochs(test_labels, load_path, args.plot_path, epochs, state=state, model_name=args.model_name, mcal=args.mcal, preds_mcal=preds_mcal)


if __name__ == "__main__":
    main()
