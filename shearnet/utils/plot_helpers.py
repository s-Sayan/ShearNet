import os
import matplotlib.pyplot as plt
import numpy as np
from ..methods.mcal import mcal_preds
from scipy.stats import binned_statistic
from flax.training import checkpoints, train_state
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

def plot_learning_curve(losses, train_loss=None, path=None):
    """Plot loss over epochs."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='Validation Loss')
    if train_loss is not None:
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend(fontsize=12)
    plt.grid()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        plt.savefig(path)
    else:
        plt.show()  # Display the plot if no path is provided

def plot_residuals(true_labels, predicted_labels, path=None, mcal=False, preds_ngmix=None, combined=False):
    """
    Plot the residuals (true - predicted) for both g1 and g2.
    Optionally combine residuals for g1 and g2 into a single distribution.
    """

    # Compute residuals
    residuals = {
        "g1": (predicted_labels[:, 0] - true_labels[:, 0], preds_ngmix[:, 0] - true_labels[:, 0]),
        "g2": (predicted_labels[:, 1] - true_labels[:, 1], preds_ngmix[:, 1] - true_labels[:, 1]),
        "sigma": (predicted_labels[:, 2] - true_labels[:, 2], preds_ngmix[:, 2] - true_labels[:, 2]),
        "flux": (predicted_labels[:, 3] - true_labels[:, 3], preds_ngmix[:, 3] - true_labels[:, 3]),
    }

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    bins = 50

    for ax, (key, (res_nn, res_ngmix)) in zip(axs.flat, residuals.items()):
        # Clip extremes to focus on the bulk distribution
        clip_min = np.percentile(np.concatenate([res_nn, res_ngmix]), 1)
        clip_max = np.percentile(np.concatenate([res_nn, res_ngmix]), 99)

        res_nn_clipped = res_nn[(res_nn >= clip_min) & (res_nn <= clip_max)]
        res_ngmix_clipped = res_ngmix[(res_ngmix >= clip_min) & (res_ngmix <= clip_max)]

        # Plot histograms
        ax.hist(res_nn_clipped, bins=bins, alpha=0.6, label="ShearNet", color='blue', density=True)
        ax.hist(res_ngmix_clipped, bins=bins, alpha=0.6, label="ngmix", color='green', density=True)
        
        ax.axvline(0, color='red', linestyle='--')

        # Add mean ± std lines (optional)
        for label, res, color in [("ShearNet", res_nn_clipped, 'blue'), ("ngmix", res_ngmix_clipped, 'green')]:
            mean = np.mean(res)
            std = np.std(res)
            ax.axvline(mean, color=color, linestyle='-', linewidth=1)
            ax.axvline(mean + std, color=color, linestyle=':', linewidth=1)
            ax.axvline(mean - std, color=color, linestyle=':', linewidth=1)

        # Labels
        ax.set_title(f"{key} residuals (pred - true)")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path + ".png")
    else:
        plt.show()

def visualize_galaxy_samples(images, true_labels, predicted_labels, snr_values, num_samples=5, path=None):
    """Visualize true and predicted labels on test images."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title(f"True e1: {true_labels[i, 0]:.2f}, e2: {true_labels[i, 1]:.2f}, SNR: {snr_values[i]:.2f}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(images[i], cmap='gray')
        axes[i, 1].set_title(f"Pred e1: {predicted_labels[i, 0]:.2f}, e2: {predicted_labels[i, 1]:.2f}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    else:
        plt.show()

def visualize_psf_samples(images, num_samples=10, path=None):
    """Visualize PSF images with log-scaled colormap (shared scaling)."""
    images = np.array(images)

    positive_vals = images[images > 0]
    vmin = positive_vals.min(initial=1e-8)
    vmax = images.max()

    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        axes[i].imshow(images[i], cmap='gray', norm=LogNorm(vmin=vmin, vmax=vmax))
        axes[i].set_title(f"Test PSF Image {i}")
        axes[i].axis('off')

    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    else:
        plt.show()

def plot_true_vs_predicted(true_labels, predicted_labels, path=None, mcal=False, preds_mcal=None):

    # True values
    g1_true = true_labels[:, 0]
    g2_true = true_labels[:, 1]
    sigma_true = true_labels[:, 2]
    flux_true = true_labels[:, 3]

    # NN predictions
    g1_nn = predicted_labels[:, 0]
    g2_nn = predicted_labels[:, 1]
    sigma_nn = predicted_labels[:, 2]
    flux_nn = predicted_labels[:, 3]

    # ngmix predictions
    g1_ngmix = preds_mcal[:, 0]
    g2_ngmix = preds_mcal[:, 1]
    sigma_ngmix = preds_mcal[:, 2]
    flux_ngmix = preds_mcal[:, 3]

    # Set up plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    quantities = [
        ("g1", g1_true, g1_nn, g1_ngmix, -1., 1.),
        ("g2", g2_true, g2_nn, g2_ngmix, -1., 1.),
        ("sigma", sigma_true, sigma_nn, sigma_ngmix, 0.2, 2.5),
        ("flux", flux_true, flux_nn, flux_ngmix, 1, 5.)
    ]

    for ax, (name, true, nn, ngmix, vmin, vmax) in zip(axs.flat, quantities):
        # Plot predictions
        ax.scatter(true, nn, alpha=0.4, label="ShearNet", s=10, color='blue', marker='o')
        ax.scatter(true, ngmix, alpha=0.4, label="ngmix", s=10, color='green', marker='^')
        
        # Reference line
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', label='y = x')
        
        # Axes formatting
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f"{name} true")
        ax.set_ylabel(f"{name} predicted")
        ax.set_title(f"{name} prediction")

        # Optional: log scale for sigma or flux
        #if name in ["sigma", "flux"]:
        #    ax.set_xscale('log')
        #    ax.set_yscale('log')

        # Metrics
        rmse_nn = np.sqrt(np.mean((nn - true)**2))
        bias_nn = np.mean(nn - true)
        rmse_ngmix = np.sqrt(np.mean((ngmix - true)**2))
        bias_ngmix = np.mean(ngmix - true)

        ax.text(0.05, 0.95, 
                f"ShearNet  RMSE: {rmse_nn:.3e}\nShearNet  Bias: {bias_nn:.3e}\n"
                f"NGmix RMSE: {rmse_ngmix:.3e}\nNGmix Bias: {bias_ngmix:.3e}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

        ax.legend()

    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path + ".png")
    else:
        plt.show()
    
    plt.close()

def plot_residuals_v1(true_labels, predicted_labels, path=None, mcal=False, preds_ngmix=None, combined=False):
    """
    Plot the residuals (true - predicted) for both g1 and g2.
    Optionally combine residuals for g1 and g2 into a single distribution.
    """
    # Compute residuals
    residuals_e1 = predicted_labels[:, 0] - true_labels[:, 0]  # Residual for e1
    residuals_e2 = predicted_labels[:, 1] - true_labels[:, 1]   # Residual for e2
    residuals_sigma = predicted_labels[:, 2] - true_labels[:, 2]  # Residual for sigma
    residuals_flux = predicted_labels[:, 3] - true_labels[:, 3]   # Residual for flux

    if mcal and preds_ngmix is not None:
        residuals_e1_ngmix = true_labels[:, 0] - preds_ngmix[:, 0]
        residuals_e2_ngmix = true_labels[:, 1] - preds_ngmix[:, 1]
        residuals_sigma_ngmix = true_labels[:, 2] - preds_ngmix[:, 2]
        residuals_flux_ngmix = true_labels[:, 3] - preds_ngmix[:, 3]

    if combined:
        # Combine residuals for e1 and e2
        residuals_combined = np.concatenate([residuals_e1, residuals_e2])
        if mcal:
            residuals_combined_ngmix = np.concatenate([residuals_e1_ngmix, residuals_e2_ngmix])

        # Plot combined residuals
        plt.figure(figsize=(8, 6))
        plt.hist(residuals_combined, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Combined Residuals')
        if mcal:
            plt.hist(residuals_combined_ngmix, bins=30, alpha=0.5, color='orange', edgecolor='black', label='Combined Residuals (NGmix)')
        plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Combined Residuals Distribution')
        plt.legend()
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path + "_combined.png")
        else:
            plt.show()
        return

    # Plot residuals for e1
    plt.figure(figsize=(8, 6))
    mask_e1 = (residuals_e1 >= -0.05) & (residuals_e1 <= 0.05)
    print(f"Number of objects excluded from g1 NN mask: {np.sum(~mask_e1)}")

    plt.hist(residuals_e1[mask_e1], bins=30, alpha=0.7, color='blue', edgecolor='black', label='Residuals e1')
    if mcal:
        mask_e1_ngmix = (residuals_e1_ngmix >= -0.15) & (residuals_e1_ngmix <= 0.15)
        print(f"Number of objects excluded from g1 ngmix mask: {np.sum(~mask_e1_ngmix)}")
        plt.hist(residuals_e1_ngmix[mask_e1_ngmix], bins=30, alpha=0.5, color='orange', edgecolor='black', label='Residuals e1 (NGmix)')        
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual g1')
    #plt.xlim(-0.5, 0.5)    
    plt.xlabel('Residuals for g1')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution for g1')
    plt.legend()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path + "_g1.png")
    else:
        plt.show()

    # Plot residuals for e2
    plt.figure(figsize=(8, 6))

    mask_e2 = (residuals_e2 >= -0.05) & (residuals_e2 <= 0.05)
    print(f"Number of objects excluded from g2 NN mask: {np.sum(~mask_e1)}")

    plt.hist(residuals_e2[mask_e2], bins=30, alpha=0.7, color='blue', edgecolor='black', label='Residuals e1')    
    if mcal:
        mask_e2_ngmix = (residuals_e2_ngmix >= -0.15) & (residuals_e1_ngmix <= 0.15)
        print(f"Number of objects excluded from g2 mask: {np.sum(~mask_e2_ngmix)}")
        plt.hist(residuals_e2_ngmix[mask_e2_ngmix], bins=30, alpha=0.5, color='purple', edgecolor='black', label='Residuals g2 (NGmix)')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual g2')
    plt.xlabel('Residuals for g2')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution for g2')
    plt.legend()
    if path:
        plt.savefig(path + "_g2.png")
    else:
        plt.show()

    # Plot residuals for sigma
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_sigma, bins=30, alpha=0.7, color='teal', edgecolor='black', label='Residuals sigma (ShearNet)')
    if mcal:
        plt.hist(residuals_sigma_ngmix, bins=30, alpha=0.5, color='cyan', edgecolor='black', label='Residuals sigma (NGmix)')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual sigma')
    plt.xlabel('Residuals for sigma')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution for sigma')
    plt.legend()
    if path:
        plt.savefig(path + "_sigma.png")
    else:
        plt.show()

    # Plot residuals for flux
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_flux, bins=30, alpha=0.7, color='brown', edgecolor='black', label='Residuals flux')
    if mcal:
        plt.hist(residuals_flux_ngmix, bins=30, alpha=0.5, color='tan', edgecolor='black', label='Residuals flux (NGmix)')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual flux')
    plt.xlabel('Residuals for flux')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution for flux')
    plt.legend()
    if path:
        plt.savefig(path + "_flux.png")
    else:
        plt.show()


def plot_true_vs_predicted_v1(true_labels, predicted_labels, path=None, mcal=False, preds_mcal=None):
    """Plot true vs predicted values for both model and MCAL with residuals and error bars."""
    
    def plot_binned(true, predicted, ax, label_true, label_pred, color_pred, plot_true=False, residuals=None):
        # Bin data
        bins = np.linspace(min(true), max(true), 20)
        bin_means, bin_edges, _ = binned_statistic(true, predicted, statistic='mean', bins=bins)
        bin_std, _, _ = binned_statistic(true, predicted, statistic='std', bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])        
        
        # Binned residuals (lower plot)
        if residuals is not None:
            residual_means, _, _ = binned_statistic(true, residuals, statistic='mean', bins=bins)
            residual_std, _, _ = binned_statistic(true, residuals, statistic='std', bins=bins)
            ax.errorbar(bin_centers, residual_means, yerr=residual_std, fmt='o', color=color_pred, label=label_pred,
                        linewidth=1.5, capsize=4, capthick=1.2)
            if plot_true:
                ax.axhline(0, color='red', linestyle='--', label='Zero Residual')
            ax.set_ylabel('Residuals')
            ax.set_xlabel(f'True {label_true}')
            return

        # Binned predictions (upper plot)
        ax.errorbar(bin_centers, bin_means, yerr=bin_std, fmt='none', ecolor=color_pred, elinewidth=1.5, capsize=4, capthick=1.2)
        ax.scatter(bin_centers, bin_means, label=label_pred, color=color_pred, s=40)
        if plot_true:
            ax.plot(true, true, 'r--', label='y=x (Perfect Prediction)')        
        ax.set_ylabel(f'Predicted {label_true}')
        ax.legend()
        ax.grid()

    # Parameters to plot
    params = [
        {'index': 0, 'name': 'e1', 'file_suffix': '_e1_scatter.png'},
        {'index': 1, 'name': 'e2', 'file_suffix': '_e2_scatter.png'},
        {'index': 2, 'name': 'sigma', 'file_suffix': '_sigma_scatter.png'},
        {'index': 3, 'name': 'flux', 'file_suffix': '_flux_scatter.png'}
    ]
    
    for param in params:
        idx = param['index']
        name = param['name']
        
        # Skip if we don't have this parameter in the predictions
        if predicted_labels.shape[1] <= idx:
            continue
            
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)
        
        # Upper plot: True vs Predicted for both model and MCAL
        plot_binned(true_labels[:, idx], predicted_labels[:, idx], ax1, name, f'ShearNet {name}', 'blue', residuals=None)
        
        if mcal and preds_mcal is not None and preds_mcal.shape[1] > idx:
            plot_binned(true_labels[:, idx], preds_mcal[:, idx], ax1, name, f'NGmix {name}', 'green', plot_true=True, residuals=None)
        
        # Lower plot: Residuals for both model and MCAL
        residual_model = predicted_labels[:, idx] - true_labels[:, idx]
        plot_binned(true_labels[:, idx], predicted_labels[:, idx], ax2, name, f'ShearNet {name}', 'blue', residuals=residual_model)
        
        if mcal and preds_mcal is not None and preds_mcal.shape[1] > idx:
            residual_mcal = preds_mcal[:, idx] - true_labels[:, idx]
            plot_binned(true_labels[:, idx], preds_mcal[:, idx], ax2, name, f'NGmix {name}', 'green', plot_true=True, residuals=residual_mcal)
        
        plt.tight_layout()
        
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path + param['file_suffix'])
        else:
            plt.show()
        
        plt.close()


def plot_true_vs_predicted_anim(true_labels, predicted_labels, path=None, mcal=False, preds_mcal=None):
    """Plot true vs predicted values for both model and MCAL with residuals and error bars."""
    
    def plot_binned(true, predicted, ax, label_true, label_pred, color_pred, plot_true=False, residuals=None):
        """Helper function to plot binned data with error bars"""
        bins = np.linspace(min(true), max(true), 20)
        bin_means, bin_edges, _ = binned_statistic(true, predicted, statistic='mean', bins=bins)
        bin_std, _, _ = binned_statistic(true, predicted, statistic='std', bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])        
        
        if residuals is not None:
            residual_means, _, _ = binned_statistic(true, residuals, statistic='mean', bins=bins)
            residual_std, _, _ = binned_statistic(true, residuals, statistic='std', bins=bins)
            ax.errorbar(bin_centers, residual_means, yerr=residual_std, fmt='o', color=color_pred, label=label_pred,
                        linewidth=1.5, capsize=4, capthick=1.2)
            if plot_true:
                ax.axhline(0, color='red', linestyle='--', label='Zero Residual')
            ax.set_ylabel('Residuals')
            ax.set_xlabel(f'True {label_true}')
            return

        ax.errorbar(bin_centers, bin_means, yerr=bin_std, fmt='none', ecolor=color_pred, elinewidth=1.5, capsize=4, capthick=1.2)
        ax.scatter(bin_centers, bin_means, label=label_pred, color=color_pred, s=40)
        if plot_true:
            ax.plot(true, true, 'r--', label='y=x (Perfect Prediction)')        
        ax.set_ylabel(f'Predicted {label_true}')
        ax.legend()
        ax.grid()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)
    
    # Initialize plot elements
    line1, = ax1.plot([], [], 'bo', label="Predicted e1")
    line2, = ax2.plot([], [], 'bo', label="Residuals e1")

    def update(frame):
        """Update the plot for each frame"""
        # Update predicted labels progressively
        current_pred_labels = predicted_labels[:frame+1]  # Take first (frame+1) predictions
        
        ax1.clear()  # Clear previous plot
        ax2.clear()  # Clear previous plot
        # Plot binned true vs predicted and residuals
        plot_binned(true_labels[:, 0], current_pred_labels[:, 0], ax1, 'e1', 'shearnet e1', 'blue', residuals=None)
        
        residual_model = current_pred_labels[:, 0] - true_labels[:, 0]
        plot_binned(true_labels[:, 0], current_pred_labels[:, 0], ax2, 'e1', 'shearnet e1', 'blue', residuals=residual_model)

        # Optionally plot MCAL predictions if provided
        if mcal and preds_mcal is not None:
            ax1.scatter(true_labels[:, 0], preds_mcal[:frame+1, 0], label="MCAL e1", color='green')
            residual_mcal = preds_mcal[:frame+1, 0] - true_labels[:, 0]
            plot_binned(true_labels[:, 0], preds_mcal[:frame+1, 0], ax2, 'e1', 'mcal e1', 'green', residuals=residual_mcal)
        
        return line1, line2

    # Set up the animation
    ani = FuncAnimation(fig, update, frames=len(predicted_labels), interval=100, blit=False)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ani.save(path + "_animation.gif", writer="imagemagick", fps=2)
    else:
        plt.show()

def animate_model_epochs(true_labels, load_path, plot_path, epochs, state, model_name="model", mcal=False, preds_mcal=None):
    """Create an animation that shows the predictions of the model over different epochs."""
    
    # Load the trained model at each epoch
    predicted_labels_epoch = []
    
    for epoch in epochs:
        state = checkpoints.restore_checkpoint(ckpt_dir=load_path, target=state, step=epoch, prefix=model_name)
        
        # Make predictions using the model at this checkpoint
        predicted_labels = state.apply_fn(state.params, true_labels)  # Adjust based on your model
        predicted_labels_epoch.append(predicted_labels)
    
    # Convert the list of predictions into a numpy array (shape: epochs x samples x labels)
    predicted_labels_epoch = np.array(predicted_labels_epoch)
    
    # Generate animation
    animation_path = os.path.join(plot_path, model_name, "animation_plot")
    plot_true_vs_predicted_anim(true_labels, predicted_labels_epoch, path=animation_path, mcal=mcal, preds_mcal=preds_mcal)

def extract_psf_properties_from_obs(observations):
    """
    Extract PSF properties from ngmix observations for systematics analysis.
    
    Parameters
    ----------
    observations : list of ngmix.Observation
        List of observation objects
        
    Returns
    -------
    psf_properties : dict
        Dictionary with 'e1', 'e2', 'T' arrays
    """
    
    psf_e1_list = []
    psf_e2_list = []
    psf_T_list = []
    
    for obs in observations:
        try:
            # First, try to get from metadata if it exists
            if hasattr(obs.psf, 'meta') and 'result' in obs.psf.meta:
                psf_result = obs.psf.meta['result']
                
                # Extract ellipticity components
                if 'e' in psf_result:
                    psf_e1_list.append(psf_result['e'][0])
                    psf_e2_list.append(psf_result['e'][1])
                elif 'g' in psf_result:
                    psf_e1_list.append(psf_result['g'][0])
                    psf_e2_list.append(psf_result['g'][1])
                else:
                    # Calculate from image
                    props = calculate_image_moments(obs.psf.image)
                    psf_e1_list.append(props['e1'])
                    psf_e2_list.append(props['e2'])
                
                # Extract PSF size
                if 'T' in psf_result:
                    psf_T_list.append(psf_result['T'])
                else:
                    props = calculate_image_moments(obs.psf.image)
                    psf_T_list.append(props['T'])
            else:
                # No metadata, calculate everything from PSF image
                props = calculate_image_moments(obs.psf.image)
                psf_e1_list.append(props['e1'])
                psf_e2_list.append(props['e2'])
                psf_T_list.append(props['T'])
                
        except Exception as e:
            print(f"Warning: Could not extract PSF properties: {e}")
            psf_e1_list.append(0.0)
            psf_e2_list.append(0.0)
            psf_T_list.append(np.nan)
    
    return {
        'e1': np.array(psf_e1_list),
        'e2': np.array(psf_e2_list),
        'T': np.array(psf_T_list)
    }

import numpy as np
import jax.numpy as jnp

def calculate_image_moments(image):
    """
    Calculate second moments and ellipticity from an image.
    
    Parameters
    ----------
    image : np.ndarray
        2D image array
        
    Returns
    -------
    dict with keys:
        - 'e1': ellipticity component 1
        - 'e2': ellipticity component 2  
        - 'T': size (trace of moment matrix)
        - 'flux': total flux
    """
    # Ensure we're working with numpy
    if hasattr(image, 'array'):  # GalSim image
        image = image.array
    image = np.asarray(image)
    
    # Create coordinate grids centered on image
    ny, nx = image.shape
    y, x = np.mgrid[0:ny, 0:nx]
    
    # Calculate centroid
    total_flux = np.sum(image)
    if total_flux <= 0:
        return {'e1': 0.0, 'e2': 0.0, 'T': np.nan, 'flux': total_flux}
    
    y_cen = np.sum(image * y) / total_flux
    x_cen = np.sum(image * x) / total_flux
    
    # Calculate second moments relative to centroid
    dy = y - y_cen
    dx = x - x_cen
    
    Ixx = np.sum(image * dx * dx) / total_flux
    Iyy = np.sum(image * dy * dy) / total_flux
    Ixy = np.sum(image * dx * dy) / total_flux
    
    # Calculate T (size)
    T = Ixx + Iyy
    
    # Calculate ellipticity components
    if T > 0:
        e1 = (Ixx - Iyy) / T
        e2 = 2.0 * Ixy / T
    else:
        e1 = 0.0
        e2 = 0.0
    
    return {
        'e1': float(e1),
        'e2': float(e2),
        'T': float(T),
        'flux': float(total_flux)
    }


def plot_psf_systematics(predicted_shears, psf_properties, 
                        response_matrix=None,  # NEW: Add response matrix
                        datasets=None, path=None, 
                        title="PSF Systematics Analysis",
                        n_bins=20, fit_method='linear'):
    """
    Plot mean shear as a function of PSF properties (DES Y3 Figure 10 style).
    
    Changes from original:
    - Uses percentile-based binning for equal galaxy counts per bin
    - Applies response correction to binned shear values
    - Prints linear fit equations with uncertainties
    
    Parameters
    ----------
    predicted_shears : np.ndarray
        Predicted shear values, shape (N, 2) for [g1, g2]
    psf_properties : dict
        Dictionary with 'e1', 'e2', 'T' for PSF properties
    response_matrix : np.ndarray, optional
        2x2 response matrix [[R11, R12], [R21, R22]]. If provided, 
        divides binned shears by response.
    datasets : list of dict, optional
        Additional datasets for comparison
    path : str, optional
        Path to save the plot
    title : str
        Plot title
    n_bins : int
        Number of bins (will have equal galaxy counts)
    fit_method : str
        Fitting method (currently only 'linear' supported)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    results : dict
        Fit parameters and statistics
    """
    
    from scipy.optimize import curve_fit
    from scipy.stats import binned_statistic
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Linear fit function
    def linear_func(x, a, b):
        return a * x + b
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Store results
    results = {
        'psf_e1': {'fits': {}, 'bins': {}},
        'psf_e2': {'fits': {}, 'bins': {}},
        'psf_T': {'fits': {}, 'bins': {}}
    }
    
    # If no additional datasets provided, create one with the input data
    if datasets is None:
        datasets = [{
            'name': 'Primary',
            'shears': predicted_shears,
            'psf_props': psf_properties,
            'color': 'blue',
            'marker': 'o',
            'response': response_matrix
        }]
    
    # Print header for fit results
    print("\n" + "="*70)
    print("PSF SYSTEMATICS - LINEAR FIT RESULTS")
    print("="*70)
    
    # ========== UPPER PANELS: PSF Ellipticity ==========
    
    # Panel (0,0): Mean shear vs PSF_1 (PSF e1)
    ax_e1 = axes[0, 0]
    
    for dataset in datasets:
        shears = dataset['shears']
        psf_props = dataset['psf_props']
        name = dataset['name']
        color = dataset.get('color', 'blue')
        marker = dataset.get('marker', 'o')
        response = dataset.get('response', response_matrix)
        
        # Get PSF e1 values
        psf_e1 = psf_props['e1']
        
        # Check if PSF properties have sufficient variation
        psf_range = psf_e1.max() - psf_e1.min()
        if psf_range < 1e-10:
            print(f"Warning: PSF e1 has insufficient variation (range={psf_range:.2e}). Skipping for {name}.")
            ax_e1.scatter(psf_e1.mean(), shears[:, 0].mean(), 
                         color=color, marker=marker, s=100, label=f'{name} (mean only)',
                         alpha=0.7)
            continue
        
        # *** CHANGE 1: Use percentile-based bins for equal galaxy counts ***
        # Instead of np.linspace, use percentiles
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(psf_e1, percentiles)
        
        # Ensure unique bin edges (in case of many identical values)
        bins = np.unique(bins)
        actual_n_bins = len(bins) - 1
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # Calculate mean and std in each bin
        mean_g1, _, _ = binned_statistic(psf_e1, shears[:, 0], 
                                         statistic='mean', bins=bins)
        std_g1, _, _ = binned_statistic(psf_e1, shears[:, 0], 
                                        statistic='std', bins=bins)
        count_g1, _, _ = binned_statistic(psf_e1, shears[:, 0], 
                                          statistic='count', bins=bins)
        
        # Standard error
        stderr_g1 = std_g1 / np.sqrt(count_g1)
        
        # *** CHANGE 2: Apply response correction if provided ***
        if response is not None:
            # R11 is the response for g1
            R11 = response[0, 0]
            print(f"\n{name}: Applying response correction (R11 = {R11:.6f}) to PSF e1 bins")
            mean_g1 = mean_g1 / R11
            stderr_g1 = stderr_g1 / R11
        
        # Remove NaN values
        valid = ~np.isnan(mean_g1)
        bin_centers_valid = bin_centers[valid]
        mean_g1_valid = mean_g1[valid]
        stderr_g1_valid = stderr_g1[valid]
        count_g1_valid = count_g1[valid]
        
        # *** VERIFICATION: Check that bins have equal counts ***
        print(f"{name} - PSF e1 bins: galaxy counts = {count_g1_valid}")
        print(f"{name} - PSF e1 bins: error bars = {stderr_g1_valid}")
        
        # Plot binned data points with error bars
        ax_e1.errorbar(bin_centers_valid, mean_g1_valid, yerr=stderr_g1_valid,
                      fmt=marker, color=color, label=name, 
                      markersize=6, capsize=3, alpha=0.7)
        
        # *** FIT TO UNBINNED DATA ***
        # Apply response correction to unbinned data for fitting
        shears_corrected = shears.copy()
        if response is not None:
            shears_corrected[:, 0] = shears[:, 0] / response[0, 0]
            shears_corrected[:, 1] = shears[:, 1] / response[1, 1]
        
        try:
            # Fit using all individual galaxies
            popt, pcov = curve_fit(linear_func, psf_e1, shears_corrected[:, 0])
            
            # Create smooth fit line across the PSF range
            x_fit = np.linspace(psf_e1.min(), psf_e1.max(), 100)
            fit_line = linear_func(x_fit, *popt)
            
            ax_e1.plot(x_fit, fit_line, '-', color=color, 
                      linewidth=2, label=f'{name} linear fit', alpha=0.8)
            
            # Store results
            slope = popt[0]
            intercept = popt[1]
            slope_err = np.sqrt(pcov[0, 0])
            intercept_err = np.sqrt(pcov[1, 1])
            
            results['psf_e1']['fits'][name] = {
                'slope': slope,
                'intercept': intercept,
                'slope_err': slope_err,
                'intercept_err': intercept_err
            }
            
            # *** CHANGE 4: Print the equation with errors ***
            print(f"\n{name} - PSF e1 (g1 vs PSF_e1):")
            print(f"  Linear fit: g1 = ({slope:.6e} ± {slope_err:.6e}) * PSF_e1 + ({intercept:.6e} ± {intercept_err:.6e})")
            print(f"  α₁ (leakage coefficient) = {slope:.6e} ± {slope_err:.6e}")
            
            # *** Add equation text to plot ***
            # Format the equation text
            eq_text = f'{name}:\n$g_1 = ({slope:.2e} \\pm {slope_err:.2e})\\,PSF_1$\n'
            eq_text += f'$\\quad\\quad + ({intercept:.2e} \\pm {intercept_err:.2e})$'
            
            # Add text box to plot
            # Position based on dataset index to avoid overlap
            y_pos = 0.95 - (len([k for k in results['psf_e1']['fits'].keys()]) - 1) * 0.25
            ax_e1.text(0.05, y_pos, eq_text,
                      transform=ax_e1.transAxes,
                      fontsize=9,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor=color, alpha=0.15, edgecolor=color))
            
        except Exception as e:
            print(f"Warning: Could not fit PSF e1 for {name}: {e}")
            pass
    
    ax_e1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_e1.set_xlabel('PSF$_1$', fontsize=12)
    ax_e1.set_ylabel('$\\langle e_1 \\rangle$', fontsize=12)
    ax_e1.legend(loc='best', fontsize=9)
    ax_e1.grid(True, alpha=0.3)
    ax_e1.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
    
    # Panel (0,1): Mean shear vs PSF_2 (PSF e2)
    ax_e2 = axes[0, 1]
    
    for dataset in datasets:
        shears = dataset['shears']
        psf_props = dataset['psf_props']
        name = dataset['name']
        color = dataset.get('color', 'blue')
        marker = dataset.get('marker', 'o')
        response = dataset.get('response', response_matrix)
        
        psf_e2 = psf_props['e2']
        
        # Check variation
        psf_range = psf_e2.max() - psf_e2.min()
        if psf_range < 1e-10:
            print(f"Warning: PSF e2 has insufficient variation (range={psf_range:.2e}). Skipping for {name}.")
            ax_e2.scatter(psf_e2.mean(), shears[:, 1].mean(),
                         color=color, marker=marker, s=100, label=f'{name} (mean only)',
                         alpha=0.7)
            continue
        
        # *** Use percentile-based binning ***
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(psf_e2, percentiles)
        bins = np.unique(bins)
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        mean_g2, _, _ = binned_statistic(psf_e2, shears[:, 1], 
                                         statistic='mean', bins=bins)
        std_g2, _, _ = binned_statistic(psf_e2, shears[:, 1], 
                                        statistic='std', bins=bins)
        count_g2, _, _ = binned_statistic(psf_e2, shears[:, 1], 
                                          statistic='count', bins=bins)
        
        stderr_g2 = std_g2 / np.sqrt(count_g2)
        
        # *** Apply response correction ***
        if response is not None:
            R22 = response[1, 1]
            print(f"\n{name}: Applying response correction (R22 = {R22:.6f}) to PSF e2 bins")
            mean_g2 = mean_g2 / R22
            stderr_g2 = stderr_g2 / R22
        
        valid = ~np.isnan(mean_g2)
        bin_centers_valid = bin_centers[valid]
        mean_g2_valid = mean_g2[valid]
        stderr_g2_valid = stderr_g2[valid]
        count_g2_valid = count_g2[valid]
        
        # Verification
        print(f"{name} - PSF e2 bins: galaxy counts = {count_g2_valid}")
        print(f"{name} - PSF e2 bins: error bars = {stderr_g2_valid}")
        
        # Plot binned points
        ax_e2.errorbar(bin_centers_valid, mean_g2_valid, yerr=stderr_g2_valid,
                      fmt=marker, color=color, label=name,
                      markersize=6, capsize=3, alpha=0.7)
        
        # *** FIT TO UNBINNED DATA with response correction ***
        shears_corrected = shears.copy()
        if response is not None:
            shears_corrected[:, 0] = shears[:, 0] / response[0, 0]
            shears_corrected[:, 1] = shears[:, 1] / response[1, 1]
        
        try:
            popt, pcov = curve_fit(linear_func, psf_e2, shears_corrected[:, 1])
            
            x_fit = np.linspace(psf_e2.min(), psf_e2.max(), 100)
            fit_line = linear_func(x_fit, *popt)
            
            ax_e2.plot(x_fit, fit_line, '-', color=color,
                      linewidth=2, label=f'{name} linear fit', alpha=0.8)
            
            slope = popt[0]
            intercept = popt[1]
            slope_err = np.sqrt(pcov[0, 0])
            intercept_err = np.sqrt(pcov[1, 1])
            
            results['psf_e2']['fits'][name] = {
                'slope': slope,
                'intercept': intercept,
                'slope_err': slope_err,
                'intercept_err': intercept_err
            }
            
            # *** Print the equation ***
            print(f"\n{name} - PSF e2 (g2 vs PSF_e2):")
            print(f"  Linear fit: g2 = ({slope:.6e} ± {slope_err:.6e}) * PSF_e2 + ({intercept:.6e} ± {intercept_err:.6e})")
            print(f"  α₂ (leakage coefficient) = {slope:.6e} ± {slope_err:.6e}")
            
            # *** Add equation text to plot ***
            eq_text = f'{name}:\n$g_2 = ({slope:.2e} \\pm {slope_err:.2e})\\,PSF_2$\n'
            eq_text += f'$\\quad\\quad + ({intercept:.2e} \\pm {intercept_err:.2e})$'
            
            y_pos = 0.95 - (len([k for k in results['psf_e2']['fits'].keys()]) - 1) * 0.25
            ax_e2.text(0.05, y_pos, eq_text,
                      transform=ax_e2.transAxes,
                      fontsize=9,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor=color, alpha=0.15, edgecolor=color))
            
        except Exception as e:
            print(f"Warning: Could not fit PSF e2 for {name}: {e}")
            pass
    
    ax_e2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_e2.set_xlabel('PSF$_2$', fontsize=12)
    ax_e2.set_ylabel('$\\langle e_2 \\rangle$', fontsize=12)
    ax_e2.legend(loc='best', fontsize=9)
    ax_e2.grid(True, alpha=0.3)
    ax_e2.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
    
    # ========== LOWER PANELS: PSF Size ==========
    
    ax_T1 = axes[1, 0]
    for dataset in datasets:
        shears = dataset['shears']
        psf_props = dataset['psf_props']
        name = dataset['name']
        color = dataset.get('color', 'blue')
        marker = dataset.get('marker', 'o')
        response = dataset.get('response', response_matrix)
        
        psf_T = psf_props['T']
        
        psf_range = psf_T.max() - psf_T.min()
        if psf_range < 1e-10:
            ax_T1.scatter(psf_T.mean(), shears[:, 0].mean(),
                         color=color, marker=marker, s=100, label=f'{name} (mean only)',
                         alpha=0.7)
            continue
        
        # *** Percentile binning ***
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(psf_T, percentiles)
        bins = np.unique(bins)
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        mean_g1, _, _ = binned_statistic(psf_T, shears[:, 0],
                                         statistic='mean', bins=bins)
        std_g1, _, _ = binned_statistic(psf_T, shears[:, 0],
                                        statistic='std', bins=bins)
        count_g1, _, _ = binned_statistic(psf_T, shears[:, 0],
                                          statistic='count', bins=bins)
        
        stderr_g1 = std_g1 / np.sqrt(count_g1)
        
        # *** Response correction ***
        if response is not None:
            R11 = response[0, 0]
            mean_g1 = mean_g1 / R11
            stderr_g1 = stderr_g1 / R11
        
        valid = ~np.isnan(mean_g1)
        bin_centers_valid = bin_centers[valid]
        mean_g1_valid = mean_g1[valid]
        stderr_g1_valid = stderr_g1[valid]
        
        ax_T1.errorbar(bin_centers_valid, mean_g1_valid, yerr=stderr_g1_valid,
                      fmt=marker, color=color, label=name,
                      markersize=6, capsize=3, alpha=0.7)
        
        # Fit to unbinned data
        shears_corrected = shears.copy()
        if response is not None:
            shears_corrected[:, 0] = shears[:, 0] / response[0, 0]
        
        try:
            popt, pcov = curve_fit(linear_func, psf_T, shears_corrected[:, 0])
            
            x_fit = np.linspace(psf_T.min(), psf_T.max(), 100)
            fit_line = linear_func(x_fit, *popt)
            
            # Plot the fit line
            ax_T1.plot(x_fit, fit_line, '-', color=color,
                      linewidth=2, label=f'{name} linear fit', alpha=0.8)
            
            slope = popt[0]
            intercept = popt[1]
            slope_err = np.sqrt(pcov[0, 0])
            intercept_err = np.sqrt(pcov[1, 1])
            
            results['psf_T']['fits'][f'{name}_g1'] = {
                'slope': slope,
                'intercept': intercept,
                'slope_err': slope_err,
                'intercept_err': intercept_err
            }
            
            print(f"\n{name} - PSF T (g1 vs T_PSF):")
            print(f"  Linear fit: g1 = ({slope:.6e} ± {slope_err:.6e}) * T_PSF + ({intercept:.6e} ± {intercept_err:.6e})")
            print(f"  β₁ (size leakage coefficient) = {slope:.6e} ± {slope_err:.6e}")
            
            # *** Add equation text to plot ***
            eq_text = f'{name}:\n$g_1 = ({slope:.2e} \\pm {slope_err:.2e})\\,T_{{PSF}}$\n'
            eq_text += f'$\\quad\\quad + ({intercept:.2e} \\pm {intercept_err:.2e})$'
            
            # Count how many fits we have so far for positioning
            num_fits = len([k for k in results['psf_T']['fits'].keys() if '_g1' in k])
            y_pos = 0.95 - (num_fits - 1) * 0.25
            ax_T1.text(0.05, y_pos, eq_text,
                      transform=ax_T1.transAxes,
                      fontsize=9,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor=color, alpha=0.15, edgecolor=color))
            
        except Exception as e:
            print(f"Warning: Could not fit PSF T (g1) for {name}: {e}")
    
    ax_T1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_T1.set_xlabel('T$_{PSF}$', fontsize=12)
    ax_T1.set_ylabel('$\\langle e_1 \\rangle$', fontsize=12)
    ax_T1.legend(loc='best', fontsize=9)
    ax_T1.grid(True, alpha=0.3)
    ax_T1.ticklabel_format(style='sci', axis='y', scilimits=(-4, -4))
    
    # Panel (1,1): Mean shear g2 vs T_PSF
    ax_T2 = axes[1, 1]
    for dataset in datasets:
        shears = dataset['shears']
        psf_props = dataset['psf_props']
        name = dataset['name']
        color = dataset.get('color', 'blue')
        marker = dataset.get('marker', 'o')
        response = dataset.get('response', response_matrix)
        
        psf_T = psf_props['T']
        
        psf_range = psf_T.max() - psf_T.min()
        if psf_range < 1e-10:
            ax_T2.scatter(psf_T.mean(), shears[:, 1].mean(),
                         color=color, marker=marker, s=100, label=f'{name} (mean only)',
                         alpha=0.7)
            continue
        
        # *** Percentile binning ***
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(psf_T, percentiles)
        bins = np.unique(bins)
        
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        mean_g2, _, _ = binned_statistic(psf_T, shears[:, 1],
                                         statistic='mean', bins=bins)
        std_g2, _, _ = binned_statistic(psf_T, shears[:, 1],
                                        statistic='std', bins=bins)
        count_g2, _, _ = binned_statistic(psf_T, shears[:, 1],
                                          statistic='count', bins=bins)
        
        stderr_g2 = std_g2 / np.sqrt(count_g2)
        
        # *** Response correction ***
        if response is not None:
            R22 = response[1, 1]
            mean_g2 = mean_g2 / R22
            stderr_g2 = stderr_g2 / R22
        
        valid = ~np.isnan(mean_g2)
        bin_centers_valid = bin_centers[valid]
        mean_g2_valid = mean_g2[valid]
        stderr_g2_valid = stderr_g2[valid]
        
        ax_T2.errorbar(bin_centers_valid, mean_g2_valid, yerr=stderr_g2_valid,
                      fmt=marker, color=color, label=name,
                      markersize=6, capsize=3, alpha=0.7)
        
        # Fit to unbinned data
        shears_corrected = shears.copy()
        if response is not None:
            shears_corrected[:, 1] = shears[:, 1] / response[1, 1]
        
        try:
            popt, pcov = curve_fit(linear_func, psf_T, shears_corrected[:, 1])
            
            x_fit = np.linspace(psf_T.min(), psf_T.max(), 100)
            fit_line = linear_func(x_fit, *popt)
            
            # Plot the fit line
            ax_T2.plot(x_fit, fit_line, '-', color=color,
                      linewidth=2, label=f'{name} linear fit', alpha=0.8)
            
            slope = popt[0]
            intercept = popt[1]
            slope_err = np.sqrt(pcov[0, 0])
            intercept_err = np.sqrt(pcov[1, 1])
            
            results['psf_T']['fits'][f'{name}_g2'] = {
                'slope': slope,
                'intercept': intercept,
                'slope_err': slope_err,
                'intercept_err': intercept_err
            }
            
            print(f"\n{name} - PSF T (g2 vs T_PSF):")
            print(f"  Linear fit: g2 = ({slope:.6e} ± {slope_err:.6e}) * T_PSF + ({intercept:.6e} ± {intercept_err:.6e})")
            print(f"  β₂ (size leakage coefficient) = {slope:.6e} ± {slope_err:.6e}")
            
            # *** Add equation text to plot ***
            eq_text = f'{name}:\n$g_2 = ({slope:.2e} \\pm {slope_err:.2e})\\,T_{{PSF}}$\n'
            eq_text += f'$\\quad\\quad + ({intercept:.2e} \\pm {intercept_err:.2e})$'
            
            # Count how many fits we have so far for positioning
            num_fits = len([k for k in results['psf_T']['fits'].keys() if '_g2' in k])
            y_pos = 0.95 - (num_fits - 1) * 0.25
            ax_T2.text(0.05, y_pos, eq_text,
                      transform=ax_T2.transAxes,
                      fontsize=9,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor=color, alpha=0.15, edgecolor=color))
            
        except Exception as e:
            print(f"Warning: Could not fit PSF T (g2) for {name}: {e}")
    
    ax_T2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_T2.set_xlabel('T$_{PSF}$', fontsize=12)
    ax_T2.set_ylabel('$\\langle e_2 \\rangle$', fontsize=12)
    ax_T2.legend(loc='best', fontsize=9)
    ax_T2.grid(True, alpha=0.3)
    ax_T2.ticklabel_format(style='sci', axis='y', scilimits=(-4, -4))
    
    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    print("\n" + "="*70 + "\n")
    
    # Save if path provided
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"PSF systematics plot saved to: {path}")
    
    return fig, results


def plot_psf_systematics_from_eval(test_obs, predicted_labels, 
                                   response_matrix=None,  # NEW parameter
                                   ngmix_preds=None, 
                                   ngmix_response=None,  # NEW parameter
                                   path=None,
                                   n_bins=20):
    """
    Convenience wrapper for plotting PSF systematics from evaluation script.
    
    This function is designed to integrate seamlessly with shearnet-eval CLI.
    
    Parameters
    ----------
    test_obs : list of ngmix.Observation
        Test observations containing PSF information
    predicted_labels : np.ndarray
        Predicted shear values from neural network, shape (N, 4) [g1, g2, sigma, flux]
    response_matrix : np.ndarray, optional
        2x2 response matrix for the neural network predictions
    ngmix_preds : np.ndarray, optional
        Predictions from NGmix for comparison, shape (N, 4)
    ngmix_response : np.ndarray, optional
        2x2 response matrix for NGmix predictions
    path : str, optional
        Path to save the plot (without extension)
    n_bins : int
        Number of bins (will have equal galaxy counts in each)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    results : dict
        Fit parameters and statistics
    """
    import numpy as np
    
    # Check if predicted_labels and test_obs have matching lengths
    if len(predicted_labels) != len(test_obs):
        print(f"Warning: Length mismatch detected!")
        print(f"  predicted_labels: {len(predicted_labels)}")
        print(f"  test_obs: {len(test_obs)}")
        print(f"  Trimming test_obs to match predicted_labels length...")
        # Assume predicted_labels was filtered and test_obs needs to match
        test_obs = test_obs[:len(predicted_labels)]
    
    # Extract PSF properties from observations (assumes this function exists)
    print("Extracting PSF properties from observations...")
    from ..utils.plot_helpers import extract_psf_properties_from_obs
    psf_properties = extract_psf_properties_from_obs(test_obs)
    
    # Check if we have valid PSF properties
    valid_psf = ~(np.isnan(psf_properties['T']) | 
                  np.isnan(psf_properties['e1']) | 
                  np.isnan(psf_properties['e2']))
    
    num_valid = np.sum(valid_psf)
    print(f"Valid PSF measurements: {num_valid}/{len(test_obs)}")
    
    if num_valid < 10:
        print(f"Warning: Only {num_valid} valid PSF measurements. Skipping PSF systematics plot.")
        return None, None
    
    # Check PSF variation
    e1_range = psf_properties['e1'][valid_psf].max() - psf_properties['e1'][valid_psf].min()
    e2_range = psf_properties['e2'][valid_psf].max() - psf_properties['e2'][valid_psf].min()
    T_range = psf_properties['T'][valid_psf].max() - psf_properties['T'][valid_psf].min()
    
    print(f"PSF property ranges:")
    print(f"  e1: {e1_range:.6f}")
    print(f"  e2: {e2_range:.6f}")
    print(f"  T:  {T_range:.6f}")
    
    if e1_range < 1e-10 and e2_range < 1e-10 and T_range < 1e-10:
        print("Warning: PSF properties have no variation. Cannot analyze systematics.")
        return None, None
    
    # Filter to valid PSF measurements
    psf_properties_valid = {
        'e1': psf_properties['e1'][valid_psf],
        'e2': psf_properties['e2'][valid_psf],
        'T': psf_properties['T'][valid_psf]
    }
    
    # Convert JAX array to numpy if needed, then slice
    predicted_labels_np = np.asarray(predicted_labels)
    predicted_shears_valid = predicted_labels_np[valid_psf, :2]  # Only g1, g2
    
    # Prepare datasets for comparison
    datasets = [
        {
            'name': 'ShearNet',
            'shears': predicted_shears_valid,
            'psf_props': psf_properties_valid,
            'color': 'blue',
            'marker': 'o',
            'response': response_matrix
        }
    ]
    
    # Add NGmix if available
    if ngmix_preds is not None:
        # Convert to numpy if needed
        ngmix_preds_np = np.asarray(ngmix_preds)
        
        # Check if ngmix_preds length matches
        if len(ngmix_preds_np) != len(test_obs):
            print(f"Warning: NGmix predictions length mismatch!")
            print(f"  ngmix_preds: {len(ngmix_preds_np)}")
            print(f"  test_obs: {len(test_obs)}")
            # Try to align - if ngmix was filtered, trim test_obs accordingly
            if len(ngmix_preds_np) < len(test_obs):
                # This shouldn't happen if we already trimmed test_obs above
                print(f"  Using first {len(ngmix_preds_np)} observations for NGmix")
        
        # Filter NGmix predictions for valid PSF as well
        ngmix_shears_valid = ngmix_preds_np[valid_psf, :2]
        
        # Remove NaN values from NGmix predictions
        ngmix_valid_mask = ~(np.isnan(ngmix_shears_valid[:, 0]) | np.isnan(ngmix_shears_valid[:, 1]))
        
        if np.sum(ngmix_valid_mask) > 10:
            print(f"NGmix valid predictions: {np.sum(ngmix_valid_mask)}/{len(ngmix_shears_valid)}")
            
            # Further filter PSF properties and predictions to only include valid NGmix data
            psf_props_ngmix_valid = {
                'e1': psf_properties_valid['e1'][ngmix_valid_mask],
                'e2': psf_properties_valid['e2'][ngmix_valid_mask],
                'T': psf_properties_valid['T'][ngmix_valid_mask]
            }
            ngmix_shears_final = ngmix_shears_valid[ngmix_valid_mask]
            
            datasets.append({
                'name': 'NGmix',
                'shears': ngmix_shears_final,
                'psf_props': psf_props_ngmix_valid,
                'color': 'green',
                'marker': '^',
                'response': ngmix_response
            })
        else:
            print(f"Warning: Only {np.sum(ngmix_valid_mask)} valid NGmix predictions. Not plotting NGmix.")
    
    # Create plot
    full_path = f"{path}.png" if path else None
    fig, results = plot_psf_systematics(
        predicted_shears=predicted_shears_valid,
        psf_properties=psf_properties_valid,
        response_matrix=response_matrix,
        datasets=datasets,
        path=full_path,
        title='PSF Systematics Analysis (Response Corrected)',
        n_bins=n_bins
    )
    
    return fig, results

def calculate_psf_leakage_coefficients(predicted_shears, psf_properties, 
                                       true_shears=None):
    """
    Calculate PSF leakage coefficients (α, β) from the systematic analysis.
    
    Following DES Y3 methodology:
    - α: PSF ellipticity leakage coefficient
    - β: PSF size leakage coefficient
    
    Parameters
    ----------
    predicted_shears : np.ndarray
        Predicted shear values, shape (N, 2) for [g1, g2]
    psf_properties : dict
        Dictionary with 'e1', 'e2', 'T' for PSF properties
    true_shears : np.ndarray, optional
        True shear values to calculate residuals, shape (N, 2)
        
    Returns
    -------
    coefficients : dict
        Dictionary containing leakage coefficients and their uncertainties
    """
    
    # Calculate residuals if true shears provided
    if true_shears is not None:
        residuals = predicted_shears - true_shears
    else:
        residuals = predicted_shears
    
    # Fit α coefficients (PSF ellipticity leakage)
    from scipy.optimize import curve_fit
    
    def linear_func(x, a, b):
        return a * x + b
    
    results = {}
    
    # α_1: g1 residual vs PSF e1
    try:
        popt1, pcov1 = curve_fit(linear_func, psf_properties['e1'], 
                                 residuals[:, 0])
        results['alpha_1'] = {
            'value': popt1[0],
            'error': np.sqrt(pcov1[0, 0]),
            'intercept': popt1[1]
        }
    except:
        results['alpha_1'] = None
    
    # α_2: g2 residual vs PSF e2
    try:
        popt2, pcov2 = curve_fit(linear_func, psf_properties['e2'],
                                 residuals[:, 1])
        results['alpha_2'] = {
            'value': popt2[0],
            'error': np.sqrt(pcov2[0, 0]),
            'intercept': popt2[1]
        }
    except:
        results['alpha_2'] = None
    
    # β coefficients (PSF size leakage)
    # β_1: g1 residual vs PSF size
    try:
        popt3, pcov3 = curve_fit(linear_func, psf_properties['T'],
                                 residuals[:, 0])
        results['beta_1'] = {
            'value': popt3[0],
            'error': np.sqrt(pcov3[0, 0]),
            'intercept': popt3[1]
        }
    except:
        results['beta_1'] = None
    
    # β_2: g2 residual vs PSF size
    try:
        popt4, pcov4 = curve_fit(linear_func, psf_properties['T'],
                                 residuals[:, 1])
        results['beta_2'] = {
            'value': popt4[0],
            'error': np.sqrt(pcov4[0, 0]),
            'intercept': popt4[1]
        }
    except:
        results['beta_2'] = None
    
    return results