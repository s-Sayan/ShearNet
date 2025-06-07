import os
import matplotlib.pyplot as plt
import numpy as np
from ..methods.mcal import mcal_preds
from scipy.stats import binned_statistic
from flax.training import checkpoints, train_state

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

def visualize_samples(images, true_labels, predicted_labels, num_samples=5, path=None):
    """Visualize true and predicted labels on test images."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title(f"True e1: {true_labels[i, 0]:.2f}, e2: {true_labels[i, 1]:.2f}")
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
