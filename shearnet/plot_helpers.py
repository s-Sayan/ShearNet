import os
import matplotlib.pyplot as plt
import numpy as np
from shearnet.mcal import mcal_preds
from scipy.stats import binned_statistic
from flax.training import checkpoints, train_state

def plot_learning_curve(losses, path=None):
    """Plot loss over epochs."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.grid()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        plt.savefig(path)
    else:
        plt.show()  # Display the plot if no path is provided

def plot_residuals(images, true_labels, predicted_labels, path=None, mcal=False, psf_fwhm=1.0, combined=False):
    """
    Plot the residuals (true - predicted) for both e1 and e2.
    Optionally combine residuals for e1 and e2 into a single distribution.
    """
    # Compute residuals
    residuals_e1 = predicted_labels[:, 0] - true_labels[:, 0]  # Residual for e1
    residuals_e2 = predicted_labels[:, 1] - true_labels[:, 1]   # Residual for e2

    if mcal:
        preds_mcal = mcal_preds(images, psf_fwhm)
        residuals_e1_mcal = true_labels[:, 0] - preds_mcal[:, 0]
        residuals_e2_mcal = true_labels[:, 1] - preds_mcal[:, 1]

    if combined:
        # Combine residuals for e1 and e2
        residuals_combined = np.concatenate([residuals_e1, residuals_e2])
        if mcal:
            residuals_combined_mcal = np.concatenate([residuals_e1_mcal, residuals_e2_mcal])

        # Plot combined residuals
        plt.figure(figsize=(8, 6))
        plt.hist(residuals_combined, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Combined Residuals')
        if mcal:
            plt.hist(residuals_combined_mcal, bins=30, alpha=0.5, color='orange', edgecolor='black', label='Combined Residuals (MCAL)')
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
    plt.hist(residuals_e1, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Residuals e1')
    if mcal:
        plt.hist(residuals_e1_mcal, bins=30, alpha=0.5, color='orange', edgecolor='black', label='Residuals e1 (MCAL)')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual e1')
    plt.xlabel('Residuals for e1')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution for e1')
    plt.legend()
    if path:
        plt.savefig(path + "_e1.png")
    else:
        plt.show()

    # Plot residuals for e2
    plt.figure(figsize=(8, 6))
    plt.hist(residuals_e2, bins=30, alpha=0.7, color='green', edgecolor='black', label='Residuals e2')
    if mcal:
        plt.hist(residuals_e2_mcal, bins=30, alpha=0.5, color='purple', edgecolor='black', label='Residuals e2 (MCAL)')
    plt.axvline(0, color='red', linestyle='--', label='Zero Residual e2')
    plt.xlabel('Residuals for e2')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution for e2')
    plt.legend()
    if path:
        plt.savefig(path + "_e2.png")
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

        

    # Plot for e1
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)

    # Upper plot: True vs Predicted for both model and MCAL
    plot_binned(true_labels[:, 0], predicted_labels[:, 0], ax1, 'e1', 'shearnet e1', 'blue',  residuals=None)
    
    if mcal and preds_mcal is not None:
        plot_binned(true_labels[:, 0], preds_mcal[:, 0], ax1, 'e1', 'mcal e1', 'green', plot_true=True,residuals=None)

    # Lower plot: Residuals for both model and MCAL
    residual_model = predicted_labels[:, 0] - true_labels[:, 0]
    plot_binned(true_labels[:, 0], predicted_labels[:, 0], ax2, 'e1', 'shearnet e1', 'blue',  residuals=residual_model)
    
    if mcal and preds_mcal is not None:
        residual_mcal = preds_mcal[:, 0] - true_labels[:, 0]
        plot_binned(true_labels[:, 0], preds_mcal[:, 0], ax2, 'e1', 'mcal e1', 'green', plot_true=True,residuals=residual_mcal)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path + "_e1_scatter.png")
    else:
        plt.show()

    # Plot for e2
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)

    # Upper plot: True vs Predicted for both model and MCAL
    plot_binned(true_labels[:, 1], predicted_labels[:, 1], ax1, 'e2', 'shearnet e2', 'blue', residuals=None)
    
    if mcal and preds_mcal is not None:
        plot_binned(true_labels[:, 1], preds_mcal[:, 1], ax1, 'e2', 'mcal e2', 'green', plot_true=True, residuals=None)

    # Lower plot: Residuals for both model and MCAL
    residual_model = true_labels[:, 1] - predicted_labels[:, 1]
    plot_binned(true_labels[:, 1], predicted_labels[:, 1], ax2, 'e2', 'shearnet e2', 'blue', residuals=residual_model)
    
    if mcal and preds_mcal is not None:
        residual_mcal = true_labels[:, 1] - preds_mcal[:, 1]
        plot_binned(true_labels[:, 1], preds_mcal[:, 1], ax2, 'e2', 'mcal e2', 'green', plot_true=True, residuals=residual_mcal)

    if path:
        plt.savefig(path + "_e2_scatter.png")
    else:
        plt.show()


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
