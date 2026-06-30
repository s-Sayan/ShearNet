"""ShearNet plotting: animation."""

import os
import matplotlib.pyplot as plt
import numpy as np
from flax.training import checkpoints
from matplotlib.animation import FuncAnimation
from scipy.stats import binned_statistic


def plot_true_vs_predicted_anim(
    true_labels, predicted_labels, path=None, mcal=False, preds_mcal=None
):
    """Plot true vs predicted values for both model and MCAL with residuals and error bars."""

    def plot_binned(
        true, predicted, ax, label_true, label_pred, color_pred, plot_true=False, residuals=None
    ):
        """Plot binned data with error bars."""
        bins = np.linspace(min(true), max(true), 20)
        bin_means, bin_edges, _ = binned_statistic(true, predicted, statistic="mean", bins=bins)
        bin_std, _, _ = binned_statistic(true, predicted, statistic="std", bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        if residuals is not None:
            residual_means, _, _ = binned_statistic(true, residuals, statistic="mean", bins=bins)
            residual_std, _, _ = binned_statistic(true, residuals, statistic="std", bins=bins)
            ax.errorbar(
                bin_centers,
                residual_means,
                yerr=residual_std,
                fmt="o",
                color=color_pred,
                label=label_pred,
                linewidth=1.5,
                capsize=4,
                capthick=1.2,
            )
            if plot_true:
                ax.axhline(0, color="red", linestyle="--", label="Zero Residual")
            ax.set_ylabel("Residuals")
            ax.set_xlabel(f"True {label_true}")
            return

        ax.errorbar(
            bin_centers,
            bin_means,
            yerr=bin_std,
            fmt="none",
            ecolor=color_pred,
            elinewidth=1.5,
            capsize=4,
            capthick=1.2,
        )
        ax.scatter(bin_centers, bin_means, label=label_pred, color=color_pred, s=40)
        if plot_true:
            ax.plot(true, true, "r--", label="y=x (Perfect Prediction)")
        ax.set_ylabel(f"Predicted {label_true}")
        ax.legend()
        ax.grid()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=(8, 6), sharex=True
    )

    # Initialize plot elements
    (line1,) = ax1.plot([], [], "bo", label="Predicted e1")
    (line2,) = ax2.plot([], [], "bo", label="Residuals e1")

    def update(frame):
        """Update the plot for each frame."""
        # Update predicted labels progressively
        current_pred_labels = predicted_labels[: frame + 1]  # Take first (frame+1) predictions

        ax1.clear()  # Clear previous plot
        ax2.clear()  # Clear previous plot
        # Plot binned true vs predicted and residuals
        plot_binned(
            true_labels[:, 0],
            current_pred_labels[:, 0],
            ax1,
            "e1",
            "shearnet e1",
            "blue",
            residuals=None,
        )

        residual_model = current_pred_labels[:, 0] - true_labels[:, 0]
        plot_binned(
            true_labels[:, 0],
            current_pred_labels[:, 0],
            ax2,
            "e1",
            "shearnet e1",
            "blue",
            residuals=residual_model,
        )

        # Optionally plot MCAL predictions if provided
        if mcal and preds_mcal is not None:
            ax1.scatter(
                true_labels[:, 0], preds_mcal[: frame + 1, 0], label="MCAL e1", color="green"
            )
            residual_mcal = preds_mcal[: frame + 1, 0] - true_labels[:, 0]
            plot_binned(
                true_labels[:, 0],
                preds_mcal[: frame + 1, 0],
                ax2,
                "e1",
                "mcal e1",
                "green",
                residuals=residual_mcal,
            )

        return line1, line2

    # Set up the animation
    ani = FuncAnimation(fig, update, frames=len(predicted_labels), interval=100, blit=False)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ani.save(path + "_animation.gif", writer="imagemagick", fps=2)
    else:
        plt.show()


def animate_model_epochs(
    true_labels,
    load_path,
    plot_path,
    epochs,
    state,
    model_name="model",
    mcal=False,
    preds_mcal=None,
):
    """Create an animation of the model predictions over different epochs."""
    # Load the trained model at each epoch
    predicted_labels_epoch = []

    for epoch in epochs:
        state = checkpoints.restore_checkpoint(
            ckpt_dir=load_path, target=state, step=epoch, prefix=model_name
        )

        # Make predictions using the model at this checkpoint
        predicted_labels = state.apply_fn(state.params, true_labels)  # Adjust based on your model
        predicted_labels_epoch.append(predicted_labels)

    # Convert the list of predictions into a numpy array (shape: epochs x samples x labels)
    predicted_labels_epoch = np.array(predicted_labels_epoch)

    # Generate animation
    animation_path = os.path.join(plot_path, model_name, "animation_plot")
    plot_true_vs_predicted_anim(
        true_labels, predicted_labels_epoch, path=animation_path, mcal=mcal, preds_mcal=preds_mcal
    )
