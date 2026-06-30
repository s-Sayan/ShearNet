"""ShearNet plotting: scatter."""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_residuals(
    true_labels, predicted_labels, path=None, mcal=False, preds_ngmix=None, combined=False
):
    """Plot the residuals (true - predicted) for both g1 and g2.

    Optionally combine residuals for g1 and g2 into a single distribution.
    """
    # Compute residuals
    residuals = {
        "g1": (predicted_labels[:, 0] - true_labels[:, 0], preds_ngmix[:, 0] - true_labels[:, 0]),
        "g2": (predicted_labels[:, 1] - true_labels[:, 1], preds_ngmix[:, 1] - true_labels[:, 1]),
        "sigma": (
            predicted_labels[:, 2] - true_labels[:, 2],
            preds_ngmix[:, 2] - true_labels[:, 2],
        ),
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
        ax.hist(res_nn_clipped, bins=bins, alpha=0.6, label="ShearNet", color="blue", density=True)
        ax.hist(res_ngmix_clipped, bins=bins, alpha=0.6, label="ngmix", color="green", density=True)

        ax.axvline(0, color="red", linestyle="--")

        # Add mean ± std lines (optional)
        for label, res, color in [
            ("ShearNet", res_nn_clipped, "blue"),
            ("ngmix", res_ngmix_clipped, "green"),
        ]:
            mean = np.mean(res)
            std = np.std(res)
            ax.axvline(mean, color=color, linestyle="-", linewidth=1)
            ax.axvline(mean + std, color=color, linestyle=":", linewidth=1)
            ax.axvline(mean - std, color=color, linestyle=":", linewidth=1)

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


def visualize_galaxy_samples(
    images, true_labels, predicted_labels, snr_values, num_samples=5, path=None
):
    """Visualize true and predicted labels on test images."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        axes[i, 0].imshow(images[i], cmap="gray")
        axes[i, 0].set_title(
            f"True e1: {true_labels[i, 0]:.2f}, "
            f"e2: {true_labels[i, 1]:.2f}, SNR: {snr_values[i]:.2f}"
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(images[i], cmap="gray")
        axes[i, 1].set_title(
            f"Pred e1: {predicted_labels[i, 0]:.2f}, e2: {predicted_labels[i, 1]:.2f}"
        )
        axes[i, 1].axis("off")

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
        axes[i].imshow(images[i], cmap="gray", norm=LogNorm(vmin=vmin, vmax=vmax))
        axes[i].set_title(f"Test PSF Image {i}")
        axes[i].axis("off")

    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    else:
        plt.show()


def plot_true_vs_predicted(true_labels, predicted_labels, path=None, mcal=False, preds_mcal=None):
    """Scatter true vs. predicted g1, g2, sigma and flux for the network.

    Args:
        true_labels: Ground-truth array with columns ``(g1, g2, sigma, flux)``.
        predicted_labels: Network predictions in the same column order.
        path: If given, the figure is saved here; otherwise it is shown.
        mcal: If ``True``, overlay the moment/NGmix predictions in ``preds_mcal``.
        preds_mcal: Comparison predictions, required when ``mcal=True``.
    """
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
        ("g1", g1_true, g1_nn, g1_ngmix, -1.0, 1.0),
        ("g2", g2_true, g2_nn, g2_ngmix, -1.0, 1.0),
        ("sigma", sigma_true, sigma_nn, sigma_ngmix, 0.2, 2.5),
        ("flux", flux_true, flux_nn, flux_ngmix, 1, 5.0),
    ]

    for ax, (name, true, nn, ngmix, vmin, vmax) in zip(axs.flat, quantities):
        # Plot predictions
        ax.scatter(true, nn, alpha=0.4, label="ShearNet", s=10, color="blue", marker="o")
        ax.scatter(true, ngmix, alpha=0.4, label="ngmix", s=10, color="green", marker="^")

        # Reference line
        ax.plot([vmin, vmax], [vmin, vmax], "r--", label="y = x")

        # Axes formatting
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{name} true")
        ax.set_ylabel(f"{name} predicted")
        ax.set_title(f"{name} prediction")

        # Optional: log scale for sigma or flux
        # if name in ["sigma", "flux"]:
        #    ax.set_xscale('log')
        #    ax.set_yscale('log')

        # Metrics
        rmse_nn = np.sqrt(np.mean((nn - true) ** 2))
        bias_nn = np.mean(nn - true)
        rmse_ngmix = np.sqrt(np.mean((ngmix - true) ** 2))
        bias_ngmix = np.mean(ngmix - true)

        ax.text(
            0.05,
            0.95,
            f"ShearNet  RMSE: {rmse_nn:.3e}\nShearNet  Bias: {bias_nn:.3e}\n"
            f"NGmix RMSE: {rmse_ngmix:.3e}\nNGmix Bias: {bias_ngmix:.3e}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

        ax.legend()

    plt.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path + ".png")
    else:
        plt.show()

    plt.close()
