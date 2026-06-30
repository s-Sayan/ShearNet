"""ShearNet plotting: curves."""

import os
import matplotlib.pyplot as plt


def plot_learning_curve(losses, train_loss=None, path=None):
    """Plot loss over epochs."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, label="Validation Loss")
    if train_loss is not None:
        plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend(fontsize=12)
    plt.grid()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        plt.savefig(path)
    else:
        plt.show()
