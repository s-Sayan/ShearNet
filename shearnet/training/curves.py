"""The learning curve every training run saves."""

from __future__ import annotations

import numpy as np

__all__ = ["plot_learning_curve"]


def plot_learning_curve(history, path) -> None:
    """Training and validation loss against epoch, with the saved epoch marked.

    Validation is drawn only at the epochs it was measured, so a run with
    ``eval_interval > 1`` shows the gaps instead of a line through invented
    values. Writes ``path`` and closes the figure (headless-safe).
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    epoch = history.column("epoch")
    train = history.column("train_loss")
    val = history.column("val_loss")
    measured = np.isfinite(val)

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        ax.plot(epoch, train, label="training", color="C0")
        ax.plot(epoch[measured], val[measured], "o-", label="validation", color="C1",
                markersize=3)
        best = history.best_epoch
        if best is not None:
            ax.axvline(best, color="0.5", linestyle=":", label=f"saved model (epoch {best})")
        positive = np.concatenate([train[np.isfinite(train)], val[measured]])
        if positive.size and np.all(positive > 0):
            ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)
