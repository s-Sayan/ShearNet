"""Array-level prediction errors and aligned filtering."""
from typing import Tuple
import numpy as np
import optax
from .logging_utils import get_logger

logger = get_logger(__name__)

def remove_nan_preds_multi(
    pred1: np.ndarray, pred2: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop rows where either prediction array contains a NaN, keeping rows aligned."""
    mask = ~np.any(np.isnan(pred1) | np.isnan(pred2), axis=1)
    num_removed = np.sum(~mask)
    if num_removed > 0:
        logger.info(f"[NaN Filter] Removed {num_removed} rows with NaNs in predictions.")
    return pred1[mask], pred2[mask], labels[mask]


def remove_nan_preds(preds: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop rows with NaN predictions, returning the filtered ``(preds, labels)``."""
    mask = ~np.any(np.isnan(preds), axis=1)
    num_removed = np.sum(~mask)
    if num_removed > 0:
        logger.info(f"[NaN Filter] Removed {num_removed} rows with NaNs in predictions.")
    return preds[mask], labels[mask]


def _per_label_metrics(preds, labels, output_keys):
    _ok = list(output_keys)
    loss_per_label = {}
    bias_per_label = {}
    for k in _ok:
        idx = _ok.index(k)
        loss_per_label[k] = float(optax.l2_loss(preds[:, idx], labels[:, idx]).mean())
        bias_per_label[k] = float((preds[:, idx] - labels[:, idx]).mean())
    if "g1" in _ok and "g2" in _ok:
        g_idx = [_ok.index("g1"), _ok.index("g2")]
        loss_per_label["g1g2_combined"] = float(
            optax.l2_loss(preds[:, g_idx], labels[:, g_idx]).mean()
        )
        bias_per_label["g1g2_combined"] = float((preds[:, g_idx] - labels[:, g_idx]).mean())
    return loss_per_label, bias_per_label
