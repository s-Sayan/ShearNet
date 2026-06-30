"""JAX device selection helpers for ShearNet."""

import jax

from ..logging_utils import get_logger

logger = get_logger(__name__)


def get_device():
    """Return (and print) the default JAX device, i.e. the CPU or GPU in use."""
    device = jax.devices()[0]
    logger.info(f"Running on device: {device}")
    return device
