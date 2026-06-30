"""Backward-compatibility shim.

WCS construction moved to :mod:`shearnet.core.wcs` (so ``core`` no longer
imports ``utils``). This keeps the old ``shearnet.utils.simutils`` import path
working.
"""

from ..core.wcs import create_wcs_from_params  # noqa: F401
