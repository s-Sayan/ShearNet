"""Re-export WCS construction from :mod:`shearnet.core.wcs`.

The implementation lives in ``core`` to avoid importing ``utils`` from there.
"""

from ..core.wcs import create_wcs_from_params  # noqa: F401
