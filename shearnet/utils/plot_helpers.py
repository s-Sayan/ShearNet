"""Backward-compatibility shim.

Plotting was promoted to the top-level :mod:`shearnet.plotting` module. This
re-export keeps the old ``shearnet.utils.plot_helpers`` import path working for
existing scripts and notebooks.
"""

from ..plotting import *  # noqa: F401,F403
from ..plotting import __dict__ as _plotting_dict

# Re-export every public name (including those not covered by ``*``) so attribute
# access like ``shearnet.utils.plot_helpers.extract_psf_properties_from_obs``
# continues to resolve.
globals().update({k: v for k, v in _plotting_dict.items() if not k.startswith("__")})
