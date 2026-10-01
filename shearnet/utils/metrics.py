"""Re-export :mod:`shearnet.metrics`, which sits above ``core`` in the dependency stack."""

from ..metrics import *  # noqa: F401,F403
from ..metrics import __dict__ as _metrics_dict

globals().update({k: v for k, v in _metrics_dict.items() if not k.startswith("__")})
