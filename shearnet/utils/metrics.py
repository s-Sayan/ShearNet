"""Backward-compatibility shim.

Metrics/evaluation was promoted to the top-level :mod:`shearnet.metrics` module
(it sits above ``core`` in the dependency stack). This keeps the old
``shearnet.utils.metrics`` import path working.
"""

from ..metrics import *  # noqa: F401,F403
from ..metrics import __dict__ as _metrics_dict

globals().update({k: v for k, v in _metrics_dict.items() if not k.startswith("__")})
