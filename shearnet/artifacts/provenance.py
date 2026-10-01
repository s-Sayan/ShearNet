"""Where a result came from: code, packages, machine, precision."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

#: Distributions whose version is recorded. The renderer and both fitters are
#: in here because a moving dependency changes numbers without changing ours.
PACKAGES = ("jax", "jaxlib", "flax", "optax", "numpy", "galsim", "JAX-GalSim", "ngmix",
            "astropy", "numba", "scipy")

_SOURCE = Path(__file__).resolve().parents[2]


def _version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git(*args) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(_SOURCE), *args], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> Dict[str, Any]:
    """The commit the package was run from, and whether the tree was dirty."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status)}


def jax_state() -> Dict[str, Any]:
    import jax

    return {
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
    }


def collect() -> Dict[str, Any]:
    """Everything worth knowing about the process that produced a result."""
    from .. import __version__
    from ..core import models
    from .checkpoints import file_sha256

    return {
        "shearnet_version": __version__,
        "git": git_state(),
        "models_sha256": file_sha256(models.__file__),
        "packages": {name: _version(name) for name in PACKAGES},
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "jax": jax_state(),
    }
