"""Build the model a config describes, and save / restore its parameters exactly.

The parameters are one msgpack file (``flax.serialization``) holding the
model's variables -- the averaged ones when the run used an EMA. Restoring
needs the architecture, which comes from the run's resolved config and nothing
else: :func:`model_kwargs` is the single map from config to ``build_model``,
used by training and by every reader of a finished run, so the two cannot
disagree about a setting, including ``d4_multiscale``. A tree that does
not match the checkpoint is an error, not a partial load.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

from .runs import atomic_write

__all__ = ["model_kwargs", "build_model_from_config", "init_variables", "save_params",
           "load_params", "file_sha256"]


def model_kwargs(config) -> Dict[str, Any]:
    """``build_model`` keyword arguments for the architecture ``config`` names."""
    return dict(
        nn=config.get("model.type"),
        galaxy_type=config.get("model.galaxy_branch"),
        psf_type=config.get("model.psf_branch"),
        fusion=config.get("model.fusion"),
        head=config.get("model.head"),
        dropout=config.get("model.dropout"),
        branch_features=config.get("model.branch_features"),
        d4_features=config.get("model.d4_features"),
        d4_depths_galaxy=config.get("model.d4_depths_galaxy"),
        d4_depths_psf=config.get("model.d4_depths_psf"),
        d4_multiscale=config.get("model.d4_multiscale"),
        orbit_scan=config.get("model.orbit_scan"),
        fusion_pos=config.get("model.fusion_pos"),
        design=config.get("model.design"),
        d_model=config.get("model.d_model"),
        num_heads=config.get("model.num_heads"),
        num_pool_heads=config.get("model.num_pool_heads"),
        num_self_attn_layers=config.get("model.num_self_attn_layers"),
        ffn_dim=config.get("model.ffn_dim"),
    )


def build_model_from_config(config):
    """The Flax module ``config`` describes."""
    from ..core.models import build_model

    return build_model(**model_kwargs(config))


def init_variables(model, config, seed: int = 0):
    """A variable tree with the right structure (values are irrelevant)."""
    import jax
    import jax.numpy as jnp

    from ..core.models import is_fork_model

    npix = int(config.get("simulation.stamp_size"))
    stamp = jnp.ones((npix, npix), dtype=jnp.float32)
    kwargs = dict(output_keys=tuple(config.get("model.output_keys")), gap=config.get("model.gap"))
    inputs = (stamp, stamp) if is_fork_model(config.get("model.type")) else (stamp,)
    return model.init(jax.random.PRNGKey(seed), *inputs, **kwargs)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_params(variables, path) -> str:
    """Write ``variables`` atomically; returns the file's sha256."""
    from flax import serialization

    data = serialization.to_bytes(variables)
    atomic_write(path, data)
    return hashlib.sha256(data).hexdigest()


def load_params(path, template):
    """Read parameters saved by :func:`save_params` into ``template``'s structure.

    Raises if the saved tree does not match ``template`` key for key and shape
    for shape -- an architecture mismatch must not load.
    """
    import jax
    import numpy as np
    from flax import serialization

    data = Path(path).read_bytes()
    raw = serialization.msgpack_restore(data)
    expected = serialization.to_state_dict(template)

    def _paths(tree, prefix=()):
        if isinstance(tree, dict):
            out = {}
            for key, value in tree.items():
                out.update(_paths(value, prefix + (key,)))
            return out
        return {prefix: np.shape(tree)}

    got, want = _paths(raw), _paths(expected)
    if got != want:
        missing = sorted("/".join(map(str, k)) for k in set(want) - set(got))
        extra = sorted("/".join(map(str, k)) for k in set(got) - set(want))
        shapes = sorted("/".join(map(str, k)) for k in set(got) & set(want) if got[k] != want[k])
        raise ValueError(
            f"{path} does not hold this architecture's parameters: "
            f"{len(missing)} missing (e.g. {missing[:3]}), {len(extra)} unexpected "
            f"(e.g. {extra[:3]}), {len(shapes)} with another shape (e.g. {shapes[:3]})")
    restored = serialization.from_state_dict(template, raw)
    return jax.tree_util.tree_map(np.asarray, restored)
