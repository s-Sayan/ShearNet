"""In-loop dataset generation: render inside the jitted training step.

The up-front path (:func:`~shearnet.core.dataset_jax.generate_dataset_jax`)
materialises every stamp before training starts. This module instead fuses
rendering into the training step, so one XLA program does

    gather truth -> render -> add fresh noise -> forward -> loss -> backward

That buys three things:

* **Fresh noise every step, for free.** The noise is a PRNG draw inside the
  step, so a galaxy is never seen twice with the same realisation and there is
  nothing to memorise. No extra pass, no materialised noise array.
* **No dataset on disk or in host RAM.** 500k stamps at 53x53 float32 with a
  PSF channel is ~11 GB; here nothing but the truth table (a few tens of MB)
  is ever resident.
* **The renderer is in the autodiff graph.** Everything the response terms
  need -- ``R^gamma``, ``R^PSF``, the projected-noise filter -- is a JVP away,
  because the shear is now a traced input to the training step rather than a
  number baked into a stored array.

Compilation
-----------
The whole point of a fused step is that it compiles **once**. Three things
would silently break that, and all three are designed around here:

1. **Variable batch shape.** A short final batch is a different input shape and
   retraces. ``steps_per_epoch = n // batch_size`` and the remainder is left
   out; the permutation is reshuffled each epoch, so nothing is systematically
   dropped.
2. **PSFEx models as a fresh pytree per batch.** Stacking the batch's models
   every step re-uploads ~440 MB/batch (the basis is ``(21, 101, 101)`` per
   model) and, worse, makes the constant change every call. Instead every
   *distinct* PSF file is stacked once into a device-resident **bank** (~86 MB
   for 50 files) and the batch selects rows with a traced integer index.
3. **Truth arrays re-uploaded per batch.** The whole truth table goes to the
   device once; a step takes only an index vector.

:func:`compilation_report` measures this directly by counting Python-level
retraces, which is the version-proof way to count compilations.

Precision
---------
Rendering runs at the process-wide precision (``JAX_ENABLE_X64``; the flag is
never toggled at runtime -- see :func:`~shearnet.core.dataset_jax.render_dtype`
for why that would break jax-galsim's internal caches). The stamps are then
**cast to ``net_dtype`` (float32 by default) before the network sees them**.

That cast is what makes ``JAX_ENABLE_X64=1`` in ``setup_env.sh`` free: the
renderer gets float64 where the k-space cancellations need it, while
parameters, activations and optimiser state stay float32, so training memory
and speed are unchanged. Without it, a global x64 flag silently doubles the
cost of every layer.

For the *training* stamps themselves float32 rendering is entirely adequate --
the render error is ~1e-7 relative, against per-pixel noise many orders larger.
float64 matters for validating against GalSim and for the response terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from ..logging_utils import get_logger
from .dataset_jax import (
    PARAM_NAMES,
    JaxRenderConfig,
    Truth,
    _get_psfex,
    make_render_one,
    sample_truth,
)

logger = get_logger(__name__)

__all__ = [
    "InLoopGenerator",
    "build_generator",
    "make_batch_render",
    "make_fused_train_step",
    "make_fused_eval_step",
    "normalize_state",
    "compilation_report",
]

#: Labels derivable without an ngmix fit. ``psf_e1``/``psf_e2``/``psf_T`` need
#: adaptive moments on the rendered PSF and so are not available in-loop.
INLOOP_LABEL_KEYS = ("g1", "g2", "hlr", "flux")


# ----------------------------------------------------------------------
# host-side setup
# ----------------------------------------------------------------------
class InLoopGenerator:
    """Device-resident truth + PSFEx bank, and the batch index bookkeeping.

    Everything expensive happens once in ``__init__``; a training step then
    needs nothing from the host but a vector of indices.
    """

    def __init__(self, truth: Truth, cfg: JaxRenderConfig, batch_size: int):
        import jax
        import jax.numpy as jnp

        self.cfg = cfg
        self.batch_size = int(batch_size)
        self.n = len(truth.params["g1"])
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.n < self.batch_size:
            raise ValueError(
                f"{self.n} objects is fewer than one batch of {self.batch_size}")

        # Truth to the device once. ~12 floats/object: 500k objects is ~48 MB.
        self.params = {k: jnp.asarray(truth.params[k]) for k in PARAM_NAMES}

        # PSFEx bank: stack every DISTINCT file once, keep the per-object row
        # index. Re-stacking per batch would move the full basis for every
        # object, every step.
        self.psf_bank = None
        self.psf_idx = None
        if cfg.exp == "superbit":
            if truth.psf_files is None:
                raise ValueError("superbit truth is missing psf_files")
            import jax

            uniq = sorted(set(truth.psf_files))
            lookup = {p: i for i, p in enumerate(uniq)}
            models = [_get_psfex(p) for p in uniq]
            self.psf_bank = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs), *models)
            self.psf_idx = jnp.asarray(
                [lookup[p] for p in truth.psf_files], dtype=jnp.int32)
            logger.info("PSFEx bank: %d distinct models resident on device", len(uniq))

    @property
    def steps_per_epoch(self) -> int:
        """Full batches per epoch. The remainder is reshuffled, not dropped."""
        return self.n // self.batch_size

    def epoch_indices(self, key):
        """``(steps_per_epoch, batch_size)`` index matrix for one epoch."""
        import jax
        import jax.numpy as jnp

        perm = jax.random.permutation(key, self.n)
        keep = self.steps_per_epoch * self.batch_size
        return jnp.reshape(perm[:keep], (self.steps_per_epoch, self.batch_size))

    def batches(self, ids, key=None):
        """``(nsteps, batch_size)`` index matrix over ``ids``.

        With ``key`` the ids are permuted first (training); without, they are
        taken in order (validation, so a given stamp lands in the same batch --
        and therefore gets the same frozen noise -- every epoch). The remainder
        that does not fill a batch is left out, because a short batch is a
        different input shape and would trigger a second XLA compile.
        """
        import jax
        import jax.numpy as jnp

        ids = jnp.asarray(ids)
        if key is not None:
            ids = jax.random.permutation(key, ids)
        nsteps = ids.shape[0] // self.batch_size
        if nsteps == 0:
            raise ValueError(
                f"{ids.shape[0]} objects cannot fill a batch of {self.batch_size}")
        return jnp.reshape(ids[:nsteps * self.batch_size],
                           (nsteps, self.batch_size))

    def labels(self, output_keys: Sequence[str]):
        """Stacked label array for the requested keys, in truth order."""
        import jax.numpy as jnp

        bad = set(output_keys) - set(INLOOP_LABEL_KEYS)
        if bad:
            raise ValueError(
                f"in-loop generation cannot produce {sorted(bad)}: the psf_* "
                f"labels need an ngmix adaptive-moments fit on the rendered "
                f"PSF, which is not available inside a jitted step. Use "
                f"dataset.backend with up-front generation for those."
            )
        return jnp.stack([self.params[k] for k in output_keys], axis=1)


def normalize_state(state):
    """Make ``state.step`` a device array so the step compiles exactly once.

    ``TrainState.create`` sets ``step`` to a Python ``int``. After the first
    ``apply_gradients`` it is an array, so the second call presents a different
    abstract value for that leaf and XLA emits a **second executable** -- a full
    recompile of the fused render+forward+backward graph. It does not show up as
    a retrace (JAX reuses the jaxpr), which is why a Python-side trace counter
    alone will miss it; watch ``_cache_size()`` instead.

    Call this once before the training loop.
    """
    import jax.numpy as jnp

    try:
        return state.replace(step=jnp.asarray(state.step))
    except (AttributeError, TypeError):  # not a flax TrainState
        return state


# ----------------------------------------------------------------------
# the fused steps
# ----------------------------------------------------------------------
def _make_batch_renderer(gen: InLoopGenerator, trace_psf_shear: bool,
                         net_dtype):
    """Return ``render(idx) -> (gal, psf)`` as a traced (not yet jitted) fn."""
    import jax
    import jax.numpy as jnp

    render_one = make_render_one(gen.cfg, trace_psf_shear)
    superbit = gen.cfg.exp == "superbit"

    def render(params_all, psf_bank, psf_idx_all, idx):
        p = {k: v[idx] for k, v in params_all.items()}
        if superbit:
            rows = psf_idx_all[idx]
            # Gather the batch's models out of the resident bank. The treedef
            # never changes, so this does not retrigger compilation.
            models = jax.tree_util.tree_map(lambda a: a[rows], psf_bank)
            gal, psf = jax.vmap(render_one)(p, models)
        else:
            gal, psf = jax.vmap(lambda q: render_one(q, None))(p)
        # Cast before the network: keeps training float32 even under a global
        # JAX_ENABLE_X64=1, which is otherwise a silent 2x on every layer.
        return gal.astype(net_dtype), psf.astype(net_dtype)

    return render


def _apply_img_norm(gal, psf, img_norm, dtype):
    """Dataset-level image standardisation, applied after the float32 cast.

    Must happen inside the step: the whole point of in-loop generation is that
    the stamps never exist outside it, so there is nowhere else to normalise
    them. Mirrors ``transform_images`` on the up-front path.
    """
    import jax.numpy as jnp

    if not img_norm:
        return gal, psf
    gal = (gal - jnp.asarray(img_norm["gal_mean"], dtype)) / jnp.asarray(
        img_norm["gal_std"], dtype)
    if "psf_mean" in img_norm:
        psf = (psf - jnp.asarray(img_norm["psf_mean"], dtype)) / jnp.asarray(
            img_norm["psf_std"], dtype)
    return gal, psf


def make_batch_render(gen: InLoopGenerator, nse_sd: float = 0.0,
                      net_dtype=None, trace_psf_shear: bool = False,
                      img_norm=None):
    """Jitted ``(idx, noise_key) -> (gal, psf)``: the generation half, alone.

    This is exactly what the fused training step runs before the forward pass --
    the same closure, not a copy -- so it is the way to inspect, plot or
    validate the stamps a step actually sees. ``nse_sd=0.0`` gives the noiseless
    render.
    """
    import jax
    import jax.numpy as jnp

    if net_dtype is None:
        net_dtype = jnp.float32
    render = _make_batch_renderer(gen, trace_psf_shear, net_dtype)
    params_all, psf_bank, psf_idx_all = gen.params, gen.psf_bank, gen.psf_idx
    sd = float(nse_sd)

    def go(idx, noise_key):
        gal, psf = render(params_all, psf_bank, psf_idx_all, idx)
        if sd:
            gal = gal + jnp.asarray(sd, net_dtype) * jax.random.normal(
                noise_key, gal.shape, net_dtype)
        return _apply_img_norm(gal, psf, img_norm, net_dtype)

    return jax.jit(go)


def make_fused_train_step(
    gen: InLoopGenerator,
    forward: Callable,
    loss_fn: Callable,
    labels_all,
    nse_sd: float,
    net_dtype=None,
    trace_psf_shear: bool = False,
    donate_state: bool = False,
    img_norm=None,
):
    """Build the jitted ``render -> noise -> forward -> loss -> grad`` step.

    Args:
        gen: the :class:`InLoopGenerator` holding device-resident truth.
        forward: ``(params, gal, psf) -> preds``. Pass a closure that already
            fixes ``output_keys``/``gap``/dropout so nothing static leaks into
            the traced signature.
        loss_fn: ``(preds, labels) -> scalar``.
        labels_all: ``(N, K)`` device array of labels, in truth order.
        nse_sd: Gaussian pixel noise std, in *image* units.
        net_dtype: dtype handed to the network. Default float32.
        trace_psf_shear: apply ``.shear(psf_g1, psf_g2)`` even when zero, so the
            ``R^PSF`` tangent is live. Costs a transform in the graph.
        donate_state: donate the optimiser state buffers. Saves a copy of the
            parameters per step; only safe if the caller never reuses the old
            state object, which the loop below does not.

    Returns:
        ``step(state, idx, noise_key, dropout_key) -> (state, loss)``.
    """
    import jax
    import jax.numpy as jnp

    if net_dtype is None:
        net_dtype = jnp.float32
    # Same closure the standalone renderer exposes: fresh noise is drawn inside
    # the step, so a galaxy is never seen twice with the same realisation and
    # nothing is ever materialised.
    generate = make_batch_render(gen, nse_sd, net_dtype, trace_psf_shear, img_norm)

    def step(state, idx, noise_key, dropout_key):
        gal, psf = generate(idx, noise_key)
        lab = labels_all[idx]

        def objective(params):
            return loss_fn(forward(params, gal, psf, dropout_key), lab)

        loss, grads = jax.value_and_grad(objective)(state.params)
        return state.apply_gradients(grads=grads), loss

    return jax.jit(step, donate_argnums=(0,) if donate_state else ())


def make_fused_eval_step(
    gen: InLoopGenerator,
    forward: Callable,
    loss_fn: Callable,
    labels_all,
    nse_sd: float,
    net_dtype=None,
    trace_psf_shear: bool = False,
    img_norm=None,
    per_key: bool = False,
):
    """Evaluation twin of :func:`make_fused_train_step`.

    The noise key is derived from the *batch index*, not the step counter, so
    validation sees a frozen noise realisation across epochs. Otherwise the
    validation loss wanders with the noise draw and early stopping starts
    reacting to that instead of to the model.
    """
    import jax
    import jax.numpy as jnp

    if net_dtype is None:
        net_dtype = jnp.float32
    generate = make_batch_render(gen, nse_sd, net_dtype, trace_psf_shear, img_norm)

    def step(state, idx, noise_key):
        gal, psf = generate(idx, noise_key)
        preds = forward(state.params, gal, psf, None)
        lab = labels_all[idx]
        loss = loss_fn(preds, lab)
        if per_key:
            return loss, jnp.mean((preds - lab) ** 2, axis=0)
        return loss

    return jax.jit(step)


# ----------------------------------------------------------------------
# compilation accounting
# ----------------------------------------------------------------------
class count_traces:
    """Count how many times a traced function's Python body actually runs.

    A ``jax.jit``-ed function re-executes its Python body exactly once per
    compilation, so a plain counter in the body is an exact, version-proof
    compilation count -- unlike poking at private cache internals.

    Use as a context manager around the calls being measured::

        with count_traces() as c:
            fn = make_fused_train_step(..., _probe=c)
    """

    def __init__(self):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def hit(self, *_):
        self.n += 1


def compilation_report(gen: InLoopGenerator, forward, loss_fn, labels_all,
                       nse_sd: float, epochs: int = 3, steps: int = 4,
                       state=None, trace_psf_shear: bool = False):
    """Run a short training loop and report how many compilations occurred.

    Returns a dict with the jit **cache sizes** (the authoritative compilation
    count; both should be 1), the Python retrace counts, and the number of
    steps executed. Anything above 1 means something in the step signature
    changes between calls, which costs a full XLA compile of the fused graph
    every time it happens.

    Note the two metrics are not equivalent: a leaf that switches from a Python
    scalar to a device array produces a second *executable* without a second
    *trace*, so the cache size is the number to trust. See
    :func:`normalize_state`.
    """
    import jax
    import jax.numpy as jnp

    probe = {"train": 0, "eval": 0}

    def counting(inner, key):
        def wrapped(*a, **k):
            probe[key] += 1
            return inner(*a, **k)
        return wrapped

    train_step = make_fused_train_step(
        gen, counting(forward, "train"), loss_fn, labels_all, nse_sd,
        trace_psf_shear=trace_psf_shear)
    eval_step = make_fused_eval_step(
        gen, counting(forward, "eval"), loss_fn, labels_all, nse_sd,
        trace_psf_shear=trace_psf_shear)

    state = normalize_state(state)
    key = jax.random.PRNGKey(0)
    val_key = jax.random.PRNGKey(1)
    n_steps = 0
    for epoch in range(epochs):
        key, sub = jax.random.split(key)
        idx_mat = gen.epoch_indices(sub)
        for s in range(min(steps, gen.steps_per_epoch)):
            key, nk, dk = jax.random.split(key, 3)
            state, _ = train_step(state, idx_mat[s], nk, dk)
            n_steps += 1
        for s in range(min(2, gen.steps_per_epoch)):
            eval_step(state, idx_mat[s], jax.random.fold_in(val_key, s))

    return {
        "train_traces": probe["train"],
        "eval_traces": probe["eval"],
        "steps_run": n_steps,
        "train_cache": getattr(train_step, "_cache_size", lambda: None)(),
        "eval_cache": getattr(eval_step, "_cache_size", lambda: None)(),
        "state": state,
    }


def build_generator(
    samples: int,
    cfg: JaxRenderConfig,
    batch_size: int,
    seed: int = 42,
    nse_sd: float = 1e-5,
    **truth_kwargs,
) -> InLoopGenerator:
    """Convenience: sample the truth table and wrap it for in-loop rendering.

    ``add_noise`` is forced off -- noise is drawn inside the step, so
    materialising a noise array here would waste ``N * npix**2`` floats and
    then be ignored.
    """
    truth_kwargs.pop("add_noise", None)
    truth = sample_truth(samples, cfg, seed=seed, nse_sd=nse_sd,
                         add_noise=False, **truth_kwargs)
    return InLoopGenerator(truth, cfg, batch_size)
