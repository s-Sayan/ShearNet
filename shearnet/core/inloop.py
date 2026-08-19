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
    "ResponseRegularization",
]

#: Labels derivable without an ngmix fit. ``psf_e1``/``psf_e2``/``psf_T`` need
#: adaptive moments on the rendered PSF and so are not available in-loop.
INLOOP_LABEL_KEYS = ("g1", "g2", "hlr", "flux")


@dataclass(frozen=True)
class ResponseRegularization:
    """Opt-in differentiable response losses for an in-loop training step.

    ``gamma_weight`` drives the applied-shear response to identity,
    ``psf_weight`` suppresses the PSF-shear response, and
    ``complement_weight`` suppresses a Hutchinson probe outside the eight
    physical image tangents.  ``orbit_weight`` compares a simulator rerender
    using a 90-degree rotated PSF with the original render under shared noise.
    This is a simulator consistency loss, not a D4 input transformation.
    """

    gamma_weight: float = 0.0
    psf_weight: float = 0.0
    complement_weight: float = 0.0
    orbit_weight: float = 0.0
    every_n_steps: int = 1
    orbit_degrees: float = 90.0

    def __post_init__(self):
        if self.every_n_steps < 1:
            raise ValueError("response every_n_steps must be at least one")
        if self.orbit_weight and self.orbit_degrees != 90.0:
            raise ValueError("the implemented PSF orbit is K=2 and requires 90 degrees")

    @property
    def enabled(self):
        return any((self.gamma_weight, self.psf_weight, self.complement_weight, self.orbit_weight))


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
            raise ValueError(f"{self.n} objects is fewer than one batch of {self.batch_size}")

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

            uniq = sorted(set(truth.psf_files))
            lookup = {p: i for i, p in enumerate(uniq)}
            models = [_get_psfex(p) for p in uniq]
            self.psf_bank = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *models)
            self.psf_idx = jnp.asarray([lookup[p] for p in truth.psf_files], dtype=jnp.int32)
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
            raise ValueError(f"{ids.shape[0]} objects cannot fill a batch of {self.batch_size}")
        return jnp.reshape(ids[: nsteps * self.batch_size], (nsteps, self.batch_size))

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
def _make_render_batch(
    cfg: JaxRenderConfig,
    trace_psf_shear: bool,
    net_dtype,
    psf_rotation_degrees: float = 0.0,
):
    """Return ``render(selected_params, selected_psf_models)``.

    Response JVPs perturb only the selected batch parameters.
    """
    import jax

    render_one = make_render_one(
        cfg,
        trace_psf_shear,
        psf_rotation_degrees=psf_rotation_degrees,
    )
    superbit = cfg.exp == "superbit"

    def render(p, models=None):
        if superbit:
            gal, psf = jax.vmap(render_one)(p, models)
        else:
            gal, psf = jax.vmap(lambda q: render_one(q, None))(p)
        return gal.astype(net_dtype), psf.astype(net_dtype)

    return render


def _make_batch_renderer(
    gen: InLoopGenerator,
    trace_psf_shear: bool,
    net_dtype,
    psf_rotation_degrees: float = 0.0,
):
    """Return ``render(idx) -> (gal, psf)`` as a traced (not yet jitted) fn."""
    import jax

    render_batch = _make_render_batch(gen.cfg, trace_psf_shear, net_dtype, psf_rotation_degrees)
    superbit = gen.cfg.exp == "superbit"

    def render(params_all, psf_bank, psf_idx_all, idx):
        p = {k: v[idx] for k, v in params_all.items()}
        if superbit:
            rows = psf_idx_all[idx]
            # Gather the batch's models out of the resident bank. The treedef
            # never changes, so this does not retrigger compilation.
            models = jax.tree_util.tree_map(lambda a: a[rows], psf_bank)
            gal, psf = render_batch(p, models)
        else:
            gal, psf = render_batch(p)
        return gal, psf

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
    gal = (gal - jnp.asarray(img_norm["gal_mean"], dtype)) / jnp.asarray(img_norm["gal_std"], dtype)
    if "psf_mean" in img_norm:
        psf = (psf - jnp.asarray(img_norm["psf_mean"], dtype)) / jnp.asarray(
            img_norm["psf_std"], dtype
        )
    return gal, psf


def make_batch_render(
    gen: InLoopGenerator,
    nse_sd: float = 0.0,
    net_dtype=None,
    trace_psf_shear: bool = False,
    img_norm=None,
    noise_condition: bool = False,
):
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
    if noise_condition and sd <= 0.0:
        raise ValueError("noise conditioning requires a positive nse_sd")

    def go(idx, noise_key):
        gal, psf = render(params_all, psf_bank, psf_idx_all, idx)
        if sd:
            gal = gal + jnp.asarray(sd, net_dtype) * jax.random.normal(
                noise_key, gal.shape, net_dtype
            )
        if noise_condition:
            gal, psf = gal / sd, psf / sd
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
    response: Optional[ResponseRegularization] = None,
    shear_indices=(0, 1),
    gamma_target=None,
    return_metrics: bool = False,
    noise_range=None,
    noise_condition: bool = False,
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
        response: optional differentiable response regularisation.  It is
            evaluated only every ``response.every_n_steps`` optimiser steps.
        shear_indices: locations of ``g1`` and ``g2`` in the model output.
        gamma_target: desired 2x2 applied-shear response in network-output
            units.  Label normalization therefore changes its diagonal.
        return_metrics: also return the five loss components for epoch logging.
        noise_range: optional ``(min_sd, max_sd)`` sampled uniformly once per
            training batch.  ``None`` uses the historic fixed ``nse_sd``.
        noise_condition: express the galaxy and PSF inputs in sampled-noise
            units.  This conditions existing architectures on depth without
            changing their input signatures; it requires positive noise.

    Returns:
        ``step(state, idx, noise_key, dropout_key) -> (state, loss)``.
    """
    import jax
    import jax.numpy as jnp

    if net_dtype is None:
        net_dtype = jnp.float32
    response = response or ResponseRegularization()
    if response.enabled and len(shear_indices) != 2:
        raise ValueError("response regularisation requires g1 and g2 output indices")
    shear_indices = tuple(shear_indices)
    if gamma_target is None:
        gamma_target = jnp.eye(2, dtype=net_dtype)
    else:
        gamma_target = jnp.asarray(gamma_target, dtype=net_dtype)

    # Response tangents in the protected subspace.  The first four describe
    # galaxy shear/size/flux, the next two PSF shear, and the last two shifts.
    protected_names = ("base_g1", "base_g2", "hlr", "flux", "psf_g1", "psf_g2", "dx", "dy")
    needs_psf_tangent = response.psf_weight or response.complement_weight
    render_batch = _make_render_batch(gen.cfg, trace_psf_shear or needs_psf_tangent, net_dtype)
    orbit_render_batch = (
        _make_render_batch(gen.cfg, trace_psf_shear, net_dtype, response.orbit_degrees)
        if response.orbit_weight
        else None
    )
    superbit = gen.cfg.exp == "superbit"
    if noise_range is None:
        noise_low = noise_high = float(nse_sd)
    else:
        noise_low, noise_high = map(float, noise_range)
        if noise_low <= 0.0 or noise_high < noise_low:
            raise ValueError("noise_range must satisfy 0 < min_sd <= max_sd")
    if noise_condition and noise_low <= 0.0:
        raise ValueError("noise conditioning requires a positive noise range")
    noise_low, noise_high = jnp.asarray(noise_low, net_dtype), jnp.asarray(noise_high, net_dtype)

    def _selected_models(idx):
        if not superbit:
            return None
        rows = gen.psf_idx[idx]
        return jax.tree_util.tree_map(lambda a: a[rows], gen.psf_bank)

    def _unit_tangent(p, name):
        return {
            key: (jnp.ones_like(value) if key == name else jnp.zeros_like(value))
            for key, value in p.items()
        }

    def _normalize_with_noise(gal, psf, noise, noise_sd):
        gal = gal + noise
        if noise_condition:
            gal, psf = gal / noise_sd, psf / noise_sd
        return _apply_img_norm(gal, psf, img_norm, net_dtype)

    def _normalize_noiseless(gal, psf):
        return _apply_img_norm(gal, psf, img_norm, net_dtype)

    def step(state, idx, noise_key, dropout_key):
        p = {key: value[idx] for key, value in gen.params.items()}
        models = _selected_models(idx)
        noise_sd = noise_low + (noise_high - noise_low) * jax.random.uniform(
            jax.random.fold_in(noise_key, 17), (), dtype=net_dtype
        )
        noise = noise_sd * jax.random.normal(
            noise_key, (idx.shape[0], gen.cfg.npix, gen.cfg.npix), net_dtype
        )
        gal, psf = _normalize_with_noise(*render_batch(p, models), noise, noise_sd)
        lab = labels_all[idx]

        def objective(params):
            supervised = loss_fn(forward(params, gal, psf, dropout_key), lab)
            if not response.enabled:
                return supervised, jnp.array([supervised, 0.0, 0.0, 0.0, 0.0])

            def response_terms(_):
                def predict_from_params(q):
                    return forward(
                        params,
                        *_normalize_with_noise(*render_batch(q, models), noise, noise_sd),
                        dropout_key,
                    )

                def image_from_params(q):
                    return _normalize_noiseless(*render_batch(q, models))

                def output_jvp(name):
                    return jax.jvp(predict_from_params, (p,), (_unit_tangent(p, name),))[1][
                        :, shear_indices
                    ]

                if response.gamma_weight:
                    gamma = jnp.stack([output_jvp("base_g1"), output_jvp("base_g2")], axis=-1)
                    gamma_loss = jnp.mean((gamma - gamma_target[None, :, :]) ** 2)
                else:
                    gamma_loss = jnp.array(0.0, net_dtype)

                if response.psf_weight:
                    psf_response = jnp.stack([output_jvp("psf_g1"), output_jvp("psf_g2")], axis=-1)
                    psf_loss = jnp.mean(psf_response**2)
                else:
                    psf_loss = jnp.array(0.0, net_dtype)

                # Project a Gaussian image-space probe away from the eight
                # physical tangents.  This is the one-sample Hutchinson
                # estimator of ||J P_perp||_F^2, evaluated per object so no
                # object's tangent basis leaks into another's projection.
                if response.complement_weight:
                    image_tangents = [
                        jax.jvp(image_from_params, (p,), (_unit_tangent(p, name),))[1]
                        for name in protected_names
                    ]
                    dgal = jnp.stack([pair[0] for pair in image_tangents], axis=1)
                    dpsf = jnp.stack([pair[1] for pair in image_tangents], axis=1)
                    basis = jnp.concatenate(
                        [
                            dgal.reshape(dgal.shape[0], len(protected_names), -1),
                            dpsf.reshape(dpsf.shape[0], len(protected_names), -1),
                        ],
                        axis=-1,
                    )
                    zkey_gal, zkey_psf = jax.random.split(jax.random.fold_in(noise_key, 1))
                    zgal = jax.random.normal(zkey_gal, gal.shape, net_dtype)
                    zpsf = jax.random.normal(zkey_psf, psf.shape, net_dtype)
                    z = jnp.concatenate(
                        [zgal.reshape(zgal.shape[0], -1), zpsf.reshape(zpsf.shape[0], -1)], axis=-1
                    )
                    gram = jnp.einsum("bsd,btd->bst", basis, basis)
                    scale = jnp.mean(jnp.diagonal(gram, axis1=-2, axis2=-1), axis=-1)
                    gram = gram + (1e-6 * scale + 1e-12)[:, None, None] * jnp.eye(8)
                    rhs = jnp.einsum("bsd,bd->bs", basis, z)[..., None]
                    coeff = jnp.linalg.solve(gram, rhs)[..., 0]
                    z_perp = z - jnp.einsum("bsd,bs->bd", basis, coeff)
                    split = gal.shape[1] * gal.shape[2]
                    zgal_perp = z_perp[:, :split].reshape(gal.shape)
                    zpsf_perp = z_perp[:, split:].reshape(psf.shape)
                    complement_jvp = jax.jvp(
                        lambda g, r: forward(params, g, r, dropout_key),
                        (gal, psf),
                        (zgal_perp, zpsf_perp),
                    )[1][:, shear_indices]
                    complement_loss = jnp.mean(complement_jvp**2)
                else:
                    complement_loss = jnp.array(0.0, net_dtype)

                if orbit_render_batch is None:
                    orbit_loss = jnp.array(0.0, net_dtype)
                else:
                    orbit_gal, orbit_psf = _normalize_with_noise(
                        *orbit_render_batch(p, models), noise, noise_sd
                    )
                    orbit_loss = jnp.mean(
                        (
                            forward(params, gal, psf, dropout_key)[:, shear_indices]
                            - forward(params, orbit_gal, orbit_psf, dropout_key)[:, shear_indices]
                        )
                        ** 2
                    )

                return jnp.array([gamma_loss, psf_loss, complement_loss, orbit_loss])

            active = (state.step % response.every_n_steps) == 0
            terms = jax.lax.cond(
                active,
                response_terms,
                lambda _: jnp.zeros(4, dtype=net_dtype),
                operand=None,
            )
            total = supervised + (
                response.gamma_weight * terms[0]
                + response.psf_weight * terms[1]
                + response.complement_weight * terms[2]
                + response.orbit_weight * terms[3]
            )
            return total, jnp.concatenate([jnp.array([supervised]), terms])

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        if return_metrics:
            return state, loss, metrics
        return state, loss

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
    noise_condition: bool = False,
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
    generate = make_batch_render(gen, nse_sd, net_dtype, trace_psf_shear, img_norm, noise_condition)

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


def compilation_report(
    gen: InLoopGenerator,
    forward,
    loss_fn,
    labels_all,
    nse_sd: float,
    epochs: int = 3,
    steps: int = 4,
    state=None,
    trace_psf_shear: bool = False,
):
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

    probe = {"train": 0, "eval": 0}

    def counting(inner, key):
        def wrapped(*a, **k):
            probe[key] += 1
            return inner(*a, **k)

        return wrapped

    train_step = make_fused_train_step(
        gen,
        counting(forward, "train"),
        loss_fn,
        labels_all,
        nse_sd,
        trace_psf_shear=trace_psf_shear,
    )
    eval_step = make_fused_eval_step(
        gen, counting(forward, "eval"), loss_fn, labels_all, nse_sd, trace_psf_shear=trace_psf_shear
    )

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
    truth = sample_truth(samples, cfg, seed=seed, nse_sd=nse_sd, add_noise=False, **truth_kwargs)
    return InLoopGenerator(truth, cfg, batch_size)
