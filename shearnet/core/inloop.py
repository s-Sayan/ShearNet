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
from .shear_algebra import compose_shear

logger = get_logger(__name__)

__all__ = [
    "InLoopGenerator",
    "build_generator",
    "make_batch_render",
    "make_fused_train_step",
    "make_fused_eval_step",
    "make_response_report",
    "format_response_report",
    "RESPONSE_REPORT_FIELDS",
    "normalize_state",
    "compilation_report",
    "ResponseRegularization",
    "RESPONSE_TERMS",
    "PROTECTED_PARAMS",
]

#: Labels derivable without an ngmix fit. ``psf_e1``/``psf_e2``/``psf_T`` need
#: adaptive moments on the rendered PSF and so are not available in-loop.
INLOOP_LABEL_KEYS = ("g1", "g2", "hlr", "flux")


#: Response loss components, in the order they appear in the metrics vector.
RESPONSE_TERMS = ("gamma", "psf", "shift", "complement", "orbit", "isotropy")

#: Why ``isotropy_weight`` has to exist, given a D4-equivariant architecture.
ISOTROPY_NOTE = """What isotropy_weight is, and why it is a RESIDUAL.

D4 cannot relate R11 to R22. Under D4 the spin-2 pair transforms with
independent signs -- ``shearnet.core.models`` sets ``w1(g) = (-1)**r`` and
``w2(g) = (-1)**(r + m)`` -- so every one of the eight elements acts as
``S = diag(+/-1, +/-1)``. The response transforms as ``R -> S R S^-1``, which
leaves ``R11`` and ``R22`` untouched and multiplies ``R12``/``R21`` by
``w1 w2``. Orbit-averaging therefore kills the off-diagonals and says nothing
whatsoever about ``R11`` vs ``R22``. The element that would relate them is a
45-degree rotation (spin-2 doubles the angle, so it acts as
``S = [[0, -1], [1, 0]]``), and D4 has no such element.

Measured on the untrained d4-fork-like model over ten independent batches of 64
rendered stamps: D4 equivariance of ``(g1, g2)`` holds to 2.4e-07 relative,
``R12`` and ``R21`` are consistent with zero (1.8 and 1.0 sigma), and
``R11 - R22`` is 35 sigma from zero before a single gradient step.

**But the target is not isotropic per object, so the term is a residual.**
The observed shape is ``w = (eps + gamma) / (1 + conj(gamma) eps)``, which is
*not* holomorphic in ``gamma``: the ``conj(gamma)`` gives it an anti-holomorphic
Wirtinger derivative ``dw/dconj(gamma) = -eps^2`` at ``gamma = 0``. For
``w = A dgamma + B dconj(gamma)`` the real Jacobian has

    R11 - R22 = 2 Re(B) = -2 (eps1^2 - eps2^2)
    R12 + R21 = 2 Im(B) = -4 eps1 eps2

so the analytic target's anisotropic part is exactly the spin-4 term
``-2 eps^2``, verified against ``compose_shear`` to 2.6e-07. Over a shape
catalogue its per-object RMS is 0.31, while its ENSEMBLE mean is zero because
``<eps^2> = 0`` for an isotropic shape distribution.

Penalising the anisotropy of ``R`` itself would therefore fight the correct
answer object by object -- the same mistake ``gamma_target: identity`` makes,
and for the same reason. The term is the anisotropic projection of the
*residual*::

    D = R - target
    isotropy = (D11 - D22)^2 + (D12 + D21)^2

which is zero exactly when the response is right, and is a reweighting of the
component of the ``gamma_weight`` residual that D4 leaves unconstrained. It
reuses the ``R^gamma`` matrix ``gamma_weight`` already computes, so it costs no
extra render, linearisation or forward pass.

``R^PSF`` needs no counterpart: its target is zero in all four entries and
``psf_weight`` already penalises them equally, which is strictly stronger.
"""

#: Simulator parameters whose image tangents span the physically realisable
#: directions. Anything orthogonal to their span is noise, and that complement
#: is what ``complement_weight`` penalises. ``psf_x``/``psf_y`` are appended for
#: ``exp='superbit'`` only, where moving on the focal plane changes the PSF.
#:
#: ``g1``/``g2`` are here as well as ``base_g1``/``base_g2``: the intrinsic and
#: applied shears give *different* image tangents (an infinitesimal shear of the
#: round profile pushed through the finite intrinsic shear is not an
#: infinitesimal shear of the already-sheared profile), and leaving the
#: intrinsic pair out would put the network's own supervised signal partly in
#: the penalised complement. The two pairs are strongly overlapping, so the
#: projection uses a thresholded pseudo-inverse rather than a solve.
PROTECTED_PARAMS = (
    "g1",
    "g2",
    "base_g1",
    "base_g2",
    "hlr",
    "flux",
    "dx",
    "dy",
    "psf_g1",
    "psf_g2",
)


@dataclass(frozen=True)
class ResponseRegularization:
    """Opt-in differentiable response losses for an in-loop training step.

    Every weight defaults to zero, so an unconfigured run is the plain fused
    step. Each term is a statement about the *derivative* of the estimator, which
    only exists as a training signal because the renderer is inside the graph:

    * ``gamma_weight`` pushes the applied-shear response ``R^gamma`` towards its
      target (see ``gamma_target``).
    * ``psf_weight`` pushes the PSF-shear response ``R^PSF`` to zero. Zero is
      the exactly-correct target: the PSF is known, so a perfect estimator's
      answer does not move when the PSF is sheared.
    * ``shift_weight`` pushes the translation response to zero -- where the
      object sits in the stamp must not change its shape estimate.
    * ``complement_weight`` penalises ``||J P_perp||_F^2`` via a one-sample
      Hutchinson probe: the estimator's sensitivity to image directions no
      simulator parameter can produce. That complement is pure noise, so this is
      bias-safe variance reduction -- it does not touch the response on the
      protected subspace and therefore cannot, on its own, de-bias anything.
    * ``orbit_weight`` renders the same galaxy against a rotated PSF under the
      *same* noise realisation and penalises the spread of the prediction across
      that orbit. Note this rotates the **PSF alone, with the galaxy held
      fixed**, by re-rendering from the profile; it is a simulator consistency
      constraint, not a D4 transformation of the input images, and it is not
      expressible as an architectural symmetry.
    * ``isotropy_weight`` penalises ``(R11 - R22)^2 + (R12 + R21)^2`` on
      ``R^gamma``: the one statement about the response that the D4 architecture
      structurally cannot make. See :data:`ISOTROPY_NOTE`.

    Every response term above is already a **full 2x2 matrix** penalised
    entry-by-entry with equal weight -- ``R^gamma``, ``R^PSF`` and the shift
    response all constrain both components, and both are logged separately in
    :data:`RESPONSE_REPORT_FIELDS`. Nothing here is first-component-only.
    ``isotropy_weight`` exists because equal treatment of the two components is
    not the same as *tying them together*, and the difference is where an
    ``R22``-vs-``R11`` asymmetry lives.

    Attributes:
        orbit_k: size of the PSF orbit. ``2`` uses ``{P, R_90 P}``, which
            cancels the whole linear (spin-2) PSF leakage because
            ``eps^PSF -> -eps^PSF`` under a 90-degree rotation. ``4`` uses
            ``{0, 45, 90, 135}`` degrees, which additionally cancels the spin-4
            term ``eps^2 conj(gamma)`` because ``sum exp(4 i theta) = 0`` over
            those angles. K=4 costs three extra renders and forward passes per
            step; measure the ``cos 4 phi`` residual before paying for it.
        gamma_target: ``'analytic'`` (default) targets the exact per-object
            response of the *label*, ``d eps_obs / d gamma``, obtained by
            differentiating :func:`~shearnet.core.shear_algebra.compose_shear`.
            That is ``1 - eps^2`` in complex form, not the identity -- the
            identity is only correct after averaging over an isotropic shape
            distribution. ``'identity'`` targets ``I`` and is the right choice
            only if you intend an ensemble-level constraint.
        every_n_steps: evaluate the response terms every N optimiser steps. The
            graph is compiled either way (both branches of the ``lax.cond`` are
            traced); only the runtime is saved.

    A caveat worth stating once: under noise the Bayes-optimal estimator has a
    *shrunk* response, so forcing ``R^gamma`` to its noiseless value trades
    variance for bias by construction. ``complement_weight`` is the term with no
    such trade; ``gamma_weight`` is a deliberate choice, not a free lunch.
    """

    gamma_weight: float = 0.0
    psf_weight: float = 0.0
    shift_weight: float = 0.0
    complement_weight: float = 0.0
    orbit_weight: float = 0.0
    isotropy_weight: float = 0.0
    every_n_steps: int = 1
    orbit_k: int = 2
    gamma_target: str = "analytic"

    def __post_init__(self):
        if self.every_n_steps < 1:
            raise ValueError("response every_n_steps must be at least one")
        if self.orbit_k not in (2, 4):
            raise ValueError(f"response orbit_k must be 2 or 4, got {self.orbit_k!r}")
        if self.gamma_target not in ("analytic", "identity"):
            raise ValueError(
                f"response gamma_target must be 'analytic' or 'identity', "
                f"got {self.gamma_target!r}"
            )

    #: Recognised ``training.response`` keys. ``report`` is consumed by the
    #: caller rather than this object, but it is listed so a config carrying it
    #: is not rejected.
    _CONFIG_KEYS = (
        "gamma_weight",
        "psf_weight",
        "shift_weight",
        "complement_weight",
        "orbit_weight",
        "isotropy_weight",
        "every_n_steps",
        "orbit_k",
        "gamma_target",
        "report",
    )

    @classmethod
    def from_config(cls, mapping):
        """Build from a ``training.response`` block, rejecting unknown keys.

        A misspelled weight would otherwise leave that term at zero and train
        happily -- an ablation that silently ran the control.
        """
        mapping = dict(mapping or {})
        unknown = set(mapping) - set(cls._CONFIG_KEYS)
        if unknown:
            raise ValueError(
                f"unknown training.response keys: {sorted(unknown)}; "
                f"expected a subset of {list(cls._CONFIG_KEYS)}"
            )
        mapping.pop("report", None)
        return cls(**mapping)

    @property
    def weights(self):
        """The term weights, ordered as :data:`RESPONSE_TERMS`."""
        return (
            float(self.gamma_weight),
            float(self.psf_weight),
            float(self.shift_weight),
            float(self.complement_weight),
            float(self.orbit_weight),
            float(self.isotropy_weight),
        )

    @property
    def enabled(self):
        return any(self.weights)

    @property
    def orbit_angles(self):
        """Non-identity PSF rotation angles, in degrees."""
        return tuple(180.0 * k / self.orbit_k for k in range(1, self.orbit_k))


def noise_schedule_from_config(mapping):
    """Parse a ``training.noise`` block into ``(noise_range, condition)``.

    ``noise_range`` is ``None`` for the fixed-depth default. Both bounds are
    required together: half a range is a typo, not a configuration.
    """
    mapping = dict(mapping or {})
    unknown = set(mapping) - {"min_sd", "max_sd", "condition"}
    if unknown:
        raise ValueError(f"unknown training.noise keys: {sorted(unknown)}")
    lo, hi = mapping.get("min_sd"), mapping.get("max_sd")
    if (lo is None) != (hi is None):
        raise ValueError("training.noise requires both min_sd and max_sd, or neither")
    noise_range = None if lo is None else (float(lo), float(hi))
    condition = bool(mapping.get("condition", False))
    if condition and noise_range is None:
        raise ValueError(
            "training.noise.condition expresses the inputs in units of the "
            "sampled noise, which needs training.noise.min_sd/max_sd"
        )
    return noise_range, condition


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


def _protected_names(cfg: JaxRenderConfig):
    """Protected tangent directions for this render configuration."""
    if cfg.exp == "superbit":
        # Focal-plane position is a real degree of freedom only when the PSF
        # actually varies across the field.
        return PROTECTED_PARAMS + ("psf_x", "psf_y")
    return PROTECTED_PARAMS


def _check_orbit_is_meaningful(gen: InLoopGenerator):
    """Refuse an orbit term that a circular PSF makes identically zero.

    Rotating an isotropic PSF is the identity, so with ``exp='ideal'`` and no
    PSF shear the orbit term is exactly 0 for every model and every step. It
    would look configured, cost three extra renders at K=4, and constrain
    nothing. Fail loudly instead: this is the class of thing that silently
    invalidates an ablation.
    """
    import numpy as np

    if gen.cfg.exp != "ideal":
        return  # a PSFEx model is anisotropic by nature
    sheared = (
        float(np.max(np.abs(np.asarray(gen.params["psf_g1"])))) > 0.0
        or float(np.max(np.abs(np.asarray(gen.params["psf_g2"])))) > 0.0
    )
    if not sheared:
        raise ValueError(
            "response.orbit_weight is set but the PSF is a circular Gaussian "
            "(exp='ideal' with no PSF shear), so rotating it is the identity "
            "and the orbit term is identically zero. Set dataset.apply_psf_shear "
            "or use exp='superbit'."
        )


def _project_out(basis, z, rtol=1e-6):
    """Remove the component of ``z`` inside the row space of ``basis``.

    ``basis`` is ``(B, S, D)`` and ``z`` is ``(B, D)``; returns ``(B, D)``.

    The Gram matrix is genuinely rank deficient -- the intrinsic and applied
    shear tangents nearly coincide, and a zero tangent (``psf_g1`` with the PSF
    shear untraced) contributes an exactly zero row -- so this uses a symmetric
    eigendecomposition with a relative threshold rather than ``linalg.solve``.
    A ridge would have to be tuned against the wildly different column norms of
    a flux tangent and a sub-pixel shift tangent; a thresholded pseudo-inverse
    does not.
    """
    import jax.numpy as jnp

    gram = jnp.einsum("bsd,btd->bst", basis, basis)
    evals, evecs = jnp.linalg.eigh(gram)
    tol = rtol * jnp.max(evals, axis=-1, keepdims=True)
    keep = evals > tol
    inv = jnp.where(keep, 1.0 / jnp.where(keep, evals, 1.0), 0.0)
    rhs = jnp.einsum("bsd,bd->bs", basis, z)
    coeff = jnp.einsum("bij,bj,blj,bl->bi", evecs, inv, evecs, rhs)
    return z - jnp.einsum("bsd,bs->bd", basis, coeff)


def _principal_cosine(a, b):
    """Largest principal cosine between two 2-planes, per object.

    ``a`` and ``b`` are ``(B, 2, D)`` stacks of (not necessarily orthonormal)
    spanning vectors. Returns ``(B,)``. This is the geometry behind the
    noise-amplification trade: where the applied-shear and PSF-shear tangents
    are nearly parallel, driving ``R^PSF`` to zero has to fight ``R^gamma``, and
    the variance goes somewhere.
    """
    import jax.numpy as jnp

    def _orthonormalize(m):
        gram = jnp.einsum("bsd,btd->bst", m, m)
        evals, evecs = jnp.linalg.eigh(gram)
        keep = evals > 1e-12 * jnp.max(evals, axis=-1, keepdims=True)
        inv_sqrt = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, evals, 1.0)), 0.0)
        whiten = jnp.einsum("bij,bj,blj->bil", evecs, inv_sqrt, evecs)
        return jnp.einsum("bij,bjd->bid", whiten, m)

    overlap = jnp.einsum("bsd,btd->bst", _orthonormalize(a), _orthonormalize(b))
    return jnp.linalg.svd(overlap, compute_uv=False)[:, 0]


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
    label_scale=None,
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
        response: optional differentiable response regularisation. It is
            evaluated only every ``response.every_n_steps`` optimiser steps.
        shear_indices: locations of ``g1`` and ``g2`` in the model output.
        label_scale: ``(2,)`` standard deviation the shear labels were divided
            by. Response targets are expressed in *network-output* units, so a
            normalised run needs them divided by the same scale. ``None`` means
            unnormalised labels.
        return_metrics: also return the per-component loss vector for logging:
            ``[supervised, *RESPONSE_TERMS, active]``, where ``active`` is 1.0
            on steps the response terms were evaluated and 0.0 otherwise, so the
            caller can average the terms over the steps that produced them.
        noise_range: optional ``(min_sd, max_sd)`` sampled uniformly once per
            training batch. ``None`` uses the historic fixed ``nse_sd``.
        noise_condition: express the galaxy and PSF inputs in sampled-noise
            units. This conditions existing architectures on depth without
            changing their input signatures; it requires positive noise.

    Returns:
        ``step(state, idx, noise_key, dropout_key) -> (state, loss)``, or
        ``(state, loss, metrics)`` when ``return_metrics``.
    """
    import jax
    import jax.numpy as jnp

    if net_dtype is None:
        net_dtype = jnp.float32
    response = response or ResponseRegularization()
    if response.enabled and len(shear_indices) != 2:
        raise ValueError("response regularisation requires g1 and g2 output indices")
    shear_indices = tuple(shear_indices)
    inv_label_scale = (
        jnp.ones(2, net_dtype) if label_scale is None else 1.0 / jnp.asarray(label_scale, net_dtype)
    )

    protected_names = _protected_names(gen.cfg)
    # The PSF-shear transform has to be in the graph for its tangent to exist at
    # all: without it psf_g1/psf_g2 are unused inputs and the JVP is exactly zero
    # -- a term that looks satisfied while measuring nothing.
    needs_psf_tangent = bool(response.psf_weight or response.complement_weight)
    if response.orbit_weight:
        # The orbit needs an anisotropic PSF, so the shear transform goes into
        # the graph whether or not the caller asked for it.
        _check_orbit_is_meaningful(gen)
        needs_psf_tangent = True
    render_batch = _make_render_batch(gen.cfg, trace_psf_shear or needs_psf_tangent, net_dtype)
    orbit_render_batches = (
        [
            _make_render_batch(gen.cfg, trace_psf_shear or needs_psf_tangent, net_dtype, angle)
            for angle in response.orbit_angles
        ]
        if response.orbit_weight
        else []
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

    def gamma_target(p):
        """The ``(B, 2, 2)`` target for ``R^gamma``, in network-output units.

        ``'analytic'`` differentiates the label itself: the observed shape is
        ``(eps + gamma) / (1 + conj(gamma) eps)``, whose derivative is
        ``1 - eps^2`` in complex form. Averaged over an isotropic shape
        distribution ``<eps^2> = 0`` and this reduces to the identity, which is
        exactly why ``'identity'`` is defensible as an ensemble target and wrong
        as a per-object one.

        Both are then divided by the label scale, because the network predicts
        normalised labels and its response is in those units.
        """
        scale = inv_label_scale[None, :, None]
        if response.gamma_target == "identity":
            return jnp.broadcast_to(jnp.eye(2, dtype=net_dtype), (1, 2, 2)) * scale

        def label_shear(q):
            o1, o2 = compose_shear(q["g1"], q["g2"], q["base_g1"], q["base_g2"])
            return jnp.stack([o1, o2], axis=-1)

        _, label_jvp = jax.linearize(label_shear, p)
        target = jnp.stack(
            [label_jvp(_unit_tangent(p, "base_g1")), label_jvp(_unit_tangent(p, "base_g2"))],
            axis=-1,
        )
        return target.astype(net_dtype) * scale

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
            preds = forward(params, gal, psf, dropout_key)
            supervised = loss_fn(preds, lab)
            if not response.enabled:
                idle = jnp.zeros(len(RESPONSE_TERMS) + 2, net_dtype)
                return supervised, idle.at[0].set(supervised.astype(net_dtype))

            def response_terms(_):
                def predict_from_params(q):
                    return forward(
                        params,
                        *_normalize_with_noise(*render_batch(q, models), noise, noise_sd),
                        dropout_key,
                    )

                def image_from_params(q):
                    return _normalize_noiseless(*render_batch(q, models))

                # One linearisation, reused for every direction. jax.jvp per
                # direction would re-render the whole batch once per tangent --
                # up to twelve renders a step for the complement basis alone.
                need_output_jvp = bool(
                    response.gamma_weight
                    or response.psf_weight
                    or response.shift_weight
                    or response.isotropy_weight
                )
                predict_jvp = jax.linearize(predict_from_params, p)[1] if need_output_jvp else None

                def output_jvp(name):
                    return predict_jvp(_unit_tangent(p, name))[:, shear_indices]

                def response_matrix(name1, name2):
                    """``R[a, b] = d output_a / d param_b`` for the named pair."""
                    return jnp.stack([output_jvp(name1), output_jvp(name2)], axis=-1)

                # R^gamma is shared by gamma_weight and isotropy_weight, so it
                # is built once if either is on.
                if response.gamma_weight or response.isotropy_weight:
                    gamma = response_matrix("base_g1", "base_g2")
                else:
                    gamma = None

                if response.gamma_weight:
                    gamma_loss = jnp.mean((gamma - gamma_target(p)) ** 2)
                else:
                    gamma_loss = jnp.array(0.0, net_dtype)

                # The anisotropic projection of the gamma RESIDUAL. See
                # ISOTROPY_NOTE: it must be the residual, not R itself. The
                # analytic target is anisotropic per object -- its
                # (R11 - R22, R12 + R21) part is exactly the spin-4 term
                # -2 eps^2, RMS 0.31 over a shape catalogue -- so driving R's own
                # anisotropy to zero would fight the correct answer object by
                # object, which is precisely the error `gamma_target: identity`
                # makes. Subtracting the target first leaves a term that is zero
                # exactly when the response is right.
                if response.isotropy_weight:
                    deviation = gamma - gamma_target(p)
                    isotropy_loss = jnp.mean(
                        (deviation[:, 0, 0] - deviation[:, 1, 1]) ** 2
                        + (deviation[:, 0, 1] + deviation[:, 1, 0]) ** 2
                    )
                else:
                    isotropy_loss = jnp.array(0.0, net_dtype)

                if response.psf_weight:
                    psf_loss = jnp.mean(response_matrix("psf_g1", "psf_g2") ** 2)
                else:
                    psf_loss = jnp.array(0.0, net_dtype)

                if response.shift_weight:
                    shift_loss = jnp.mean(response_matrix("dx", "dy") ** 2)
                else:
                    shift_loss = jnp.array(0.0, net_dtype)

                # A Gaussian image-space probe projected away from the physical
                # tangents: the one-sample Hutchinson estimator of
                # ||J P_perp||_F^2. Done per object, so no object's tangent
                # basis leaks into another's projection.
                if response.complement_weight:
                    _, image_jvp = jax.linearize(image_from_params, p)
                    tangents = [image_jvp(_unit_tangent(p, name)) for name in protected_names]
                    nrows = len(protected_names)
                    basis = jnp.concatenate(
                        [
                            jnp.stack([t[0] for t in tangents], axis=1).reshape(
                                gal.shape[0], nrows, -1
                            ),
                            jnp.stack([t[1] for t in tangents], axis=1).reshape(
                                psf.shape[0], nrows, -1
                            ),
                        ],
                        axis=-1,
                    )
                    zkey_gal, zkey_psf = jax.random.split(jax.random.fold_in(noise_key, 1))
                    zgal = jax.random.normal(zkey_gal, gal.shape, net_dtype)
                    zpsf = jax.random.normal(zkey_psf, psf.shape, net_dtype)
                    z = jnp.concatenate(
                        [zgal.reshape(zgal.shape[0], -1), zpsf.reshape(zpsf.shape[0], -1)], axis=-1
                    )
                    z_perp = _project_out(basis, z)
                    split = gal.shape[1] * gal.shape[2]
                    complement_jvp = jax.jvp(
                        lambda g, r: forward(params, g, r, dropout_key),
                        (gal, psf),
                        (
                            z_perp[:, :split].reshape(gal.shape),
                            z_perp[:, split:].reshape(psf.shape),
                        ),
                    )[1][:, shear_indices]
                    complement_loss = jnp.mean(complement_jvp**2)
                else:
                    complement_loss = jnp.array(0.0, net_dtype)

                if not orbit_render_batches:
                    orbit_loss = jnp.array(0.0, net_dtype)
                else:
                    # Same galaxy, same noise realisation, rotated PSF. The
                    # spread of the prediction across the orbit is what a
                    # PSF-leakage term contributes and a correct estimate does
                    # not, so penalise the variance rather than one difference:
                    # it treats every orbit member alike and generalises to K=4.
                    members = [preds[:, shear_indices]] + [
                        forward(
                            params,
                            *_normalize_with_noise(*rb(p, models), noise, noise_sd),
                            dropout_key,
                        )[:, shear_indices]
                        for rb in orbit_render_batches
                    ]
                    stacked = jnp.stack(members, axis=0)
                    orbit_loss = jnp.mean((stacked - jnp.mean(stacked, axis=0)) ** 2)

                return jnp.stack(
                    [gamma_loss, psf_loss, shift_loss, complement_loss, orbit_loss,
                     isotropy_loss]
                ).astype(net_dtype)

            active = (state.step % response.every_n_steps) == 0
            terms = jax.lax.cond(
                active,
                response_terms,
                lambda _: jnp.zeros(len(RESPONSE_TERMS), dtype=net_dtype),
                operand=None,
            )
            total = supervised + sum(w * t for w, t in zip(response.weights, terms) if w)
            metrics = jnp.concatenate(
                [
                    jnp.reshape(supervised.astype(net_dtype), (1,)),
                    terms,
                    jnp.reshape(active.astype(net_dtype), (1,)),
                ]
            )
            return total, metrics

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
# response diagnostics
# ----------------------------------------------------------------------
#: Field order of the array :func:`make_response_report` returns.
RESPONSE_REPORT_FIELDS = (
    "R_gamma_11",
    "R_gamma_12",
    "R_gamma_21",
    "R_gamma_22",
    "R_gamma_target_11",
    "R_gamma_target_22",
    "R_psf_11",
    "R_psf_12",
    "R_psf_21",
    "R_psf_22",
    "R_shift_x",
    "R_shift_y",
    "cos_gamma_psf",
    "pred_var_g1",
    "pred_var_g2",
)


def make_response_report(
    gen: InLoopGenerator,
    forward: Callable,
    nse_sd: float,
    net_dtype=None,
    img_norm=None,
    shear_indices=(0, 1),
    label_scale=None,
    noise_condition: bool = False,
):
    """Build a jitted ``(params, idx, noise_key) -> (report, per_object_cos)``.

    Validation MSE is the wrong selection criterion for a response-trained
    model: a network can lower it while its ``R^gamma`` drifts and its PSF
    leakage grows, because both are properties of the *derivative* and the loss
    only sees the value. This measures the derivatives directly.

    ``report`` is a vector of :data:`RESPONSE_REPORT_FIELDS`, batch-averaged:
    the full 2x2 ``R^gamma`` (off-diagonals included -- they are where residual
    leakage shows up), the diagonal of the analytic target for comparison, the
    full 2x2 ``R^PSF``, the translation response, the mean principal cosine
    between the applied-shear and PSF-shear image tangents, and the prediction
    variance across the batch.

    ``per_object_cos`` is the ``(B,)`` tangent alignment, returned separately so
    the caller can bin ``m`` and the additive bias by it and quantify the
    noise-amplification trade rather than assume it.

    The PSF-shear transform is always traced here regardless of the training
    configuration: measuring ``R^PSF`` requires the handle to exist, and an
    untraced PSF shear would report an exactly-zero leakage that means nothing.
    """
    import jax
    import jax.numpy as jnp

    if net_dtype is None:
        net_dtype = jnp.float32
    shear_indices = tuple(shear_indices)
    inv_label_scale = (
        jnp.ones(2, net_dtype) if label_scale is None else 1.0 / jnp.asarray(label_scale, net_dtype)
    )
    render_batch = _make_render_batch(gen.cfg, True, net_dtype)
    superbit = gen.cfg.exp == "superbit"
    sd = float(nse_sd)
    if noise_condition and sd <= 0.0:
        raise ValueError("noise conditioning requires a positive nse_sd")

    def report(params, idx, noise_key):
        p = {k: v[idx] for k, v in gen.params.items()}
        models = None
        if superbit:
            models = jax.tree_util.tree_map(lambda a: a[gen.psf_idx[idx]], gen.psf_bank)

        def _prepare(g, r, noise):
            g = g + noise
            if noise_condition:
                g, r = g / sd, r / sd
            return _apply_img_norm(g, r, img_norm, net_dtype)

        noise = jnp.asarray(sd, net_dtype) * jax.random.normal(
            noise_key, (idx.shape[0], gen.cfg.npix, gen.cfg.npix), net_dtype
        )

        def predict(q):
            return forward(params, *_prepare(*render_batch(q, models), noise), None)

        def image(q):
            return _apply_img_norm(*render_batch(q, models), img_norm, net_dtype)

        unit = lambda name: {  # noqa: E731 -- one-line tangent builder
            k: (jnp.ones_like(v) if k == name else jnp.zeros_like(v)) for k, v in p.items()
        }
        preds, predict_jvp = jax.linearize(predict, p)

        def matrix(a, b):
            return jnp.stack(
                [predict_jvp(unit(a))[:, shear_indices], predict_jvp(unit(b))[:, shear_indices]],
                axis=-1,
            )

        r_gamma = jnp.mean(matrix("base_g1", "base_g2"), axis=0)
        r_psf = jnp.mean(matrix("psf_g1", "psf_g2"), axis=0)
        # Translation response has no sign convention worth averaging (it is
        # zero in the mean for a symmetric population even when large), so
        # report the RMS over both outputs.
        r_shift = jnp.sqrt(jnp.mean(matrix("dx", "dy") ** 2, axis=(0, 1)))

        def label_shear(q):
            a, b = compose_shear(q["g1"], q["g2"], q["base_g1"], q["base_g2"])
            return jnp.stack([a, b], axis=-1)

        _, label_jvp = jax.linearize(label_shear, p)
        target = (
            jnp.stack([label_jvp(unit("base_g1")), label_jvp(unit("base_g2"))], axis=-1).astype(
                net_dtype
            )
            * inv_label_scale[None, :, None]
        )
        target = jnp.mean(target, axis=0)

        _, image_jvp = jax.linearize(image, p)

        def flat_tangent(name):
            dg, dr = image_jvp(unit(name))
            return jnp.concatenate(
                [dg.reshape(dg.shape[0], -1), dr.reshape(dr.shape[0], -1)], axis=-1
            )

        v_gamma = jnp.stack([flat_tangent("base_g1"), flat_tangent("base_g2")], axis=1)
        v_psf = jnp.stack([flat_tangent("psf_g1"), flat_tangent("psf_g2")], axis=1)
        cos = _principal_cosine(v_gamma, v_psf)

        pred_var = jnp.var(preds[:, shear_indices], axis=0)
        flat = jnp.concatenate(
            [
                r_gamma.reshape(-1),
                jnp.diagonal(target),
                r_psf.reshape(-1),
                r_shift,
                jnp.reshape(jnp.mean(cos), (1,)),
                pred_var,
            ]
        )
        return flat, cos

    return jax.jit(report)


def format_response_report(values):
    """One-line human-readable form of a :func:`make_response_report` vector."""
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    fields = dict(zip(RESPONSE_REPORT_FIELDS, v))
    return (
        "Rgamma=[[{R_gamma_11:+.4f} {R_gamma_12:+.4f}] [{R_gamma_21:+.4f} {R_gamma_22:+.4f}]] "
        "target_diag=[{R_gamma_target_11:+.4f} {R_gamma_target_22:+.4f}] "
        "Rpsf=[[{R_psf_11:+.2e} {R_psf_12:+.2e}] [{R_psf_21:+.2e} {R_psf_22:+.2e}]] "
        "Rshift=[{R_shift_x:.2e} {R_shift_y:.2e}] "
        "cos(vg,vpsf)={cos_gamma_psf:.3f} "
        "var(pred)=[{pred_var_g1:.3e} {pred_var_g2:.3e}]"
    ).format(**fields)


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
