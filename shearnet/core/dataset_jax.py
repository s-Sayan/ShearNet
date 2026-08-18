"""Differentiable jax-galsim dataset generation.

A standalone alternative to :mod:`shearnet.core.dataset` that renders postage
stamps with `jax-galsim <https://github.com/GalSim-developers/JAX-GalSim>`_
instead of GalSim. Two things this buys that the GalSim path cannot:

1. **The renderer is differentiable.** ``make_render_fn`` returns a pure JAX
   function of a traced parameter pytree, so ``jax.jvp`` / ``jax.jacrev``
   give ``dI/dtheta`` for every physical parameter -- including the applied
   shear and the PSF shear. That is the primitive the response terms
   (``R^gamma``, ``R^PSF``), the projected-noise filter and the PSF-orbit
   consistency term are all built from: one operation, different tangents.
2. **Batched GPU rendering.** ``jit(vmap(...))`` over a pinned FFT grid renders
   a whole batch in one fused kernel instead of dispatching ~580 XLA ops per
   object serially.

Selected with ``dataset.backend: jax-galsim`` (default ``galsim``).

Faithfulness to the GalSim path
-------------------------------
This is meant to be a *drop-in A/B*, so every per-object random draw is
replicated bit-for-bit from :func:`~shearnet.core.dataset.sim_func`: the same
``np.random.RandomState(seed=i)`` in the same consumption order (PSF shear,
then shift, then noise), and the same ``galsim.UniformDeviate(seed*1_000_003+i)``
for the PSFEx file/position. Swap the backend and the *only* thing that changes
is the renderer. Anything else would make a backend comparison meaningless.

Requirements
------------
Needs a jax-galsim with a batchable ``DES_PSFEx`` (pytree children traced,
``file_name`` out of the tree) for ``exp="superbit"``. That is JAX-GalSim
PR #261; until it merges, pin the fork by **commit SHA**, not branch name --
a moving branch silently changes the renderer under an in-flight run::

    pip install "git+https://github.com/AdamField118/JAX-GalSim@0213d84"

Known differences from GalSim, all deliberate
---------------------------------------------
* **The FFT grid is pinned** (``jax_fft_size``). ``jit``/``vmap`` need static
  shapes, so there is no per-object grid sizing and no ``real_space`` fallback
  for compact objects. Validated at Delta-m = -2e-05 against GalSim at 256.
* **Precision follows the process.** JAX defaults to float32 (~1e-7); export
  ``JAX_ENABLE_X64=1`` before importing jax for float64. The flag is *never*
  toggled at runtime: jax-galsim caches jitted internals (e.g.
  ``Lanczos._interp_kval``) with no precision in the cache key, so flipping it
  mid-process makes a later lowering fail with a dtype verifier error. float32
  rendering is fine for training stamps -- the error is ~1e-7 relative against
  far larger pixel noise -- and float64 is what the GalSim comparison and the
  response terms need.
* **The draw method is derived from the PSF mode**, identically to the GalSim
  path -- see :func:`~shearnet.core.dataset.draw_method_for`. ``exp="ideal"``
  draws with ``"auto"`` (integrate over the pixel); ``exp="superbit"`` draws
  with ``"no_pixel"``, because a PSFEx model is fitted to already-pixelized
  stars and its profile therefore already carries the pixel response. It is not
  configurable: no single setting is correct for both modes.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..logging_utils import get_logger
from .dataset import (
    MARGIN,
    PSF_DATA_DIR,
    WCS_PARAMS,
    DatasetResult,
    _load_cosmos_cat,
    draw_method_for,
    search_psf_files,
)
from .moments import get_admoms_ngmix_fit

logger = get_logger(__name__)

__all__ = [
    "JaxRenderConfig",
    "PARAM_NAMES",
    "make_render_one",
    "make_render_fn",
    "sample_truth",
    "generate_dataset_jax",
    "make_wcs",
    "render_dtype",
]

#: Traced per-object parameters, in a fixed order. Every one of these is a
#: differentiable handle: build a tangent that is 1.0 on ``base_g1`` and 0
#: elsewhere and ``jax.jvp`` returns ``dI/dgamma1``.
PARAM_NAMES: Tuple[str, ...] = (
    "g1", "g2",            # intrinsic galaxy shape
    "hlr", "flux",         # size / flux
    "dx", "dy",            # sub-pixel offset (arcsec)
    "base_g1", "base_g2",  # APPLIED shear -- the R^gamma handle
    "psf_g1", "psf_g2",    # PSF shear      -- the R^PSF handle
    "psf_x", "psf_y",      # PSFEx focal-plane position (pixels)
)


@dataclass(frozen=True)
class JaxRenderConfig:
    """Static (compile-time) settings for the batched renderer.

    Everything here lands in ``aux_data`` or closes over the traced function, so
    changing any field triggers an XLA recompile. Per-object values belong in
    the traced parameter dict instead.
    """

    npix: int = 53
    scale: float = 0.141
    psf_fwhm: float = 0.5
    gal_type: str = "exp"          # 'exp' | 'gauss'
    exp: str = "ideal"             # 'ideal' | 'superbit'
    fft_size: int = 256
    batch_size: int = 256

    def __post_init__(self):
        if self.gal_type not in ("exp", "gauss"):
            raise ValueError("gal_type must be 'exp' or 'gauss'")
        if self.exp not in ("ideal", "superbit"):
            raise ValueError("exp must be 'ideal' or 'superbit'")
        if self.fft_size <= 0 or self.fft_size % 2:
            raise ValueError("fft_size must be a positive even integer")
        if self.npix <= 0:
            raise ValueError("npix must be positive")

    @property
    def draw_method(self) -> str:
        """GalSim ``drawImage`` method implied by ``exp``; not configurable."""
        return draw_method_for(self.exp)

    @property
    def x64(self) -> bool:
        """Whether rendering is float64 -- read from the process, not settable."""
        return render_dtype() is np.float64


# ----------------------------------------------------------------------
# precision
# ----------------------------------------------------------------------
_warned_f32 = False


def render_dtype():
    """Render dtype implied by the process-wide ``jax_enable_x64`` flag.

    Deliberately read-only. An earlier version scoped x64 with a context
    manager so generation could be float64 while training stayed float32; that
    is unsafe, because jax-galsim caches jitted internals without precision in
    the cache key, and a cached float32 interpolant reused under float64 fails
    at lowering with a dtype verifier error. Set ``JAX_ENABLE_X64=1`` in the
    environment instead, and cast to float32 before the network (the in-loop
    trainer does) so training cost is unchanged.
    """
    global _warned_f32
    import jax

    if jax.config.jax_enable_x64:
        return np.float64
    if not _warned_f32:
        _warned_f32 = True
        logger.warning(
            "Rendering in float32 (jax_enable_x64 is off): render accuracy is "
            "~1e-7 relative. Fine for training stamps; export JAX_ENABLE_X64=1 "
            "before importing jax for the GalSim comparison and response terms."
        )
    return np.float32


def make_wcs():
    """jax-galsim TanWCS mirroring :func:`shearnet.core.wcs.create_wcs_from_params`.

    Built with jax-galsim's own types so the PSFEx model stays inside one
    backend; the numbers are identical to the GalSim WCS.
    """
    import jax_galsim as jgs

    p = WCS_PARAMS
    theta = p.get("theta", 0.0) * jgs.degrees
    ps = p["pixel_scale"]
    fiducial = jgs.ImageF(p["image_xsize"], p["image_ysize"])
    affine = jgs.AffineTransform(
        np.cos(theta.rad) * ps, -np.sin(theta.rad) * ps,
        np.sin(theta.rad) * ps, np.cos(theta.rad) * ps,
        origin=fiducial.true_center,
    )
    sky = jgs.CelestialCoord(
        ra=p["center_ra"] * jgs.hours, dec=p["center_dec"] * jgs.degrees
    )
    return jgs.TanWCS(affine, sky, units=jgs.arcsec)


_psfex_cache: dict = {}


def _get_psfex(path):
    """Load (and cache) a jax-galsim ``DES_PSFEx``; cheap to re-stack, dear to read.

    Keyed on ``(path, x64)``, not on the path alone. The model's arrays take
    their dtype from ``jax_enable_x64`` at construction time, and this flag is
    toggled per call by :func:`render_truth`, so a path-only cache hands a
    float32 model to a float64 render (or the reverse) as soon as both paths run
    in one process -- which shows up as an order-dependent test failure rather
    than anything local to the caller.
    """
    import jax
    import jax_galsim as jgs

    key = (path, bool(jax.config.jax_enable_x64))
    if key not in _psfex_cache:
        _psfex_cache[key] = jgs.des.DES_PSFEx(path, wcs=make_wcs())
    return _psfex_cache[key]


def _stack_psfex(paths):
    """Stack per-object PSFEx models into ONE batched pytree for ``vmap``.

    Legal only because the PSFEx arrays are pytree *children* (traced) and just
    ``fit_order``/``fit_size`` are static aux_data, so models read from
    different files share a tree structure -- JAX-GalSim PR #261. Without it
    the batched path has to pre-render each PSF to a stamp and wrap it in an
    ``InterpolatedImage``, which costs a second interpolation kernel and a wing
    truncation, and biases the render by Delta-m = -2.4e-03.
    """
    import jax
    import jax.numpy as jnp

    models = [_get_psfex(p) for p in paths]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *models)


# ----------------------------------------------------------------------
# the differentiable render
# ----------------------------------------------------------------------
def make_render_one(cfg: JaxRenderConfig, trace_psf_shear: bool = False):
    """Return the single-object render ``(params, psf_model) -> (gal, psf)``.

    The one place a stamp is defined. :func:`make_render_fn` vmaps it over a
    stacked PSFEx pytree; the in-loop trainer vmaps the same function over
    models gathered from a device-resident bank. Sharing this body is what
    guarantees the up-front and in-loop paths render identically -- two copies
    would drift.
    """
    import jax_galsim as jgs

    npix, scale = cfg.npix, cfg.scale
    gsp = jgs.GSParams(minimum_fft_size=cfg.fft_size, maximum_fft_size=cfg.fft_size)
    profile = jgs.Exponential if cfg.gal_type == "exp" else jgs.Gaussian
    superbit = cfg.exp == "superbit"
    method = cfg.draw_method
    # drawImage builds an ImageF (float32) unless told otherwise -- the
    # jax_enable_x64 flag does NOT change that default. Without an explicit
    # dtype the render silently runs at float32 even with x64 enabled.
    dtype = render_dtype()

    def render_one(p, psf_model):
        # Order matches sim_func exactly: intrinsic shear, shift, applied shear.
        gal = profile(half_light_radius=p["hlr"], flux=p["flux"])
        gal = gal.shear(g1=p["g1"], g2=p["g2"])
        gal = gal.shift(p["dx"], p["dy"])
        gal = gal.shear(g1=p["base_g1"], g2=p["base_g2"])

        if superbit:
            psf = psf_model.getPSF(
                jgs.PositionD(p["psf_x"], p["psf_y"]), gsparams=gsp
            )
        else:
            psf = jgs.Gaussian(fwhm=cfg.psf_fwhm, gsparams=gsp)
        if trace_psf_shear:
            psf = psf.shear(g1=p["psf_g1"], g2=p["psf_g2"])

        obj = jgs.Convolve(gal, psf, gsparams=gsp)
        gal_im = obj.drawImage(
            nx=npix, ny=npix, scale=scale, method=method, dtype=dtype).array
        psf_im = psf.withGSParams(gsp).drawImage(
            nx=npix, ny=npix, scale=scale, method=method, dtype=dtype).array
        return gal_im, psf_im

    return render_one


def make_render_fn(cfg: JaxRenderConfig, trace_psf_shear: bool = False):
    """Build ``jit(vmap(render_one))``: the differentiable rendering primitive.

    Returns a callable ``render(params, psf_batch) -> (gal, psf)`` where

    * ``params`` is a dict of ``(N,)`` arrays keyed by :data:`PARAM_NAMES`,
    * ``psf_batch`` is the stacked ``DES_PSFEx`` pytree for ``exp='superbit'``
      and ``None`` for ``exp='ideal'``,
    * ``gal`` is ``(N, npix, npix)`` noiseless and ``psf`` likewise.

    Because it is a pure function of ``params``, every physical derivative is
    one JVP away::

        tangent = {k: jnp.zeros_like(v) for k, v in params.items()}
        tangent["base_g1"] = jnp.ones_like(params["base_g1"])
        gal, dgal_dgamma1 = jax.jvp(lambda p: render(p, psf_batch)[0],
                                    (params,), (tangent,))

    Swap which key is set to one and the same call yields ``dI/dgamma2``,
    ``dI/dg^psf``, ``dI/ddx`` (translation), ``dI/dhlr``, ``dI/dflux``.

    Args:
        cfg: static render settings.
        trace_psf_shear: apply ``.shear(psf_g1, psf_g2)`` to the PSF even when
            both are zero. Off by default so a run with ``apply_psf_shear:
            false`` builds exactly the profile the GalSim path builds; turn it
            on to expose the ``R^PSF`` derivative handle.
    """
    import jax

    render_one = make_render_one(cfg, trace_psf_shear)
    superbit = cfg.exp == "superbit"

    if superbit:
        return jax.jit(jax.vmap(render_one))
    # no PSF pytree to map over; keep the same 2-argument call signature
    inner = jax.jit(jax.vmap(lambda p: render_one(p, None)))
    return lambda params, psf_batch=None: inner(params)


# ----------------------------------------------------------------------
# truth sampling -- byte-identical to the galsim path
# ----------------------------------------------------------------------
@dataclass
class Truth:
    """Per-object draws plus the bookkeeping the renderer needs."""

    params: dict                       # PARAM_NAMES -> (N,) float64
    psf_files: Optional[list] = None   # (N,) chosen PSFEx paths, superbit only
    noise: Optional[np.ndarray] = None  # (N, npix, npix)
    labels_raw: dict = field(default_factory=dict)


def sample_truth(
    samples: int,
    cfg: JaxRenderConfig,
    seed: int = 42,
    nse_sd: float = 1e-5,
    apply_psf_shear: bool = False,
    psf_shear_range: float = 0.05,
    base_shear_g1: float = 0.0,
    base_shear_g2: float = 0.0,
    psf_file_or_dir: str = PSF_DATA_DIR,
    hlr_type: str = "constant",
    flux_type: str = "constant",
    cosmos_cat_fname: Optional[str] = None,
    add_noise: bool = True,
) -> Truth:
    """Draw every per-object quantity, matching ``sim_func`` draw-for-draw.

    ``sim_func`` seeds ``np.random.RandomState(seed=i)`` per object and consumes
    it in a fixed order -- PSF shear (only if requested), then the shift, then
    two noise fields. The PSFEx file and focal-plane position come from a
    separate ``galsim.UniformDeviate(seed*1_000_003 + i)``. Both are reproduced
    here, including the conditional consumption, so switching backends changes
    the renderer and nothing else.
    """
    import galsim  # only for UniformDeviate parity; no rendering happens here

    cat = _load_cosmos_cat(seed=seed, cat_path=cosmos_cat_fname)
    g1_list, g2_list = cat["G1"], cat["G2"]
    hlr_list, flux_list = cat["HLR"], cat["FLUX"]
    if samples > len(g1_list):
        raise ValueError(
            f"Requested {samples} samples but the catalog has only "
            f"{len(g1_list)} rows. Lower `samples` or use a larger catalog."
        )

    psf_paths = None
    if cfg.exp == "superbit":
        if os.path.isfile(psf_file_or_dir):
            psf_paths = [psf_file_or_dir]
        elif os.path.isdir(psf_file_or_dir):
            psf_paths = search_psf_files(path=psf_file_or_dir)
            if not psf_paths:
                raise FileNotFoundError(f"No PSF files found in {psf_file_or_dir}")
        else:
            raise FileNotFoundError(
                f"{psf_file_or_dir} is neither a file nor a directory")

    p = {k: np.zeros(samples, dtype=np.float64) for k in PARAM_NAMES}
    chosen, noise = [], (np.empty((samples, cfg.npix, cfg.npix)) if add_noise else None)

    for i in range(samples):
        rng = np.random.RandomState(seed=i)          # sim_func: seed=i
        if apply_psf_shear:
            p["psf_g1"][i] = rng.uniform(-psf_shear_range, psf_shear_range)
            p["psf_g2"][i] = rng.uniform(-psf_shear_range, psf_shear_range)
        dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
        p["dx"][i], p["dy"][i] = dx, dy
        if add_noise:
            noise[i] = rng.normal(size=(cfg.npix, cfg.npix), scale=nse_sd)

        p["g1"][i], p["g2"][i] = g1_list[i], g2_list[i]
        p["hlr"][i] = float(hlr_list[i]) if hlr_type == "catalog" else 0.5
        p["flux"][i] = float(flux_list[i]) if flux_type == "catalog" else 1.0
        p["base_g1"][i], p["base_g2"][i] = base_shear_g1, base_shear_g2

        if psf_paths is not None:
            ud = galsim.UniformDeviate(int(seed) * 1_000_003 + int(i))
            p["psf_x"][i] = MARGIN + (WCS_PARAMS["image_xsize"] - 2 * MARGIN) * ud()
            p["psf_y"][i] = MARGIN + (WCS_PARAMS["image_ysize"] - 2 * MARGIN) * ud()
            chosen.append(psf_paths[int(len(psf_paths) * ud())])

    return Truth(
        params=p,
        psf_files=(chosen or None),
        noise=noise,
        labels_raw={"g1": p["g1"], "g2": p["g2"],
                    "hlr": p["hlr"], "flux": p["flux"]},
    )


# ----------------------------------------------------------------------
# batched generation
# ----------------------------------------------------------------------
def render_truth(truth: Truth, cfg: JaxRenderConfig, trace_psf_shear: bool = False):
    """Render a :class:`Truth` in pinned-FFT batches. Returns ``(gal, psf)`` numpy.

    The final partial batch is padded up to ``batch_size`` before dispatch: a
    short batch is a different input shape and would trigger a full XLA
    recompile, which on a per-epoch generator costs more than the wasted
    stamps.

    Precision comes from the process (see :func:`render_dtype`). Under float32
    the render is good to ~1e-7 and the PSF stamp -- the sharpest thing on the
    grid -- degrades first.
    """
    import jax.numpy as jnp

    n = len(truth.params["g1"])
    bs = min(cfg.batch_size, n) or 1
    render = make_render_fn(cfg, trace_psf_shear=trace_psf_shear)

    gal = np.empty((n, cfg.npix, cfg.npix))
    psf = np.empty((n, cfg.npix, cfg.npix))

    for start in range(0, n, bs):
        stop = min(start + bs, n)
        pad = bs - (stop - start)
        sl = slice(start, stop)
        batch = {k: jnp.asarray(np.concatenate(
            [v[sl], np.repeat(v[stop - 1:stop], pad)]) if pad else v[sl])
            for k, v in truth.params.items()}
        psf_batch = None
        if cfg.exp == "superbit":
            files = truth.psf_files[sl] + [truth.psf_files[stop - 1]] * pad
            psf_batch = _stack_psfex(files)
        g_b, p_b = render(batch, psf_batch)
        keep = stop - start
        gal[sl] = np.asarray(g_b)[:keep]
        psf[sl] = np.asarray(p_b)[:keep]

    return gal, psf


def generate_dataset_jax(
    samples,
    psf_fwhm,
    npix=53,
    scale=0.141,
    type="exp",
    exp="ideal",
    nse_sd=1e-5,
    seed=42,
    return_psf=False,
    return_obs=False,
    apply_psf_shear=False,
    psf_shear_range=0.05,
    base_shear_g1=0.0,
    base_shear_g2=0.0,
    psf_file_or_dir=PSF_DATA_DIR,
    output_keys=("g1", "g2"),
    hlr_type="constant",
    flux_type="constant",
    cosmos_cat_fname=None,
    as_result=False,
    add_noise=True,
    jax_fft_size=256,
    jax_batch_size=256,
    **_ignored,
):
    """jax-galsim twin of :func:`shearnet.core.dataset.generate_dataset`.

    Same arguments, same return shapes, so the two are interchangeable behind
    ``dataset.backend``. Arguments that only make sense for the GalSim path
    (``nproc``, ``compute_metacal``, ``return_clean``) are accepted and ignored:
    batched rendering has no worker pool, and metacal reconvolutions are not
    needed by a backend whose entire point is analytic responses.

    ``return_obs`` is **not** supported -- ngmix ``Observation`` packaging is a
    GalSim-path concept and the benchmarks that need it run on that path.
    """
    if return_obs:
        raise NotImplementedError(
            "return_obs is not supported by the jax-galsim backend; use "
            "dataset.backend: galsim for the ngmix Observation path."
        )

    _valid = {"g1", "g2", "hlr", "flux", "psf_e1", "psf_e2", "psf_T"}
    requested = set(output_keys)
    if not requested.issubset(_valid):
        raise ValueError(
            f"Invalid output_keys: {requested - _valid}. Must be subset of {_valid}.")
    if hlr_type not in ("catalog", "constant"):
        raise ValueError("hlr can only be 'constant' or 'catalog'")
    if flux_type not in ("catalog", "constant"):
        raise ValueError("flux can only be 'constant' or 'catalog'")

    cfg = JaxRenderConfig(
        npix=npix, scale=scale, psf_fwhm=psf_fwhm, gal_type=type, exp=exp,
        fft_size=jax_fft_size, batch_size=jax_batch_size,
    )

    truth = sample_truth(
        samples, cfg, seed=seed, nse_sd=nse_sd,
        apply_psf_shear=apply_psf_shear, psf_shear_range=psf_shear_range,
        base_shear_g1=base_shear_g1, base_shear_g2=base_shear_g2,
        psf_file_or_dir=psf_file_or_dir, hlr_type=hlr_type,
        flux_type=flux_type, cosmos_cat_fname=cosmos_cat_fname,
        add_noise=add_noise,
    )

    logger.info(
        "jax-galsim generation: %d samples, exp=%s, fft_size=%d, batch=%d, dtype=%s",
        samples, exp, jax_fft_size, jax_batch_size, render_dtype().__name__,
    )
    gal, psf = render_truth(truth, cfg, trace_psf_shear=apply_psf_shear)

    if add_noise:
        gal = gal + truth.noise

    labels = _build_labels(truth, psf, output_keys, scale)

    images = np.stack([gal, psf], axis=-1) if return_psf else gal
    images = images.astype(np.float32)
    if as_result:
        return DatasetResult(images=images, labels=labels, obs=None)
    return images, labels


def _build_labels(truth: Truth, psf_stamps, output_keys, scale):
    """Assemble the label array, fitting PSF adaptive moments only if asked."""
    import ngmix

    n = len(truth.params["g1"])
    avail = dict(truth.labels_raw)
    if {"psf_e1", "psf_e2", "psf_T"} & set(output_keys):
        e1 = np.full(n, np.nan)
        e2 = np.full(n, np.nan)
        tt = np.full(n, np.nan)
        cen = psf_stamps.shape[1] // 2
        jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)
        for i in range(n):
            im = psf_stamps[i]
            target = im.max() / 1000.0
            obs = ngmix.Observation(
                image=im, weight=np.ones_like(im) / target ** 2, jacobian=jac)
            res = get_admoms_ngmix_fit(obs=obs, reduced=True)
            if res["flags"] == 0:
                e1[i], e2[i], tt[i] = res["e1"], res["e2"], res["T"]
        avail.update({"psf_e1": e1, "psf_e2": e2, "psf_T": tt})

    return np.stack([np.asarray(avail[k], dtype=np.float32)
                     for k in output_keys], axis=1)
