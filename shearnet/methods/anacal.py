"""AnaCal-backed shear calibration for isolated postage stamps.

Three estimators have to be compared on the same footing: FPFS (AnaCal's own),
ngmix, and the network. They do not share a response mechanism -- FPFS carries
an analytic ``de/dg`` in its catalogue, ngmix has none, and the network has an
exact JVP -- so this module provides *one* protocol they can all be measured
under, plus each estimator's native measurement.

The response protocol
---------------------
The shear response of any estimator is

.. math:: R_{ab} = \\partial \\hat{e}_a / \\partial \\gamma_b

and because the simulator is ours, that derivative can be taken by re-rendering
the *scene* at :math:`\\gamma \\pm \\delta` with the noise realisation held
fixed and re-measuring. That works for an estimator that is a black box (ngmix)
and for one that is differentiable (the network), and it is the same object
metacal estimates by shearing the *image* instead of the scene. Measuring every
estimator this way is what makes the comparison a comparison; FPFS's analytic
response is then a free consistency check on the protocol itself.

Deconvolution
-------------
FPFS deconvolves by construction -- that is what it is -- and
:func:`measure_fpfs` uses AnaCal's implementation unmodified. **The network is
never deconvolved.** It is trained on convolved stamps and evaluated on
convolved stamps; nothing here Fourier-divides its inputs. If a future estimator
needs a deconvolve/reconvolve round trip, say so at the call site, because for a
network trained without one it is an out-of-distribution input and any bias it
produces belongs to the round trip rather than to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "BiasEstimate",
    "ShapeMeasurement",
    "fpfs_shapes",
    "measure_fpfs",
    "paired_bias",
    "renderer_shear_response",
    "resmooth",
]

#: AnaCal's FPFS catalogue prefixes. ``fpfs1`` is the ``sigma_shapelets1``
#: kernel, always measured; ``fpfs2`` appears only when a second kernel is set.
FPFS_KERNELS = ("fpfs1", "fpfs2")


@dataclass(frozen=True)
class ShapeMeasurement:
    """Per-object ellipticity and its shear response.

    Attributes:
        e: ``(N, 2)`` ellipticity in whatever convention the estimator uses.
            It does not have to be reduced shear -- FPFS's is not -- because the
            response divides it out.
        dedg: ``(N, 2, 2)`` response, ``dedg[:, a, b] = de_a / dgamma_b``.
        flags: ``(N,)`` bool, True where the measurement failed. Failed rows are
            carried rather than dropped so plus/minus populations stay paired.
    """

    e: np.ndarray
    dedg: np.ndarray
    flags: Optional[np.ndarray] = None

    def __post_init__(self):
        e = np.asarray(self.e, dtype=float)
        dedg = np.asarray(self.dedg, dtype=float)
        if e.ndim != 2 or e.shape[1] != 2:
            raise ValueError(f"e must be (N, 2), got {e.shape}")
        if dedg.shape != (e.shape[0], 2, 2):
            raise ValueError(f"dedg must be {(e.shape[0], 2, 2)}, got {dedg.shape}")
        object.__setattr__(self, "e", e)
        object.__setattr__(self, "dedg", dedg)
        if self.flags is None:
            object.__setattr__(self, "flags", np.zeros(e.shape[0], dtype=bool))
        else:
            object.__setattr__(self, "flags", np.asarray(self.flags, dtype=bool))

    @property
    def n(self):
        return self.e.shape[0]


@dataclass(frozen=True)
class BiasEstimate:
    """Multiplicative and additive bias with jackknife errors.

    ``m`` and ``c`` follow the convention already used by
    ``research/shear_bias``: the estimate is the *pair-matched* difference
    ``(e_plus - e_minus) / 2`` divided by the mean response, and ``c`` is the
    additive bias in the component that was **not** sheared. Keeping that
    convention is what lets these numbers sit in the same table as the existing
    metacal results.
    """

    m: float
    m_err: float
    c: float
    c_err: float
    response: np.ndarray
    n_used: int
    bins: Optional[Mapping[str, np.ndarray]] = None

    def describe(self, label=""):
        head = f"{label}: " if label else ""
        return (
            f"{head}m = {self.m:+.5e} +/- {self.m_err:.1e}   "
            f"c = {self.c:+.5e} +/- {self.c_err:.1e}   "
            f"R = [[{self.response[0, 0]:.4f} {self.response[0, 1]:.4f}] "
            f"[{self.response[1, 0]:.4f} {self.response[1, 1]:.4f}]]   "
            f"N = {self.n_used}"
        )


# ----------------------------------------------------------------------
# FPFS, via AnaCal's own implementation
# ----------------------------------------------------------------------
def _anacal():
    """Import AnaCal only when an AnaCal measurement was actually requested."""
    try:
        import anacal
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "AnaCal calibration requires the optional 'anacal' package "
            "(pip install 'shearnet[calibration]', or the mr-superonion/AnaCal "
            "distribution). It is optional because it is only used by the "
            "research benchmarks."
        ) from exc
    return anacal


def measure_fpfs(
    galaxy_images: np.ndarray,
    psf_images: np.ndarray,
    *,
    pixel_scale: float,
    noise_variance: float,
    sigma_shapelets: float = 0.52,
    c0: float = 30.0,
    mag_zero: float = 30.0,
    noise_images: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run AnaCal's FPFS pipeline on already-detected isolated stamps.

    Detection is deliberately skipped: the benchmark measures *given* detections
    (that is the stated scope of the paper), so each stamp's centre is handed to
    AnaCal directly as a one-row forced catalogue rather than running the AnaCal
    detector over the image.

    One AnaCal call per stamp, not one per exposure. Tiling the stamps into a
    mosaic would be far faster, but ``process_image`` takes a single
    ``psf_array`` for the whole exposure, and the SuperBIT PSF varies per object.
    Correct per-object PSFs are worth more here than throughput.

    **Odd stamps are zero-padded to even, on the low edge.** AnaCal needs an even
    array *and* puts its Fourier origin at pixel ``npix // 2``; a GalSim-drawn
    odd stamp has the object at ``(npix - 1) // 2``, one pixel lower. Padding a
    row and a column onto the **low** edge moves the object onto ``npix // 2``
    exactly. This is not cosmetic: padding the high edge instead leaves the
    object one pixel off the origin and biases the recovered shear by about
    **-15%** -- large, constant, and indistinguishable from a real
    multiplicative bias in the output table.

    An even input stamp cannot be aligned this way: GalSim centres it on a pixel
    *corner*, half a pixel from any integer index, so there is no integer pad
    that lands it on the origin. Such stamps raise rather than silently carry
    that misalignment.

    Args:
        galaxy_images: ``(N, npix, npix)``.
        psf_images: ``(N, npix, npix)``, one PSF per stamp.
        pixel_scale: arcsec/pixel.
        noise_variance: per-pixel noise variance of ``galaxy_images``.
        sigma_shapelets: FPFS shapelet kernel scale (AnaCal's
            ``sigma_shapelets1``).
        noise_images: optional pure-noise stamps for AnaCal's noise-bias
            subtraction. Supplying them matters at low S/N.

    Returns the AnaCal structured catalogue, one row per stamp.
    """
    anacal = _anacal()
    galaxy_images = np.ascontiguousarray(galaxy_images, dtype=np.float64)
    psf_images = np.ascontiguousarray(psf_images, dtype=np.float64)
    if galaxy_images.ndim != 3 or psf_images.shape != galaxy_images.shape:
        raise ValueError("galaxy_images and psf_images must share an (N, Y, X) shape")
    if galaxy_images.shape[1] != galaxy_images.shape[2]:
        raise ValueError("FPFS requires square stamps")
    if not noise_variance > 0.0:
        raise ValueError("FPFS requires a positive noise_variance")
    if noise_images is not None:
        noise_images = np.ascontiguousarray(noise_images, dtype=np.float64)
        if noise_images.shape != galaxy_images.shape:
            raise ValueError("noise_images must match galaxy_images")

    npix = galaxy_images.shape[1]
    if npix % 2:
        pad = ((0, 0), (1, 0), (1, 0))
        galaxy_images = np.ascontiguousarray(np.pad(galaxy_images, pad))
        psf_images = np.ascontiguousarray(np.pad(psf_images, pad))
        if noise_images is not None:
            noise_images = np.ascontiguousarray(np.pad(noise_images, pad))
        logger.info(
            "FPFS: low-edge padded %d-pixel stamps to %d, putting the object on "
            "AnaCal's Fourier origin",
            npix,
            npix + 1,
        )
        npix += 1
    else:
        raise ValueError(
            f"stamp_size {npix} is even. GalSim centres an even stamp on a pixel "
            "corner, half a pixel from AnaCal's Fourier origin at npix // 2, and "
            "no integer padding fixes that -- the misalignment costs roughly 15% "
            "in m. Use an odd stamp_size (ShearNet's default is 53), or draw with "
            "a +0.5 pixel offset."
        )
    centre = npix // 2

    config = anacal.fpfs.FpfsConfig(
        npix=npix, sigma_shapelets1=float(sigma_shapelets), c0=float(c0)
    )
    detection = np.zeros(1, dtype=[("y", np.int32), ("x", np.int32)])
    detection["y"] = detection["x"] = centre

    logger.info(
        "FPFS: measuring %d stamps (sigma_shapelets=%.3g)", len(galaxy_images), sigma_shapelets
    )
    rows = []
    for i in range(len(galaxy_images)):
        rows.append(
            anacal.fpfs.process_image(
                fpfs_config=config,
                pixel_scale=float(pixel_scale),
                noise_variance=float(noise_variance),
                mag_zero=float(mag_zero),
                gal_array=galaxy_images[i],
                psf_array=psf_images[i],
                noise_array=None if noise_images is None else noise_images[i],
                detection=detection,
            )[0]
        )
    return np.asarray(rows)


def fpfs_shapes(catalogue: np.ndarray, kernel: str = "fpfs1") -> ShapeMeasurement:
    """Read AnaCal's analytic ellipticity and response out of its catalogue.

    FPFS's ``e`` is not a reduced shear -- for a ``|e| = 0.2`` galaxy it reads
    about half that -- which is exactly why the response has to divide it out
    rather than be assumed to be one.
    """
    if kernel not in FPFS_KERNELS:
        raise ValueError(f"kernel must be one of {FPFS_KERNELS}, got {kernel!r}")
    names = set(catalogue.dtype.names or ())
    required = tuple(
        f"{kernel}_{n}" for n in ("e1", "e2", "de1_dg1", "de1_dg2", "de2_dg1", "de2_dg2")
    )
    missing = sorted(set(required) - names)
    if missing:
        raise ValueError(
            f"AnaCal FPFS catalogue is missing {missing}. For {kernel!r} the "
            "measurement needs the matching sigma_shapelets kernel configured."
        )
    e = np.column_stack([catalogue[f"{kernel}_e1"], catalogue[f"{kernel}_e2"]])
    dedg = np.stack(
        [
            np.column_stack([catalogue[f"{kernel}_de1_dg1"], catalogue[f"{kernel}_de1_dg2"]]),
            np.column_stack([catalogue[f"{kernel}_de2_dg1"], catalogue[f"{kernel}_de2_dg2"]]),
        ],
        axis=1,
    )
    return ShapeMeasurement(e=e, dedg=dedg, flags=~np.isfinite(e).all(axis=1))


# ----------------------------------------------------------------------
# the AnaCal linear observable
# ----------------------------------------------------------------------
def resmooth(
    galaxy_images: np.ndarray,
    psf_images: np.ndarray,
    *,
    pixel_scale: float,
    sigma: float,
    noise_images: Optional[np.ndarray] = None,
    kmax_thres: float = 1e-12,
) -> np.ndarray:
    r"""AnaCal's re-smoothed, re-noised image ``f_h`` -- Eq. 3 of Lin et al. (2026).

    .. math::
        f_h(x) = \mathcal{F}^{-1}\left[
            \left( \frac{f_p(k)}{p(k)} - \frac{n(k)}{p_{90}(k)} \right) h(k)
        \right]

    Deconvolve by the PSF, subtract a noise realisation deconvolved by the
    *90-degree-rotated* PSF, then re-smooth with a Gaussian of width ``sigma``.

    This is the input domain the D4CNN of Lin et al. operates in, and it is
    **not** ShearNet's. Two consequences worth being explicit about:

    * The PSF is divided out before the network sees anything, so the model is
      PSF-homogenised by construction. Comparing its PSF leakage to that of a
      network fed the raw convolved stamp compares input pipelines, not
      architectures.
    * The deconvolution is why that model needs no PSF branch: a single-channel
      ``f_h`` carries everything. ShearNet's two-branch models take the observed
      stamp and the PSF stamp instead, and are never deconvolved.

    The rotated-PSF term is AnaCal's noise-bias cancellation: it injects a noise
    field with the same power but a PSF-orthogonal anisotropy, so the noise bias
    in the shape cancels in the ensemble. Pass ``noise_images=None`` for a
    noiseless population, where there is nothing to cancel.

    Args:
        galaxy_images: ``(N, npix, npix)`` observed (PSF-convolved) stamps.
        psf_images: ``(N, npix, npix)`` PSF stamps, normalised or not.
        pixel_scale: arcsec/pixel, so ``sigma`` is in arcsec.
        sigma: Gaussian re-smoothing width in arcsec. Must exceed the PSF size
            or the deconvolution amplifies noise without bound.
        noise_images: ``(N, npix, npix)`` pure-noise realisation for the
            re-noising term.
        kmax_thres: modes where ``|p(k)|`` falls below this fraction of
            ``|p(0)|`` are zeroed rather than divided -- the deconvolution is
            otherwise unbounded where the PSF has no power.

    Returns ``(N, npix, npix)`` real ``f_h``, centred on the same pixel the
    input was. Dividing by ``p(k)`` removes the PSF's centring phase along with
    its profile, so the raw quotient comes back centred on pixel ``(0, 0)``;
    the result is rolled back by the stamp centre. That is exact for an odd
    stamp, whose centre is an integer pixel, and half a pixel off for an even
    one -- which is why odd stamps are the well-defined case here as well as
    for :func:`measure_fpfs`.
    """
    galaxy_images = np.asarray(galaxy_images, dtype=np.float64)
    psf_images = np.asarray(psf_images, dtype=np.float64)
    if galaxy_images.ndim != 3 or psf_images.shape != galaxy_images.shape:
        raise ValueError("galaxy_images and psf_images must share an (N, Y, X) shape")
    if not sigma > 0.0:
        raise ValueError("resmoothing sigma must be positive")

    npix = galaxy_images.shape[-1]
    freq = np.fft.fftfreq(npix, d=pixel_scale) * 2.0 * np.pi
    kx, ky = np.meshgrid(freq, freq, indexing="xy")
    gaussian = np.exp(-0.5 * (kx**2 + ky**2) * sigma**2)

    psf_k = np.fft.fft2(psf_images)
    peak = np.abs(psf_k[:, :1, :1])
    keep = np.abs(psf_k) > kmax_thres * np.maximum(peak, np.finfo(float).tiny)
    safe = np.where(keep, psf_k, 1.0)
    ratio = np.where(keep, np.fft.fft2(galaxy_images) / safe, 0.0)

    if noise_images is not None:
        noise_images = np.asarray(noise_images, dtype=np.float64)
        if noise_images.shape != galaxy_images.shape:
            raise ValueError("noise_images must match galaxy_images")
        # p_90: the PSF rotated by 90 degrees, i.e. transposed axes in real space.
        psf90_k = np.fft.fft2(np.rot90(psf_images, k=1, axes=(-2, -1)))
        keep90 = np.abs(psf90_k) > kmax_thres * np.maximum(np.abs(psf90_k[:, :1, :1]), 1e-300)
        safe90 = np.where(keep90, psf90_k, 1.0)
        ratio = ratio - np.where(keep90, np.fft.fft2(noise_images) / safe90, 0.0)

    centre = (npix - 1) // 2
    if npix % 2 == 0:
        logger.warning(
            "resmooth: an even %d-pixel stamp has its object on a pixel corner, "
            "so the deconvolution phase cannot be undone by an integer roll; "
            "f_h will be half a pixel off centre.",
            npix,
        )
    return np.roll(np.real(np.fft.ifft2(ratio * gaussian[None])), (centre, centre), axis=(-2, -1))


# ----------------------------------------------------------------------
# the shared response protocol
# ----------------------------------------------------------------------
def renderer_shear_response(
    render: Callable[[float, float], Tuple[np.ndarray, np.ndarray]],
    measure: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    step: float = 0.01,
    psf: bool = False,
) -> np.ndarray:
    """Per-object ``de/dgamma`` by central-differencing the *scene*.

    ``render(d1, d2)`` must return ``(galaxy, psf)`` stamps for the population
    with an extra applied shear ``(d1, d2)`` -- and with **the same galaxies,
    the same PSFs and the same noise realisation** as the unperturbed call.
    ``measure(galaxy, psf)`` returns ``(N, 2)`` ellipticities.

    Fixing the noise across the five renders is what makes this usable at finite
    S/N: the noise-induced scatter cancels in the difference exactly, so the
    step size is limited by the estimator's nonlinearity, not by noise.

    Args:
        step: half-width of the central difference. Matching metacal's
            ``mcal_shear`` makes the two responses directly comparable.
        psf: differentiate with respect to the *PSF* shear instead, giving
            ``R^PSF``. ``render`` then receives the PSF-shear offsets.

    Returns ``(N, 2, 2)`` with ``[:, a, b] = de_a / dgamma_b``.
    """
    del psf  # signalled to the caller through `render`; kept for call-site clarity
    if not step > 0.0:
        raise ValueError("step must be positive")
    columns = []
    for axis in (0, 1):
        offset_p = (step, 0.0) if axis == 0 else (0.0, step)
        offset_m = (-step, 0.0) if axis == 0 else (0.0, -step)
        e_p = np.asarray(measure(*render(*offset_p)), dtype=float)[:, :2]
        e_m = np.asarray(measure(*render(*offset_m)), dtype=float)[:, :2]
        columns.append((e_p - e_m) / (2.0 * step))
    return np.stack(columns, axis=-1)


# ----------------------------------------------------------------------
# bias estimation
# ----------------------------------------------------------------------
def _weighted_mean(values, keep):
    return np.nanmean(values[keep], axis=0)


def paired_bias(
    plus: ShapeMeasurement,
    minus: ShapeMeasurement,
    applied_shear: float,
    *,
    component: int = 0,
    njac: int = 20,
    c_convention: str = "shearnet",
    resample: str = "jackknife",
    bin_values: Optional[np.ndarray] = None,
    nbins: int = 8,
) -> BiasEstimate:
    """Pair-matched ``m``/``c`` with jackknife errors.

    ``plus`` and ``minus`` must be the *same galaxies* rendered with
    ``+applied_shear`` and ``-applied_shear``, row for row. The estimator is the
    one ``research/shear_bias/m/helpers.jackknife_mc_v2`` already uses --
    per-object difference over pair-mean response --

    .. math::
        \\hat{\\gamma} = \\frac{\\langle (e^+ - e^-)/2 \\rangle}
                              {\\langle (R^+ + R^-)/2 \\rangle},
        \\quad m = \\hat{\\gamma}/\\gamma_\\mathrm{true} - 1

    so shape noise cancels in the numerator instead of being averaged down.
    Ratio-of-means, not mean-of-ratios: a per-object ``R`` can be near zero at
    low S/N and the per-object ratio has no finite variance.

    Two conventions for ``c`` are in circulation and they are not the same
    number, so it is a parameter rather than a choice buried in the code:

    * ``'shearnet'`` (default) is what ``jackknife_mc_v2`` reports -- the raw
      mean shape in the component that was **not** sheared, uncalibrated.
    * ``'lin2026'`` is Eq. 23 of Lin et al. (2026) -- the mean shape in the
      **sheared** component, divided by the mean response. Use it when a number
      has to sit next to theirs.

    ``resample`` selects ``'jackknife'`` (delete-one-block, matching the existing
    harness) or ``'bootstrap'`` (matching Lin et al.'s 20x bootstrap). They
    estimate the same variance; mixing them across a table is what to avoid.

    A row is dropped only if it failed in *either* population, so the pairing is
    never broken. ``bin_values`` (one scalar per object, e.g. flux or S/N)
    produces the same statistics per quantile stratum.
    """
    if plus.n != minus.n:
        raise ValueError(f"paired populations differ in size: {plus.n} vs {minus.n}")
    if applied_shear == 0.0:
        raise ValueError("applied_shear must be nonzero")
    if component not in (0, 1):
        raise ValueError("component must be 0 (g1) or 1 (g2)")
    other = 1 - component

    keep = ~(plus.flags | minus.flags)
    keep &= np.isfinite(plus.e).all(axis=1) & np.isfinite(minus.e).all(axis=1)
    keep &= np.isfinite(plus.dedg).all(axis=(1, 2)) & np.isfinite(minus.dedg).all(axis=(1, 2))
    n_used = int(np.sum(keep))
    if n_used < 2:
        raise RuntimeError(f"only {n_used} paired measurements survived; cannot estimate a bias")
    if njac < 2:
        raise ValueError("njac must be >= 2")
    njac = min(njac, n_used)

    e_diff = 0.5 * (plus.e[keep] - minus.e[keep])
    e_sum = 0.5 * (plus.e[keep] + minus.e[keep])
    r_pair = 0.5 * (plus.dedg[keep] + minus.dedg[keep])

    if c_convention not in ("shearnet", "lin2026"):
        raise ValueError(f"c_convention must be 'shearnet' or 'lin2026', got {c_convention!r}")
    if resample not in ("jackknife", "bootstrap"):
        raise ValueError(f"resample must be 'jackknife' or 'bootstrap', got {resample!r}")

    def _stats(index):
        r_mean = np.nanmean(r_pair[index, component, component])
        gamma = np.nanmean(e_diff[index, component]) / r_mean
        if c_convention == "lin2026":
            c_value = np.nanmean(e_sum[index, component]) / r_mean
        else:
            c_value = np.nanmean(e_sum[index, other])
        return gamma / applied_shear - 1.0, c_value

    every = np.arange(n_used)
    m_full, c_full = _stats(every)
    response = np.nanmean(r_pair, axis=0)

    if resample == "jackknife":
        samples = [np.setdiff1d(every, chunk) for chunk in np.array_split(every, njac)]
        factor = njac - 1
    else:
        rng = np.random.RandomState(n_used)
        samples = [rng.randint(0, n_used, n_used) for _ in range(njac)]
        factor = 1.0
    drawn = np.asarray([_stats(index) for index in samples])
    m_err = np.sqrt(factor * np.mean((drawn[:, 0] - drawn[:, 0].mean()) ** 2))
    c_err = np.sqrt(factor * np.mean((drawn[:, 1] - drawn[:, 1].mean()) ** 2))

    bins = None
    if bin_values is not None:
        values = np.asarray(bin_values)[keep]
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, nbins + 1)))
        rows = {"edges": edges, "m": [], "m_err": [], "c": [], "c_err": [], "count": []}
        for j, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            last = j == len(edges) - 2
            mask = (values >= low) & (values <= high if last else values < high)
            if int(np.sum(mask)) < njac:
                rows["m"].append(np.nan)
                rows["m_err"].append(np.nan)
                rows["c"].append(np.nan)
                rows["c_err"].append(np.nan)
                rows["count"].append(int(np.sum(mask)))
                continue
            sub_plus = ShapeMeasurement(plus.e[keep][mask], plus.dedg[keep][mask])
            sub_minus = ShapeMeasurement(minus.e[keep][mask], minus.dedg[keep][mask])
            sub = paired_bias(
                sub_plus,
                sub_minus,
                applied_shear,
                component=component,
                njac=njac,
                c_convention=c_convention,
                resample=resample,
            )
            rows["m"].append(sub.m)
            rows["m_err"].append(sub.m_err)
            rows["c"].append(sub.c)
            rows["c_err"].append(sub.c_err)
            rows["count"].append(sub.n_used)
        bins = {k: np.asarray(v) for k, v in rows.items()}

    return BiasEstimate(
        m=float(m_full),
        m_err=float(m_err),
        c=float(c_full),
        c_err=float(c_err),
        response=response,
        n_used=n_used,
        bins=bins,
    )


def leakage(
    e: np.ndarray,
    psf_response: np.ndarray,
    *,
    njac: int = 20,
) -> Mapping[str, np.ndarray]:
    """Mean ellipticity and mean ``R^PSF`` with jackknife errors.

    The PSF-leakage counterpart of :func:`paired_bias`: there is no plus/minus
    pair, so the numbers are the raw ensemble mean shape (which a perfect
    estimator leaves at the mean intrinsic shape) and the mean PSF response
    (which a perfect estimator leaves at zero).
    """
    e = np.asarray(e, dtype=float)[:, :2]
    psf_response = np.asarray(psf_response, dtype=float)
    keep = np.isfinite(e).all(axis=1) & np.isfinite(psf_response).all(axis=(1, 2))
    n = int(np.sum(keep))
    if n < 2:
        raise RuntimeError("no finite leakage measurements")
    njac = min(max(njac, 2), n)
    e, psf_response = e[keep], psf_response[keep]

    chunks = np.array_split(np.arange(n), njac)

    def _jack(values):
        full = np.nanmean(values, axis=0)
        samples = []
        for chunk in chunks:
            mask = np.ones(n, dtype=bool)
            mask[chunk] = False
            samples.append(np.nanmean(values[mask], axis=0))
        samples = np.asarray(samples)
        err = np.sqrt((njac - 1) * np.mean((samples - samples.mean(axis=0)) ** 2, axis=0))
        return full, err

    e_mean, e_err = _jack(e)
    r_mean, r_err = _jack(psf_response)
    return {
        "mean_e": e_mean,
        "mean_e_err": e_err,
        "psf_response": r_mean,
        "psf_response_err": r_err,
        "n_used": np.asarray(n),
    }


__all__.append("leakage")
