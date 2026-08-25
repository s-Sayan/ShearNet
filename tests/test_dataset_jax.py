"""Tests for the differentiable jax-galsim dataset backend."""

import numpy as np
import pytest

jax_galsim = pytest.importorskip("jax_galsim")
jax = pytest.importorskip("jax")

import galsim  # noqa: E402  (hard dependency of the repo)
import galsim.des  # noqa: E402

from shearnet.core.dataset import WCS_PARAMS, draw_method_for  # noqa: E402
from shearnet.core.dataset_jax import (  # noqa: E402
    PARAM_NAMES,
    JaxRenderConfig,
    generate_dataset_jax,
    make_render_fn,
    render_dtype,
    render_truth,
    sample_truth,
)
from shearnet.core.specs import DatasetSpec  # noqa: E402
from shearnet.core.wcs import create_wcs_from_params  # noqa: E402

NPIX, SCALE = 53, 0.141
# 0.5" FWHM is what the unit-test configs actually use. The package default of
# 0.25" is only 1.8 pixels across at this scale -- see
# test_undersampled_psf_is_the_known_disagreement.
FWHM = 0.5

# Agreement with GalSim, and JVP-vs-finite-difference, are float64-level
# statements. Rendering follows the process-wide flag (it is never toggled at
# runtime -- see render_dtype), so these assertions only hold under
# JAX_ENABLE_X64=1, which is what setup_env.sh exports.
needs_f64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="float64-level agreement; run with JAX_ENABLE_X64=1",
)


def _galsim_reference(truth, cfg, n):
    """Render the same truth with GalSim, mirroring sim_func's construction."""
    gsp = galsim.GSParams(maximum_fft_size=32768)
    p = truth.params
    gal = np.empty((n, cfg.npix, cfg.npix))
    psf = np.empty((n, cfg.npix, cfg.npix))
    wcs = create_wcs_from_params(WCS_PARAMS) if cfg.exp == "superbit" else None
    for i in range(n):
        prof = galsim.Exponential if cfg.gal_type == "exp" else galsim.Gaussian
        g = prof(half_light_radius=p["hlr"][i], flux=p["flux"][i])
        g = g.shear(g1=p["g1"][i], g2=p["g2"][i])
        g = g.shift(p["dx"][i], p["dy"][i])
        g = g.shear(g1=p["base_g1"][i], g2=p["base_g2"][i])
        if cfg.exp == "superbit":
            ps = galsim.des.DES_PSFEx(truth.psf_files[i], wcs=wcs).getPSF(
                galsim.PositionD(p["psf_x"][i], p["psf_y"][i]))
        else:
            ps = galsim.Gaussian(fwhm=cfg.psf_fwhm)
        kw = dict(nx=cfg.npix, ny=cfg.npix, scale=cfg.scale,
                  dtype=np.float64, method=cfg.draw_method)
        gal[i] = galsim.Convolve([g, ps], gsparams=gsp).withGSParams(gsp).drawImage(**kw).array
        psf[i] = ps.withGSParams(gsp).drawImage(**kw).array
    return gal, psf


def _rel(a, b):
    return np.abs(a - b).max() / np.abs(b).sum(axis=(1, 2)).max()


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------
def test_render_config_validation():
    for bad in (dict(gal_type="sersic"), dict(exp="hst"),
                dict(fft_size=255), dict(fft_size=0), dict(npix=0)):
        with pytest.raises(ValueError):
            JaxRenderConfig(**bad)
    JaxRenderConfig()  # defaults must be valid


def test_draw_method_is_derived_from_psf_mode():
    """Not configurable: an empirical PSF must never be re-convolved by the pixel."""
    assert JaxRenderConfig(exp="ideal").draw_method == "auto"
    assert JaxRenderConfig(exp="superbit").draw_method == "no_pixel"
    assert draw_method_for("ideal") == "auto"
    assert draw_method_for("superbit") == "no_pixel"
    with pytest.raises(TypeError):
        JaxRenderConfig(draw_method="auto")      # the knob is gone


# ----------------------------------------------------------------------
# truth sampling parity
# ----------------------------------------------------------------------
def test_sample_truth_matches_sim_func_draw_order():
    """dx/dy must come off RandomState(seed=i) in sim_func's exact order."""
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM)
    t = sample_truth(5, cfg, seed=0, nse_sd=1e-5, add_noise=True)
    for i in range(5):
        rng = np.random.RandomState(seed=i)
        dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
        assert t.params["dx"][i] == pytest.approx(dx)
        assert t.params["dy"][i] == pytest.approx(dy)
        nse = rng.normal(size=(NPIX, NPIX), scale=1e-5)
        assert np.allclose(t.noise[i], nse)


def test_sample_truth_psf_shear_consumes_rng_first():
    """With apply_psf_shear the PSF draws come BEFORE the shift, as in sim_func."""
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM)
    t = sample_truth(3, cfg, seed=0, apply_psf_shear=True,
                     psf_shear_range=0.05, add_noise=False)
    for i in range(3):
        rng = np.random.RandomState(seed=i)
        assert t.params["psf_g1"][i] == pytest.approx(rng.uniform(-0.05, 0.05))
        assert t.params["psf_g2"][i] == pytest.approx(rng.uniform(-0.05, 0.05))
        dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
        assert t.params["dx"][i] == pytest.approx(dx)


def test_sample_truth_is_deterministic():
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM)
    a = sample_truth(4, cfg, seed=7, add_noise=True)
    b = sample_truth(4, cfg, seed=7, add_noise=True)
    for k in PARAM_NAMES:
        assert np.array_equal(a.params[k], b.params[k])
    assert np.array_equal(a.noise, b.noise)


def test_sample_truth_rejects_oversized_request():
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM)
    with pytest.raises(ValueError, match="catalog has only"):
        sample_truth(10 ** 7, cfg)


# ----------------------------------------------------------------------
# render agreement with galsim
# ----------------------------------------------------------------------
@needs_f64
def test_ideal_render_matches_galsim():
    n = 6
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM, exp="ideal",
                          batch_size=n, fft_size=256)
    t = sample_truth(n, cfg, seed=0, add_noise=False)
    gj, pj = render_truth(t, cfg)
    gg, pg = _galsim_reference(t, cfg, n)
    assert _rel(gj, gg) < 1e-8
    assert _rel(pj, pg) < 1e-6


@needs_f64
def test_superbit_render_matches_galsim():
    """Stacked DES_PSFEx models from *different* files, rendered in one vmap."""
    n = 4
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM, exp="superbit",
                          batch_size=n, fft_size=256)
    try:
        t = sample_truth(n, cfg, seed=0, add_noise=False)
    except FileNotFoundError:
        pytest.skip("no PSFEx files available")
    gj, pj = render_truth(t, cfg)
    gg, pg = _galsim_reference(t, cfg, n)
    assert len(set(t.psf_files)) > 1, "test should exercise >1 PSFEx file"
    assert _rel(gj, gg) < 1e-4
    assert _rel(pj, pg) < 1e-12      # no_pixel: PSF stamp is exact


@needs_f64
def test_undersampled_psf_is_the_known_disagreement():
    """Document, and pin, why the default psf_fwhm moved from 0.25 to 0.5.

    A 0.25" FWHM Gaussian is 1.8 pixels across at 0.141"/pix. Integrating that
    over the pixel -- which ``exp='ideal'`` must do -- is the one regime where
    GalSim and jax-galsim disagree, at ~5e-3. At the 0.5" default the same
    comparison is ~1e-8, so this is undersampling, not a renderer bug.
    """
    n = 4
    common = dict(npix=NPIX, scale=SCALE, exp="ideal", batch_size=n, fft_size=256)

    under = JaxRenderConfig(psf_fwhm=0.25, **common)
    t = sample_truth(n, under, seed=0, add_noise=False)
    _, pj = render_truth(t, under)
    _, pg = _galsim_reference(t, under, n)
    assert _rel(pj, pg) > 1e-4                       # the known disagreement

    ok = JaxRenderConfig(psf_fwhm=0.5, **common)
    _, pj2 = render_truth(t, ok)
    _, pg2 = _galsim_reference(t, ok, n)
    assert _rel(pj2, pg2) < 1e-6                     # resolved -> agrees


def test_batch_size_does_not_change_output():
    """Chunking (and the final-batch padding) must be invisible."""
    n = 7
    mk = lambda bs: JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM,
                                    batch_size=bs, fft_size=256)
    t = sample_truth(n, mk(4), seed=1, add_noise=False)
    a, _ = render_truth(t, mk(4))     # 4 + padded 3
    b, _ = render_truth(t, mk(n))     # single full batch
    # determinism, not precision: scale the bound to the working dtype
    eps = 1e-12 if render_dtype() is np.float64 else 1e-4
    assert np.abs(a - b).max() / np.abs(b).max() < eps


# ----------------------------------------------------------------------
# differentiability -- the reason this backend exists
# ----------------------------------------------------------------------
@needs_f64
@pytest.mark.parametrize(
    "key", ["base_g1", "base_g2", "psf_g1", "psf_g2", "hlr", "flux", "dx", "dy"])
def test_jvp_matches_finite_difference(key):
    """dI/dtheta from one JVP must reproduce a central difference."""
    n = 3
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM, exp="ideal",
                          batch_size=n, fft_size=256)
    t = sample_truth(n, cfg, seed=0, add_noise=False)
    if True:
        import jax.numpy as jnp

        render = make_render_fn(cfg, trace_psf_shear=True)
        P = {k: jnp.asarray(v) for k, v in t.params.items()}
        tan = {k: jnp.zeros_like(v) for k, v in P.items()}
        tan[key] = jnp.ones_like(P[key])
        base, dgal = jax.jvp(lambda p: render(p, None)[0], (P,), (tan,))

        d = 1e-4
        plus, minus = dict(P), dict(P)
        plus[key], minus[key] = P[key] + d, P[key] - d
        fd = (np.asarray(render(plus, None)[0])
              - np.asarray(render(minus, None)[0])) / (2 * d)
        flux = np.abs(np.asarray(base)).sum(axis=(1, 2)).max()
        assert np.abs(np.asarray(dgal) - fd).max() / flux < 1e-7
        # a derivative that is identically zero would pass trivially
        assert np.abs(np.asarray(dgal)).max() / flux > 1e-6


def test_psf_shear_tangent_is_off_by_default():
    """Without trace_psf_shear the PSF-shear handle is (correctly) dead."""
    n = 2
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM,
                          batch_size=n, fft_size=256)
    t = sample_truth(n, cfg, seed=0, add_noise=False)
    if True:
        import jax.numpy as jnp

        render = make_render_fn(cfg, trace_psf_shear=False)
        P = {k: jnp.asarray(v) for k, v in t.params.items()}
        tan = {k: jnp.zeros_like(v) for k, v in P.items()}
        tan["psf_g1"] = jnp.ones_like(P["psf_g1"])
        _, dgal = jax.jvp(lambda p: render(p, None)[0], (P,), (tan,))
        assert np.abs(np.asarray(dgal)).max() == 0.0


# ----------------------------------------------------------------------
# precision plumbing
# ----------------------------------------------------------------------
def test_render_dtype_follows_the_process():
    """Precision is read from JAX_ENABLE_X64, never toggled at runtime.

    Toggling it mid-process is unsafe: jax-galsim caches jitted internals (e.g.
    Lanczos._interp_kval) with no precision in the cache key, so a cached
    float32 interpolant reused under float64 fails at lowering.
    """
    want = np.float64 if jax.config.jax_enable_x64 else np.float32
    assert render_dtype() is want
    assert JaxRenderConfig().x64 is (want is np.float64)
    with pytest.raises(TypeError):
        JaxRenderConfig(x64=True)          # not a settable field


def test_drawimage_gets_an_explicit_dtype():
    """Guard the ImageF trap: drawImage is float32 unless given a dtype."""
    n = 2
    cfg = JaxRenderConfig(npix=NPIX, scale=SCALE, psf_fwhm=FWHM,
                          batch_size=n, fft_size=256)
    t = sample_truth(n, cfg, seed=0, add_noise=False)
    render = make_render_fn(cfg)
    import jax.numpy as jnp
    P = {k: jnp.asarray(v) for k, v in t.params.items()}
    assert render(P, None)[0].dtype == render_dtype()


# ----------------------------------------------------------------------
# generate_dataset_jax: drop-in behaviour
# ----------------------------------------------------------------------
def test_generate_dataset_jax_shapes_and_labels():
    im, lab = generate_dataset_jax(6, psf_fwhm=FWHM, npix=33, seed=0,
                                   jax_batch_size=6,
                                   output_keys=("g1", "g2", "hlr", "flux"))
    assert im.shape == (6, 33, 33)
    assert im.dtype == np.float32
    assert lab.shape == (6, 4)
    assert np.isfinite(im).all() and np.isfinite(lab).all()


def test_generate_dataset_jax_return_psf_channel():
    im, _ = generate_dataset_jax(4, psf_fwhm=FWHM, npix=33, seed=0,
                                 return_psf=True, jax_batch_size=4)
    assert im.shape == (4, 33, 33, 2)


def test_generate_dataset_jax_as_result():
    res = generate_dataset_jax(4, psf_fwhm=FWHM, npix=33, seed=0,
                               as_result=True, jax_batch_size=4)
    assert res.obs is None
    assert res.images.shape == (4, 33, 33)


def test_generate_dataset_jax_noise_toggle():
    kw = dict(psf_fwhm=FWHM, npix=33, seed=0, jax_batch_size=4, nse_sd=1e-3)
    noisy, _ = generate_dataset_jax(4, add_noise=True, **kw)
    clean, _ = generate_dataset_jax(4, add_noise=False, **kw)
    assert not np.allclose(noisy, clean)
    assert np.abs(noisy - clean).max() > 1e-4


def test_generate_dataset_jax_rejects_return_obs():
    with pytest.raises(NotImplementedError, match="return_obs"):
        generate_dataset_jax(2, psf_fwhm=FWHM, npix=33, return_obs=True)


def test_generate_dataset_jax_validates_output_keys():
    with pytest.raises(ValueError, match="Invalid output_keys"):
        generate_dataset_jax(2, psf_fwhm=FWHM, npix=33, output_keys=("g1", "nope"))


def test_generate_dataset_jax_ignores_galsim_only_kwargs():
    """nproc/compute_metacal are accepted and ignored, so a shared spec works."""
    im, _ = generate_dataset_jax(4, psf_fwhm=FWHM, npix=33, seed=0,
                                 jax_batch_size=4, nproc=8, compute_metacal=True)
    assert im.shape == (4, 33, 33)


# ----------------------------------------------------------------------
# config plumbing
# ----------------------------------------------------------------------
def test_dataset_spec_defaults_to_galsim():
    spec = DatasetSpec(samples=4, psf_fwhm=FWHM)
    assert spec.backend == "galsim"
    kw = spec.as_kwargs()
    assert "backend" not in kw
    assert "nproc" in kw and "jax_fft_size" not in kw


def test_dataset_spec_jax_backend_filters_kwargs():
    spec = DatasetSpec(samples=4, psf_fwhm=FWHM, backend="jax-galsim")
    kw = spec.as_kwargs()
    assert "jax_fft_size" in kw and "jax_batch_size" in kw
    assert "draw_method" not in kw
    assert "nproc" not in kw and "compute_metacal" not in kw
    assert "backend" not in kw


def test_dataset_spec_rejects_unknown_backend():
    with pytest.raises(ValueError, match="dataset.backend"):
        DatasetSpec(samples=4, psf_fwhm=FWHM, backend="pytorch")


def test_dataset_spec_build_dispatches_to_jax():
    spec = DatasetSpec(samples=4, psf_fwhm=FWHM, npix=33, seed=0,
                       backend="jax-galsim", jax_batch_size=4)
    im, lab = spec.build()
    ref, _ = generate_dataset_jax(4, psf_fwhm=FWHM, npix=33, seed=0,
                                  jax_batch_size=4)
    assert np.array_equal(im, ref)


def test_psf_directory_draw_is_deterministic_and_spread():
    """A directory of PSFEx files must give a stable, seeded, spread-out draw.

    search_psf_files used to return raw glob order, which is directory order and
    therefore filesystem-dependent -- on the 50-file SuperBIT set it comes back
    emp47, emp48, emp26, emp11, ... Since sample_truth picks each object's PSF as
    psf_paths[int(len(psf_paths) * ud())], that index is part of the seeded draw,
    so an unsorted list means the same seed picks different PSFs on a different
    machine.
    """
    import os
    from glob import glob

    from shearnet.core.dataset import search_psf_files

    directory = os.environ.get("SHEARNET_TEST_PSF_DIR")
    if not directory or not os.path.isdir(directory):
        pytest.skip("no PSFEx directory available")
    found = search_psf_files(directory)
    if len(found) < 2:
        pytest.skip("need at least two .psf files")

    assert found == sorted(found), "search_psf_files must sort"
    # and the sort must actually be doing something -- otherwise this test
    # would pass on a filesystem that happens to return sorted order
    assert set(found) == set(glob(os.path.join(directory, "*.psf")))

    cfg = JaxRenderConfig(npix=53, scale=0.141, psf_fwhm="superbit", exp="superbit",
                          fft_size=256, batch_size=16)
    kw = dict(seed=7, add_noise=False, hlr_type="constant", flux_type="constant")
    a = sample_truth(64, cfg, psf_file_or_dir=directory, **kw)
    b = sample_truth(64, cfg, psf_file_or_dir=directory, **kw)
    assert a.psf_files == b.psf_files, "same seed must give the same PSF files"
    # the draw spreads over the directory rather than collapsing onto one file
    assert len(set(a.psf_files)) > 1
    # a single file still works and yields exactly that file
    one = sample_truth(8, cfg, psf_file_or_dir=found[0], **kw)
    assert set(one.psf_files) == {found[0]}
