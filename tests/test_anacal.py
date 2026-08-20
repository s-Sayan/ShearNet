"""Tests for the AnaCal calibration bridge.

These check the module against AnaCal and against *known* shears, not against a
hand-written catalogue in the shape the module happens to expect. A test that
builds a fake catalogue with the column names the reader looks for proves the
reader is self-consistent and nothing about whether AnaCal produces them.
"""

import numpy as np
import pytest

galsim = pytest.importorskip("galsim")
anacal = pytest.importorskip("anacal")

from shearnet.methods.anacal import (  # noqa: E402
    ShapeMeasurement,
    fpfs_shapes,
    leakage,
    measure_fpfs,
    paired_bias,
    renderer_shear_response,
    resmooth,
)

SCALE, NPIX, FWHM = 0.141, 53, 0.5


def _population(n, applied_g1=0.0, applied_g2=0.0, noise_sd=0.0, seed=0, psf_g1=0.0):
    """A small ring-free population; identical galaxies and noise for any shear."""
    psf = galsim.Gaussian(fwhm=FWHM)
    if psf_g1:
        psf = psf.shear(g1=psf_g1, g2=0.0)
    psf_im = psf.drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array.astype(np.float64)
    gals, psfs = [], []
    for i in range(n):
        rng = np.random.RandomState(seed * 1000 + i)
        e1, e2 = rng.normal(0.0, 0.25, 2)
        while np.hypot(e1, e2) >= 0.8:
            e1, e2 = rng.normal(0.0, 0.25, 2)
        gal = galsim.Exponential(half_light_radius=0.5, flux=1.0e4).shear(g1=e1, g2=e2)
        gal = gal.shear(g1=applied_g1, g2=applied_g2)
        im = galsim.Convolve(gal, psf).drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array
        im = im.astype(np.float64)
        if noise_sd:
            # Seeded on i alone, so the +g and -g populations share a realisation.
            im = im + np.random.RandomState(90_000 + i).normal(0.0, noise_sd, im.shape)
        gals.append(im)
        psfs.append(psf_im)
    return np.asarray(gals), np.asarray(psfs)


# ----------------------------------------------------------------------
# FPFS against AnaCal itself
# ----------------------------------------------------------------------
def test_fpfs_recovers_a_known_shear():
    """End-to-end: applied shear in, m ~ 0 out. This is the real check."""
    n, shear = 64, 0.02
    plus = _population(n, applied_g1=shear, seed=1)
    minus = _population(n, applied_g1=-shear, seed=1)
    kw = dict(pixel_scale=SCALE, noise_variance=1.0e-6, sigma_shapelets=0.52)
    shapes_p = fpfs_shapes(measure_fpfs(*plus, **kw))
    shapes_m = fpfs_shapes(measure_fpfs(*minus, **kw))

    # FPFS's e is not a reduced shear: the response is what makes it one.
    assert 0.3 < np.mean(shapes_p.dedg[:, 0, 0]) < 1.5

    bias = paired_bias(shapes_p, shapes_m, shear, njac=8)
    # FPFS on correctly centred stamps is unbiased at the 1e-3 level. A loose
    # tolerance here would pass with the object one pixel off AnaCal's Fourier
    # origin, which costs about -15% in m.
    assert abs(bias.m) < 5e-3, bias.describe("fpfs")
    assert abs(bias.c) < 5e-2


def test_fpfs_rejects_an_even_stamp_it_cannot_centre():
    """An even stamp sits half a pixel off AnaCal's origin with no integer fix."""
    galaxy, psf = _population(2, seed=3)
    with pytest.raises(ValueError, match="even"):
        measure_fpfs(
            galaxy[:, :52, :52],
            psf[:, :52, :52],
            pixel_scale=SCALE,
            noise_variance=1e-6,
        )


def test_fpfs_padding_puts_the_object_on_anacals_origin():
    """The regression guard for a -15% multiplicative bias.

    AnaCal's Fourier origin for an even array is ``npix // 2``; a GalSim-drawn
    odd stamp has the object at ``(npix - 1) // 2``. Padding the *low* edge moves
    it onto the origin, padding the high edge leaves it one pixel short, and the
    difference is a large constant m that looks exactly like a real one.
    """
    n, shear = 24, 0.02
    kw = dict(pixel_scale=SCALE, noise_variance=1.0e-6, sigma_shapelets=0.52)
    good = paired_bias(
        fpfs_shapes(measure_fpfs(*_population(n, applied_g1=shear, seed=4), **kw)),
        fpfs_shapes(measure_fpfs(*_population(n, applied_g1=-shear, seed=4), **kw)),
        shear,
        njac=6,
    )
    assert abs(good.m) < 5e-3, good.describe("low-edge pad")

    def wrong_pad(images):
        return np.pad(images, ((0, 0), (0, 1), (0, 1)))

    import anacal

    def measure_high_pad(galaxy, psf):
        npix = galaxy.shape[1] + 1
        config = anacal.fpfs.FpfsConfig(npix=npix, sigma_shapelets1=0.52)
        detection = np.zeros(1, dtype=[("y", np.int32), ("x", np.int32)])
        detection["y"] = detection["x"] = galaxy.shape[1] // 2
        rows = [
            anacal.fpfs.process_image(
                fpfs_config=config,
                pixel_scale=SCALE,
                noise_variance=1e-6,
                mag_zero=30.0,
                gal_array=np.ascontiguousarray(g),
                psf_array=np.ascontiguousarray(p),
                detection=detection,
            )[0]
            for g, p in zip(wrong_pad(galaxy), wrong_pad(psf))
        ]
        return fpfs_shapes(np.asarray(rows))

    bad = paired_bias(
        measure_high_pad(*_population(n, applied_g1=shear, seed=4)),
        measure_high_pad(*_population(n, applied_g1=-shear, seed=4)),
        shear,
        njac=6,
    )
    assert bad.m < -0.05, "the misalignment this guards against no longer bites"


def test_fpfs_shapes_reports_missing_kernel_columns():
    catalogue = np.zeros(2, dtype=[("fpfs1_e1", "f8")])
    with pytest.raises(ValueError, match="missing"):
        fpfs_shapes(catalogue)
    with pytest.raises(ValueError, match="kernel must be"):
        fpfs_shapes(catalogue, kernel="fpfs9")


# ----------------------------------------------------------------------
# the shared response protocol
# ----------------------------------------------------------------------
def test_renderer_response_matches_the_analytic_fpfs_response():
    """The protocol every estimator is calibrated with, against a known answer.

    FPFS carries its own analytic de/dg, so differencing re-rendered scenes must
    reproduce it. If this fails, the ngmix and network columns are wrong in a
    way nothing else in the harness would reveal.
    """
    n = 32
    kw = dict(pixel_scale=SCALE, noise_variance=1.0e-6, sigma_shapelets=0.52)
    analytic = fpfs_shapes(measure_fpfs(*_population(n, seed=2), **kw))

    def render(d1, d2):
        return _population(n, applied_g1=d1, applied_g2=d2, seed=2)

    def measure(galaxy, psf):
        return fpfs_shapes(measure_fpfs(galaxy, psf, **kw)).e

    numeric = renderer_shear_response(render, measure, step=0.01)
    np.testing.assert_allclose(
        np.mean(numeric, axis=0), np.mean(analytic.dedg, axis=0), rtol=0.05, atol=2e-3
    )


def test_renderer_response_rejects_a_nonpositive_step():
    with pytest.raises(ValueError, match="step must be positive"):
        renderer_shear_response(lambda a, b: (None, None), lambda g, p: None, step=0.0)


# ----------------------------------------------------------------------
# bias estimation
# ----------------------------------------------------------------------
def _flat(e_value, r_value, n=64):
    return ShapeMeasurement(e=np.full((n, 2), e_value), dedg=np.repeat(r_value[None], n, axis=0))


def test_paired_bias_is_exact_for_a_noiseless_construction():
    shear, response = 0.01, 2.0
    plus = _flat(np.array([shear * response, 0.0]), response * np.eye(2))
    minus = _flat(np.array([-shear * response, 0.0]), response * np.eye(2))
    bias = paired_bias(plus, minus, shear, njac=8)
    assert bias.m == pytest.approx(0.0, abs=1e-12)
    assert bias.c == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(bias.response, response * np.eye(2))
    assert bias.n_used == 64


def test_paired_bias_recovers_an_injected_multiplicative_bias():
    shear, injected = 0.01, 0.05
    plus = _flat(np.array([shear * (1 + injected), 0.0]), np.eye(2))
    minus = _flat(np.array([-shear * (1 + injected), 0.0]), np.eye(2))
    assert paired_bias(plus, minus, shear, njac=8).m == pytest.approx(injected, rel=1e-9)


def test_paired_bias_measures_c_in_the_unsheared_component():
    """Matches the convention in research/shear_bias/m/helpers.jackknife_mc_v2."""
    shear, offset = 0.01, 3e-4
    plus = _flat(np.array([shear, offset]), np.eye(2))
    minus = _flat(np.array([-shear, offset]), np.eye(2))
    bias = paired_bias(plus, minus, shear, njac=8)
    assert bias.c == pytest.approx(offset, rel=1e-9)


def test_paired_bias_matches_the_legacy_jackknife():
    """The new estimator must reproduce the harness the paper's numbers came from."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("research/shear_bias/m").resolve()))
    from helpers import jackknife_mc_v2

    rng = np.random.RandomState(0)
    n, shear, njac = 200, 0.01, 10
    g_p = np.column_stack([rng.normal(shear, 0.2, n), rng.normal(0.0, 0.2, n)])
    g_m = np.column_stack([g_p[:, 0] - 2 * shear, g_p[:, 1]])
    r11 = rng.normal(1.0, 0.05, n)
    matrix = np.zeros((n, 2, 2))
    matrix[:, 0, 0] = matrix[:, 1, 1] = r11

    def _table(g, r):
        table = np.zeros(n, dtype=[("g_noshear", "f8", 2), ("r11", "f8")])
        table["g_noshear"] = g
        table["r11"] = r
        return table

    legacy = jackknife_mc_v2(_table(g_p, r11), _table(g_m, r11), shear, njac=njac)
    new = paired_bias(
        ShapeMeasurement(g_p, matrix), ShapeMeasurement(g_m, matrix), shear, njac=njac
    )
    assert new.m == pytest.approx(legacy[0], rel=1e-10)
    assert new.c == pytest.approx(legacy[1], rel=1e-10)
    assert new.m_err == pytest.approx(legacy[3], rel=1e-8)
    assert new.c_err == pytest.approx(legacy[5], rel=1e-8)


def test_paired_bias_drops_a_row_that_failed_in_either_population():
    n = 32
    e = np.zeros((n, 2))
    e[:, 0] = 0.01
    response = np.repeat(np.eye(2)[None], n, axis=0)
    flags = np.zeros(n, dtype=bool)
    flags[:4] = True
    plus = ShapeMeasurement(e, response, flags=flags)
    minus = ShapeMeasurement(-e, response)
    assert paired_bias(plus, minus, 0.01, njac=4).n_used == n - 4


def test_paired_bias_bins_by_a_supplied_scalar():
    n = 200
    e = np.zeros((n, 2))
    e[:, 0] = 0.01
    response = np.repeat(np.eye(2)[None], n, axis=0)
    bias = paired_bias(
        ShapeMeasurement(e, response),
        ShapeMeasurement(-e, response),
        0.01,
        njac=5,
        bin_values=np.arange(n),
        nbins=4,
    )
    assert bias.bins is not None
    assert len(bias.bins["m"]) == 4
    assert bias.bins["count"].sum() == n
    np.testing.assert_allclose(bias.bins["m"], 0.0, atol=1e-12)


def test_paired_bias_rejects_mismatched_populations():
    a = _flat(np.array([0.01, 0.0]), np.eye(2), n=8)
    b = _flat(np.array([-0.01, 0.0]), np.eye(2), n=9)
    with pytest.raises(ValueError, match="differ in size"):
        paired_bias(a, b, 0.01)
    with pytest.raises(ValueError, match="applied_shear must be nonzero"):
        paired_bias(a, a, 0.0)


def test_shape_measurement_validates_its_shapes():
    with pytest.raises(ValueError, match=r"e must be \(N, 2\)"):
        ShapeMeasurement(np.zeros((4, 3)), np.zeros((4, 2, 2)))
    with pytest.raises(ValueError, match="dedg must be"):
        ShapeMeasurement(np.zeros((4, 2)), np.zeros((4, 2)))


def test_leakage_reports_mean_shape_and_psf_response():
    n = 64
    e = np.tile([0.02, -0.01], (n, 1))
    psf_response = np.repeat((0.1 * np.eye(2))[None], n, axis=0)
    stats = leakage(e, psf_response, njac=8)
    np.testing.assert_allclose(stats["mean_e"], [0.02, -0.01])
    np.testing.assert_allclose(stats["psf_response"], 0.1 * np.eye(2))
    np.testing.assert_allclose(stats["mean_e_err"], 0.0, atol=1e-12)
    assert int(stats["n_used"]) == n


# ----------------------------------------------------------------------
# the AnaCal linear observable (Lin et al. 2026, Eq. 3)
# ----------------------------------------------------------------------
def test_resmooth_removes_the_psf():
    """f_h is PSF-homogenised: that is the whole point, and the whole caveat.

    The same galaxy behind a round PSF and behind a sheared one must produce the
    same re-smoothed image. It is why the D4CNN of Lin et al. needs no PSF
    branch -- and why comparing its PSF leakage against a network fed the raw
    convolved stamp compares input pipelines, not architectures.
    """
    gal = galsim.Exponential(half_light_radius=0.5, flux=1.0e4).shear(g1=0.2, g2=0.1)
    rendered = []
    for psf_g1 in (0.0, 0.08):
        psf = galsim.Gaussian(fwhm=FWHM).shear(g1=psf_g1, g2=0.0)
        image = galsim.Convolve(gal, psf).drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array
        psf_image = psf.drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array
        rendered.append(resmooth(image[None], psf_image[None], pixel_scale=SCALE, sigma=0.35)[0])
    round_psf, sheared_psf = rendered
    assert np.abs(round_psf - sheared_psf).max() / np.abs(round_psf).max() < 1e-4


def test_resmooth_matches_a_direct_render_of_the_smoothing_kernel():
    """f_h against independent ground truth, not against a hand-rolled moment.

    With no noise term, Eq. 3 reduces to "the galaxy convolved with the Gaussian
    re-smoothing kernel and nothing else", which GalSim can render directly. The
    residual is finite-stamp FFT aliasing in the deconvolution.

    This is also the guard on the centring: dividing by ``p(k)`` removes the
    PSF's centring phase, so the raw quotient lands on pixel (0, 0) and has to be
    rolled back. Without that roll the peak is in the corner and any downstream
    measurement is measuring a wrapped image.
    """
    psf = galsim.Gaussian(fwhm=FWHM)
    psf_image = psf.drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array
    galaxy = galsim.Exponential(half_light_radius=0.5, flux=1.0e4).shear(g1=0.3, g2=0.0)
    observed = galsim.Convolve(galaxy, psf).drawImage(nx=NPIX, ny=NPIX, scale=SCALE).array

    sigma = 0.35
    f_h = resmooth(observed[None], psf_image[None], pixel_scale=SCALE, sigma=sigma)[0]
    direct = (
        galsim.Convolve(galaxy, galsim.Gaussian(sigma=sigma))
        .drawImage(nx=NPIX, ny=NPIX, scale=SCALE)
        .array
    )

    assert np.unravel_index(np.argmax(f_h), f_h.shape) == (NPIX // 2, NPIX // 2)
    assert np.abs(f_h - direct).max() / direct.max() < 0.02


def test_resmooth_validates_its_arguments():
    galaxy, psf = _population(2, seed=6)
    with pytest.raises(ValueError, match="sigma must be positive"):
        resmooth(galaxy, psf, pixel_scale=SCALE, sigma=0.0)
    with pytest.raises(ValueError, match="share an"):
        resmooth(galaxy, psf[:1], pixel_scale=SCALE, sigma=0.35)


# ----------------------------------------------------------------------
# bias conventions
# ----------------------------------------------------------------------
def test_c_conventions_are_different_numbers():
    """Lin et al. Eq. 23 divides by the response and uses the sheared component."""
    shear, response, offset = 0.01, 2.0, 4e-4
    plus = _flat(np.array([shear * response + offset, 0.0]), response * np.eye(2))
    minus = _flat(np.array([-shear * response + offset, 0.0]), response * np.eye(2))
    house = paired_bias(plus, minus, shear, njac=8)
    paper = paired_bias(plus, minus, shear, njac=8, c_convention="lin2026")
    assert house.c == pytest.approx(0.0, abs=1e-12)  # nothing in the g2 column
    assert paper.c == pytest.approx(offset / response, rel=1e-9)
    assert house.m == pytest.approx(paper.m, rel=1e-12)  # m is unaffected
    with pytest.raises(ValueError, match="c_convention"):
        paired_bias(plus, minus, shear, c_convention="other")


def test_bootstrap_and_jackknife_errors_agree_in_scale():
    rng = np.random.RandomState(1)
    n, shear = 400, 0.01
    # Per-object response scatter is what makes the pair difference vary; with a
    # constant response the paired estimator is exact and both errors are zero.
    r11 = rng.normal(1.0, 0.1, n)
    intrinsic = np.column_stack([rng.normal(0.0, 0.3, n), rng.normal(0.0, 0.3, n)])
    e = intrinsic + np.column_stack([r11 * shear, np.zeros(n)])
    response = np.zeros((n, 2, 2))
    response[:, 0, 0] = response[:, 1, 1] = r11
    plus = ShapeMeasurement(e, response)
    minus = ShapeMeasurement(
        np.column_stack([intrinsic[:, 0] - r11 * shear, intrinsic[:, 1]]), response
    )
    jack = paired_bias(plus, minus, shear, njac=20)
    boot = paired_bias(plus, minus, shear, njac=20, resample="bootstrap")
    assert jack.m == pytest.approx(boot.m, rel=1e-12)
    assert 0.4 < boot.m_err / jack.m_err < 2.5, (jack.m_err, boot.m_err)
    with pytest.raises(ValueError, match="resample"):
        paired_bias(plus, minus, shear, resample="jacknife")
