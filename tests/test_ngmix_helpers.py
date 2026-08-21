"""Unit tests for the pure FFT/sampling helpers in methods.ngmix."""

import numpy as np
import pytest

pytest.importorskip("ngmix")

from shearnet.methods.ngmix import (  # noqa: E402
    convolve2d,
    fft_ifft,
    fourier_transform,
    inverse_fourier_transform,
    sample_half_gaussian,
)


def test_fourier_roundtrip_recovers_image():
    rng = np.random.RandomState(0)
    img = np.abs(rng.rand(16, 16))
    recovered = inverse_fourier_transform(fourier_transform(img))
    assert recovered.shape == img.shape
    assert np.allclose(recovered, img, atol=1e-8)


def test_fft_ifft_roundtrip():
    rng = np.random.RandomState(1)
    img = np.abs(rng.rand(8, 8))
    assert np.allclose(fft_ifft(img), img, atol=1e-8)


def test_convolve2d_preserves_shape():
    img = np.zeros((21, 21))
    img[10, 10] = 1.0
    psf = np.ones((5, 5)) / 25.0
    out = convolve2d(img, psf)
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_sample_half_gaussian_count_and_floor():
    s = sample_half_gaussian(size=200, sigma=0.5, seed=0)
    assert len(s) == 200
    assert (s > 0.14).all()


def test_sample_half_gaussian_reproducible():
    a = sample_half_gaussian(size=50, sigma=0.5, seed=7)
    b = sample_half_gaussian(size=50, sigma=0.5, seed=7)
    assert np.allclose(a, b)


# ----------------------------------------------------------------------
# metacal: worker count and what comes back from the pool
# ----------------------------------------------------------------------
def _metacal_obs(n, npix=53, scale=0.141, seed=0):
    """A few real stamps, packaged the way the benchmark packages them."""
    galsim = pytest.importorskip("galsim")
    import ngmix

    rng = np.random.RandomState(seed)
    centre = (npix - 1) / 2.0
    jacobian = ngmix.DiagonalJacobian(row=centre, col=centre, scale=scale)
    psf = galsim.Gaussian(fwhm=0.5)
    psf_image = psf.drawImage(nx=npix, ny=npix, scale=scale).array.astype(float)
    out = []
    for i in range(n):
        gal = galsim.Exponential(half_light_radius=0.5, flux=12258.97).shear(
            g1=0.02 * (-1) ** i, g2=-0.01
        )
        image = galsim.Convolve(gal, psf).drawImage(nx=npix, ny=npix, scale=scale).array.astype(
            float
        )
        image = image + rng.normal(scale=5.0, size=image.shape)
        psf_obs = ngmix.Observation(
            np.ascontiguousarray(psf_image),
            weight=np.full(psf_image.shape, 1e6),
            jacobian=jacobian,
        )
        out.append(
            ngmix.Observation(
                np.ascontiguousarray(image),
                weight=np.full(image.shape, 1.0 / 5.0**2),
                jacobian=jacobian,
                psf=psf_obs,
            )
        )
    return out


def test_resolve_nproc_reads_the_allocation_not_the_node(monkeypatch):
    """os.cpu_count() is the node's cores; a shared node is not ours to fill."""
    from shearnet.parallel import resolve_nproc

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert resolve_nproc() == 1  # off-cluster: serial, not os.cpu_count()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "18")
    assert resolve_nproc() == 18
    assert resolve_nproc(nproc=4) == 4  # explicit wins
    assert resolve_nproc(n_tasks=3) == 3  # never more workers than work
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    assert resolve_nproc() == 1  # a malformed allocation is not a crash


@pytest.mark.slow
@pytest.mark.parametrize("nproc", [1, 2])
def test_metacal_returns_only_structs(nproc):
    """The pool must not ship back the sheared observations.

    ``boot.go`` produces a ``resdict`` (~4.3 MB/object) and an ``obsdict``
    (~3.6 MB/object) beside the 1.4 kB struct that is the answer. Returning
    either one is what makes a 200k-object population impossible; the second
    element of the tuple is empty unless explicitly asked for.
    """
    from shearnet.methods.ngmix import _get_priors, mp_fit_one_single

    obs = _metacal_obs(4)
    data_list, resdict_list = mp_fit_one_single(
        obs, _get_priors(3), np.random.RandomState(3), nproc=nproc
    )
    assert len(data_list) == 4
    assert resdict_list == [], "the pool shipped back the sheared observations"
    for rows in data_list:
        types = {str(row["shear_type"]) for row in rows}
        assert {"noshear", "1p", "1m", "2p", "2m"}.issubset(types)
        assert rows.nbytes < 4096, rows.nbytes  # a struct, not an image


@pytest.mark.slow
def test_metacal_response_agrees_between_serial_and_pooled():
    """Parallelising changes only the random guesses, not the estimator.

    The RNG stream advances per worker, so the two are not bit-identical; the
    converged responses must still agree, or the pool is fitting something else.
    """
    from shearnet.methods.ngmix import _get_priors, mp_fit_one_single

    obs = _metacal_obs(6, seed=1)

    def response(nproc):
        data_list, _ = mp_fit_one_single(
            obs, _get_priors(7), np.random.RandomState(7), nproc=nproc,
            mcal_pars={"psf": "dilate", "mcal_shear": 0.01},
        )
        out = []
        for rows in data_list:
            by_type = {str(row["shear_type"]): row for row in rows}
            if any(by_type[k]["flags"] != 0 for k in ("1p", "1m")):
                continue
            out.append((by_type["1p"]["g"][0] - by_type["1m"]["g"][0]) / 0.02)
        return np.asarray(out)

    serial, pooled = response(1), response(2)
    assert serial.size and serial.size == pooled.size
    assert np.allclose(serial, pooled, rtol=1e-3, atol=1e-4), (serial, pooled)
