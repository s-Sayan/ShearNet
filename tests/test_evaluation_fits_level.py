"""The evaluation writer honours ``catalog_level`` end to end.

:mod:`test_catalog_schema` covers the policy in isolation. These tests go
through ``run.py``'s own ``_write_evaluation_fits``, which is where a wiring
mistake would actually cost a run: a level that is accepted but ignored writes
the gigabyte anyway, and a level that is validated too late wastes the two hours
of measurement that precede the write.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("astropy")

RUN_DIR = Path(__file__).resolve().parents[1] / "research" / "shear_bias"
sys.path.insert(0, str(RUN_DIR))


@pytest.fixture(scope="module")
def run_module():
    """``run.py`` imports the whole benchmarking stack; skip if it is absent."""
    return pytest.importorskip("run")


class _Section(dict):
    """Stands in for the ``eval.evaluate`` config mapping."""


class _Benchmark:
    """The two ``get`` keys ``_write_evaluation_fits`` reads from the config."""

    def __init__(self, root):
        self._root = str(root)

    def get(self, key, default=None):
        if key == "paths.root":
            return self._root
        return default


def _columns(n=64, seed=0):
    rng = np.random.default_rng(seed)
    col = {
        "gpsf": rng.normal(0, 0.042, (n, 2)),
        "Tpsf": rng.normal(0.35, 0.02, n),
        "s2n": rng.lognormal(3.0, 0.6, n),
        "hlr_th": rng.lognormal(-0.7, 0.4, n),
        "flux_th": rng.lognormal(9.0, 0.8, n),
    }
    for station in ("", "_r90"):
        col[f"g_th{station}"] = rng.normal(0.01, 0.25, (n, 2))
        col[f"e_shearnet{station}"] = rng.normal(0, 0.25, (n, 2))
        col[f"R_shearnet_sim{station}"] = rng.normal(0.9, 0.1, (n, 2, 2))
        col[f"R_shearnet_metacal{station}"] = rng.normal(0.9, 0.1, (n, 2, 2))
    return col


def _write(run_module, tmp_path, level):
    tables = {0: {"plus": _columns(seed=1), "minus": _columns(seed=2)}}
    section = _Section(output="evaluation.fits")
    if level is not None:
        section["catalog_level"] = level
    return run_module._write_evaluation_fits(
        _Benchmark(tmp_path), section, tables, _columns(seed=3),
        {"seed": 42, "shear_true": 0.01},
    )


def _hdu_names(path):
    from astropy.io import fits

    with fits.open(path) as hdul:
        return [hdu.name for hdu in hdul[1:]]


def test_summary_level_omits_the_per_object_tables(run_module, tmp_path):
    names = _hdu_names(_write(run_module, tmp_path, "summary"))
    assert "TAB_P" not in names and "LEAKAGE" not in names
    assert {"SUMMARY", "BINNED", "LEAKSUM"} <= set(names)


def test_paper_level_keeps_the_per_object_tables(run_module, tmp_path):
    names = _hdu_names(_write(run_module, tmp_path, "paper"))
    assert {"TAB_P", "TAB_M", "LEAKAGE"} <= set(names)


def test_the_level_reaches_the_file_rather_than_being_ignored(run_module, tmp_path):
    """The failure this guards is a config key that parses and does nothing."""
    from astropy.io import fits

    for level in ("summary", "paper", "full"):
        path = _write(run_module, tmp_path / level, level)
        with fits.open(path) as hdul:
            assert hdul[0].header["CATLEVEL"] == level


def test_paper_is_smaller_than_full_on_the_same_measurement(run_module, tmp_path):
    full = _write(run_module, tmp_path / "a", "full").stat().st_size
    paper = _write(run_module, tmp_path / "b", "paper").stat().st_size
    summary = _write(run_module, tmp_path / "c", "summary").stat().st_size
    assert summary < paper < full


def test_unset_level_still_writes_the_per_object_tables(run_module, tmp_path):
    """Defaulting must not silently drop rows for anyone who has not read this."""
    names = _hdu_names(_write(run_module, tmp_path, None))
    assert "TAB_P" in names


def test_paper_level_stores_float32(run_module, tmp_path):
    from astropy.io import fits

    path = _write(run_module, tmp_path, "paper")
    with fits.open(path) as hdul:
        data = hdul["TAB_P"].data
        assert data["e_shearnet"].dtype.itemsize == 4


def test_a_bad_level_is_rejected(run_module, tmp_path):
    with pytest.raises(ValueError, match="catalog_level"):
        _write(run_module, tmp_path, "papper")


def test_a_bad_level_is_rejected_before_any_measurement(run_module):
    """``_run_evaluation`` validates up front, not at write time.

    Failing at the write would throw away the whole measurement pass.
    """
    import inspect

    source = inspect.getsource(run_module._run_evaluation)
    body = source[: source.index("_measure_shear_pair")]
    assert "resolve_level" in body


def test_paper_level_reproduces_summary_to_float32(run_module, tmp_path):
    """What the default level costs, stated as a number rather than assumed.

    `full` lets the per-object columns reproduce SUMMARY bit-for-bit, which is
    the contract test_shear_bias_run.py asserts at rel=1e-9. `paper` stores
    float32, so the same recomputation agrees to ~1e-7 instead.

    That is five orders below the 1e-4 jackknife error on m and so costs the
    science nothing -- but it is a real difference between the levels, and it
    should be visible here rather than discovered by someone whose exact
    comparison stopped working.
    """
    from astropy.io import fits

    exact = _write(run_module, tmp_path / "f", "full")
    stored = _write(run_module, tmp_path / "p", "paper")

    def m_of(path):
        with fits.open(path) as hdul:
            plus, minus = hdul["TAB_P"].data, hdul["TAB_M"].data
            num = 0.5 * (np.asarray(plus["e_shearnet"], float)[:, 0]
                         - np.asarray(minus["e_shearnet"], float)[:, 0])
            den = 0.5 * (np.asarray(plus["R_shearnet_sim"], float)[:, 0, 0]
                         + np.asarray(minus["R_shearnet_sim"], float)[:, 0, 0])
            return num.mean() / den.mean()

    a, b = m_of(exact), m_of(stored)
    assert a == pytest.approx(b, rel=1e-6), (a, b)
    # And tight enough that it could never move a reported m1 at the 1e-3 level.
    assert abs(a - b) < 1e-6 * abs(a)
