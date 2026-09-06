"""The evaluation catalog schema policy: what it keeps, drops, and preserves.

The point of these tests is that slimming must never change an answer. A level
that silently dropped a column the leakage fit needs, or that rounded a shape
below the precision the jackknife resolves, would produce a smaller file and a
different paper.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "shear_bias"))

from catalog import (CATALOG_LEVELS, DEFAULT_LEVEL, PAPER_KEEP_PREFIXES,  # noqa: E402
                     estimate_row_bytes, meta_table, resolve_level, slim_columns)

STATIONS = ("", "_r45", "_r90", "_r135")


def _pair_columns(n=512, seed=3):
    """A pair table with the production column names, including ring stations."""
    rng = np.random.default_rng(seed)
    col = {
        "gpsf": rng.normal(0, 0.042, (n, 2)),
        "Tpsf": rng.normal(0.35, 0.02, n),
        "s2n": rng.lognormal(3.0, 0.6, n),
        "hlr_th": rng.lognormal(-0.7, 0.4, n),
        "flux_th": rng.lognormal(9.0, 0.8, n),
        "s2n_ngmix": rng.lognormal(3.0, 0.6, n),
        "T_ngmix": rng.normal(0.5, 0.1, n),
        "hlr_shearnet": rng.lognormal(-0.7, 0.4, n),
    }
    for s in STATIONS:
        col[f"g_th{s}"] = rng.normal(0.01, 0.25, (n, 2))
        for est in ("ngmix", "shearnet"):
            col[f"e_{est}{s}"] = rng.normal(0, 0.3, (n, 2))
            col[f"e_{est}_uncorrected{s}"] = rng.normal(0, 0.3, (n, 2))
            col[f"e_{est}_metacal_raw{s}"] = rng.normal(0, 0.3, (n, 2))
            col[f"e_{est}_metacal_corrected{s}"] = rng.normal(0, 0.3, (n, 2))
            col[f"R_{est}_sim{s}"] = rng.normal(0.9, 0.1, (n, 2, 2))
            col[f"R_{est}_metacal{s}"] = rng.normal(0.64, 0.1, (n, 2, 2))
            col[f"Rgamma_{est}_metacal{s}"] = rng.normal(0.64, 0.1, (n, 2, 2))
            col[f"Rpsf_{est}_sim{s}"] = rng.normal(0, 0.1, (n, 2, 2))
            col[f"Rpsf_{est}_metacal{s}"] = rng.normal(0, 0.1, (n, 2, 2))
            col[f"Rbarpsf_{est}_metacal"] = np.full(n, 0.2727)
            col[f"flag_{est}{s}"] = rng.integers(0, 2, n).astype(np.int32)
    return col


# ----------------------------------------------------------------------
# what each level keeps
# ----------------------------------------------------------------------
def test_summary_level_writes_no_rows():
    kept, report = slim_columns(_pair_columns(), "summary")
    assert kept == {}
    assert report.after_bytes == 0
    assert report.dropped, "every column should be accounted for as dropped"


def test_paper_level_keeps_every_station_of_the_scored_pair():
    """m and c are ring averages, so no station may go missing."""
    kept, _ = slim_columns(_pair_columns(), "paper")
    for s in STATIONS:
        assert f"e_shearnet{s}" in kept
        assert f"R_shearnet_sim{s}" in kept
        assert f"e_ngmix_metacal_corrected{s}" in kept
        assert f"R_ngmix_metacal{s}" in kept
        assert f"Rgamma_ngmix_metacal{s}" in kept
        assert f"g_th{s}" in kept


def test_paper_level_keeps_the_direct_psf_response_for_both_estimators():
    """The right-hand panel of the response-vs-S/N figure needs both.

    R^PSF is measured the direct way for every estimator, so it is not part of
    the crossed pair that gets dropped -- an easy thing to lose to a pattern
    that keys on the correction token alone.
    """
    kept, _ = slim_columns(_pair_columns(), "paper")
    for s in STATIONS:
        assert f"Rpsf_ngmix_sim{s}" in kept
        assert f"Rpsf_shearnet_sim{s}" in kept


def test_paper_level_keeps_the_leakage_regressors():
    """alpha and beta are fitted jointly against e_PSF and T_PSF."""
    kept, _ = slim_columns(_pair_columns(), "paper")
    assert "gpsf" in kept and "Tpsf" in kept


def test_paper_level_keeps_the_binning_variables():
    """The S/N and size trends bin on these; without them the figure is lost."""
    kept, _ = slim_columns(_pair_columns(), "paper")
    assert "s2n" in kept and "hlr_th" in kept


def test_paper_level_drops_only_the_unscored_correction():
    """ngmix is scored through metacal, ShearNet through sim; the crosses go."""
    kept, report = slim_columns(_pair_columns(), "paper")
    for s in STATIONS:
        assert f"R_ngmix_sim{s}" not in kept
        assert f"R_shearnet_metacal{s}" not in kept
        assert f"Rgamma_shearnet_metacal{s}" not in kept
    assert all(report.dropped[f"R_ngmix_sim{s}"] for s in STATIONS)


def test_full_level_keeps_every_measured_column():
    columns = _pair_columns()
    kept, report = slim_columns(columns, "full")
    # Only constants and exact duplicates may vanish at 'full'.
    assert set(kept) | set(report.constants) | set(report.aliases) == set(columns)


def test_every_response_family_the_harness_writes_is_matched():
    """A single ``R_*`` pattern matches none of Rgamma_/Rpsf_/Rbarpsf_.

    This is the mistake that would silently drop the metacal shear response and
    the PSF response from every file, so it is pinned rather than trusted.
    """
    import fnmatch

    for name in ("R_shearnet_sim", "Rgamma_ngmix_metacal", "Rpsf_ngmix_sim"):
        assert any(fnmatch.fnmatchcase(name, p) for p in PAPER_KEEP_PREFIXES), name


def test_full_is_larger_than_paper_is_larger_than_summary():
    columns = _pair_columns()
    sizes = [slim_columns(columns, level)[1].after_bytes for level in CATALOG_LEVELS]
    assert sizes == sorted(sizes), f"levels not monotonic in size: {sizes}"


# ----------------------------------------------------------------------
# what it does to the values
# ----------------------------------------------------------------------
def test_values_survive_to_float32_precision():
    columns = _pair_columns()
    kept, _ = slim_columns(columns, "paper")
    for name, array in kept.items():
        if np.issubdtype(array.dtype, np.floating):
            np.testing.assert_allclose(array, columns[name], rtol=1e-6, atol=0)


def test_ensemble_mean_is_unchanged_far_below_the_jackknife_error():
    """The quantity that actually matters: <e> over the population.

    Shape noise on 5x10^5 objects gives sigma_<e> ~ 3x10^-4. The float32 storage
    error on the mean must sit orders below that or the saving is not free.
    """
    columns = _pair_columns(n=200000, seed=11)
    kept, _ = slim_columns(columns, "paper")
    exact = columns["e_shearnet"].mean(axis=0)
    stored = kept["e_shearnet"].astype(np.float64).mean(axis=0)
    jackknife_sigma = columns["e_shearnet"].std(axis=0) / np.sqrt(len(columns["e_shearnet"]))
    assert np.all(np.abs(stored - exact) < 1e-3 * jackknife_sigma)


def test_response_ratio_is_unchanged():
    """m is a ratio of means; check the ratio, not just the numerator."""
    columns = _pair_columns(n=50000, seed=5)
    kept, _ = slim_columns(columns, "paper")
    exact = columns["e_shearnet"][:, 0].mean() / columns["R_shearnet_sim"][:, 0, 0].mean()
    stored = (kept["e_shearnet"][:, 0].astype(np.float64).mean()
              / kept["R_shearnet_sim"][:, 0, 0].astype(np.float64).mean())
    assert abs(stored - exact) < 1e-6 * abs(exact)


def test_downcast_applies_to_big_endian_columns_read_from_disk():
    """A column read back from FITS is '>f8'; `dtype == np.float64` is False.

    This is not hypothetical -- it is the only dtype shrink_fits ever sees.
    """
    columns = {"e_shearnet": np.zeros((16, 2), dtype=">f8"),
               "R_shearnet_sim": np.ones((16, 2, 2), dtype=">f8")}
    kept, _ = slim_columns(columns, "paper")
    assert all(a.dtype.itemsize == 4 for a in kept.values()), {
        k: str(v.dtype) for k, v in kept.items()
    }


# ----------------------------------------------------------------------
# constants and aliases
# ----------------------------------------------------------------------
def test_broadcast_scalar_becomes_a_recorded_constant():
    """`Rbar_psf_*` is one number repeated once per object."""
    columns = dict(_pair_columns())
    columns["Rbar_psf_ngmix"] = np.full(512, 0.2727)
    kept, report = slim_columns(columns, "full")
    assert "Rbar_psf_ngmix" not in kept
    assert report.constants["Rbar_psf_ngmix"] == pytest.approx(0.2727)


def test_duplicate_column_becomes_a_recorded_alias():
    """`e_<est>_raw` is `e_<est>` whenever no correction was applied."""
    e = np.random.default_rng(1).normal(0, 0.3, (512, 2))
    columns = {"e_ngmix": e, "e_ngmix_raw": e.copy()}
    kept, report = slim_columns(columns, "full")
    assert "e_ngmix" in kept and "e_ngmix_raw" not in kept
    assert report.aliases["e_ngmix_raw"] == "e_ngmix"


def test_distinct_columns_are_not_aliased():
    rng = np.random.default_rng(2)
    columns = {"e_ngmix": rng.normal(0, 0.3, (512, 2)),
               "e_shearnet": rng.normal(0, 0.3, (512, 2))}
    kept, report = slim_columns(columns, "full")
    assert set(kept) == set(columns)
    assert report.aliases == {}


def test_meta_table_round_trips_the_constants():
    columns = {"Rbar_psf_ngmix": np.full(64, 0.2727),
               "e_ngmix": np.random.default_rng(4).normal(0, 0.3, (64, 2))}
    _, report = slim_columns(columns, "full")
    table = meta_table([("LEAKAGE", report)])
    row = table[table["column"] == "Rbar_psf_ngmix"][0]
    assert row["kind"] == "constant"
    assert float(row["value"]) == pytest.approx(0.2727)


def test_meta_table_is_writable_when_nothing_was_dropped():
    """An empty SLIMMETA still needs a schema, or the HDU cannot be built."""
    from astropy.io import fits

    _, report = slim_columns({"e_ngmix": np.arange(8.0).reshape(4, 2)}, "full")
    hdu = fits.BinTableHDU(meta_table([("TAB_P", report)]), name="SLIMMETA")
    assert hdu.data is not None


# ----------------------------------------------------------------------
# the level itself
# ----------------------------------------------------------------------
def test_unset_level_defaults_rather_than_failing():
    assert resolve_level(None) == DEFAULT_LEVEL


@pytest.mark.parametrize("level", CATALOG_LEVELS)
def test_every_declared_level_resolves(level):
    assert resolve_level(level) == level
    assert resolve_level(level.upper()) == level


def test_a_typo_raises_instead_of_writing_a_gigabyte():
    with pytest.raises(ValueError, match="catalog_level"):
        resolve_level("papper")


def test_row_bytes_estimate_matches_the_columns():
    columns = {"a": np.zeros((100, 2), dtype=np.float32),
               "b": np.zeros(100, dtype=np.float32)}
    assert estimate_row_bytes(columns) == pytest.approx(12.0)
