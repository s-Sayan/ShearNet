"""The train/eval split must not put one galaxy on both sides.

``notebooks/detection_catalog_split.ipynb`` tiles each source galaxy
``N_AUGMENTS`` times with fresh position angles and only then permutes, so the
rotations of a single galaxy land in both subsets: the network trains on some
orientations of a galaxy and is evaluated on the others, with identical
half-light radius, flux and axis ratio. These pin the corrected behaviour, and
the leakage test is written against the observable property -- the tiled
``(Q, HLR, FLUX)`` triple -- rather than against an internal index, so it keeps
working if the implementation changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "shear_bias"))

galsim = pytest.importorskip("galsim")


def _parent(path: Path, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns([
            fits.Column(name="c10_sersic_fit_q", format="D",
                        array=rng.uniform(0.2, 1.0, n)),
            fits.Column(name="c10_sersic_fit_phi", format="D",
                        array=rng.uniform(0, np.pi, n)),
            fits.Column(name="c10_sersic_fit_hlr", format="D",
                        array=rng.lognormal(2.0, 0.8, n)),
            fits.Column(name="crates_b", format="D",
                        array=rng.lognormal(-3, 0.6, n)),
        ]),
    ]).writeto(path, overwrite=True)
    return path


def _galaxy_keys(data):
    """The orientation-independent triple identifying a source galaxy."""
    return set(zip(np.round(data["Q"], 12),
                   np.round(data["HLR"], 12),
                   np.round(data["FLUX"], 12)))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from build_catalogs import build

    directory = tmp_path_factory.mktemp("catalogs")
    parent = _parent(directory / "parent.fits")
    written = build(parent, directory, n_augments=10, min_resolution=1.0,
                    psf_fwhm=0.5, seed=7)
    return {name: fits.open(path)[1] for name, path in written.items()}


def test_no_source_galaxy_appears_in_both_subsets(built):
    train, evaluation = _galaxy_keys(built["train"].data), _galaxy_keys(built["eval"].data)
    assert train and evaluation
    assert not (train & evaluation), (
        f"{len(train & evaluation)} source galaxies appear in both subsets; "
        "the split ran on augmented rows rather than on galaxies"
    )


def test_every_galaxy_keeps_its_full_multiplicity(built):
    """Filtering the parent, not the finished catalog, keeps all copies.

    Cutting the augmented catalog afterwards would leave galaxies with fewer
    than N_AUGMENTS orientations only by accident of the split; cutting the
    parent first means every survivor is augmented in full.
    """
    for name, hdu in built.items():
        galaxies = len(_galaxy_keys(hdu.data))
        assert len(hdu.data) == galaxies * hdu.header["NAUG"], name


def test_size_floor_is_applied_and_recorded(built):
    for name, hdu in built.items():
        floor = hdu.header["HLRFLOOR"]
        assert floor == pytest.approx(0.25)  # 1.0 PSF half-width at FWHM 0.5"
        assert np.all(np.asarray(hdu.data["HLR"], dtype=float) >= floor), name
        assert hdu.header["GRPSPLIT"] is True


def test_schema_matches_what_the_loader_indexes(built):
    """dataset._load_cosmos_cat reads these four by name."""
    for name, hdu in built.items():
        for column in ("G1", "G2", "HLR", "FLUX"):
            assert column in hdu.data.columns.names, f"{column} missing from {name}"


def test_orientations_are_redrawn_not_copied(built):
    """Augmentation must vary the angle; tiling it would add nothing at all."""
    data = built["train"].data
    by_galaxy = {}
    for row in range(len(data)):
        key = (round(float(data["Q"][row]), 12), round(float(data["HLR"][row]), 12),
               round(float(data["FLUX"][row]), 12))
        by_galaxy.setdefault(key, []).append(float(data["PHI"][row]))
    angles = next(iter(by_galaxy.values()))
    assert len(set(angles)) == len(angles)
