"""Build the train/eval catalogs from the detection truth catalog.

This is ``notebooks/detection_catalog_split.ipynb`` as a script, with the size
floor applied where it belongs and one defect fixed.

WHY NOT CUT THE FINISHED CATALOG
--------------------------------
``cut_catalog.py`` filters ``cosmos_catalog_eval.fits`` after the fact, which
costs 79% of the rows at a 0.375" floor and leaves too few to render 200000
samples. That is the wrong place to cut. The notebook tiles every source galaxy
``N_AUGMENTS`` times with fresh position angles, so the eval catalog's 357984
rows are 89496 distinct galaxies repeated ten times over. Filtering afterwards
throws away whole galaxies *and* their copies; filtering the parent first and
then augmenting keeps every surviving galaxy at full multiplicity.

THE DEFECT: THE SPLIT LEAKS ACROSS TRAIN AND EVAL
-------------------------------------------------
The notebook augments first and splits second::

    shuffled_idx = rng.permutation(n_total)      # n_total = AUGMENTED rows

``n_total`` counts augmented rows, so the ten rotations of one source galaxy
are scattered across both sides. The network trains on six orientations of a
galaxy and is then evaluated on the other four -- same half-light radius, same
flux, same axis ratio, only the position angle differs. That is not a held-out
set, and it is the first thing a referee will look for in an ML shear paper.

This script splits on the SOURCE galaxy, before augmentation, so no galaxy
appears on both sides in any orientation.

WHY RAISING N_AUGMENTS IS NOT THE WAY BACK TO 200000 ROWS
---------------------------------------------------------
It is tempting, since 27 rotations per galaxy would restore the row count. But
the benchmark already runs a ring test -- every object is measured at 0, 45, 90
and 135 degrees and averaged -- so extra catalog orientations are largely
redundant with the ring. They add rows, not independent information about the
size and flux distribution, and the error on m would shrink as though they had.
Rows above the natural multiplicity buy precision that is not there.

    python research/shear_bias/build_catalogs.py \\
        --parent simulated_detection_truth_backcast_unit_fiducial_dec04.fits \\
        --out-dir . --min-resolution 1.0 --psf-fwhm 0.5

    python research/shear_bias/build_catalogs.py --parent ... --report-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

#: The notebook's conversions, unchanged -- this is the same catalog, built the
#: same way, so a run on it stays comparable to everything already measured.
PIXEL_SCALE = 0.03
FLUX_FACTOR = 300 * 36 / 0.343
HLR_CEILING = 1.0


def read_parent(path: Path):
    """``(q, phi, hlr, flux)`` from the detection truth catalog."""
    catalog = Table.read(path)
    q = np.array(catalog["c10_sersic_fit_q"], dtype=np.float64)
    phi = np.array(catalog["c10_sersic_fit_phi"], dtype=np.float64)
    hlr = np.array(catalog["c10_sersic_fit_hlr"], dtype=np.float64) * PIXEL_SCALE * np.sqrt(q)
    hlr = np.minimum(hlr, HLR_CEILING)
    hlr = np.where(np.isfinite(hlr) & (hlr > 0), hlr, 1e-6)
    flux = np.array(catalog["crates_b"], dtype=np.float64) * FLUX_FACTOR
    return q, phi, hlr, flux


def shears_from(q: np.ndarray, phi: np.ndarray):
    """``(g1, g2)`` for axis ratio and position angle, via GalSim."""
    import galsim

    g1 = np.empty(len(q))
    g2 = np.empty(len(q))
    for i, (qi, phi_i) in enumerate(zip(q, phi)):
        qi = float(qi)
        if qi > 1.0:
            qi = 1.0 / qi
        shear = galsim.Shear(q=qi, beta=float(phi_i) * galsim.radians)
        g1[i], g2[i] = shear.g1, shear.g2
    return g1, g2


def size_floor(hlr, *, min_hlr=None, min_resolution=None, psf_fwhm=None):
    """Keep galaxies a survey could actually measure, as a truth-space mask."""
    floor = min_hlr
    if min_resolution is not None:
        if psf_fwhm is None:
            raise SystemExit("--min-resolution needs --psf-fwhm")
        from_resolution = min_resolution * (psf_fwhm / 2.0)
        floor = from_resolution if floor is None else max(floor, from_resolution)
    if floor is None:
        return np.ones(len(hlr), dtype=bool), None
    return hlr >= floor, floor


def _table(q, phi, g1, g2, hlr, flux, header_items):
    columns = [
        fits.Column(name="Q", format="D", array=q, unit=""),
        fits.Column(name="PHI", format="D", array=phi, unit="rad"),
        fits.Column(name="G1", format="D", array=g1, unit=""),
        fits.Column(name="G2", format="D", array=g2, unit=""),
        fits.Column(name="HLR", format="D", array=hlr, unit="arcsec"),
        fits.Column(name="FLUX", format="D", array=flux, unit="count"),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header.comments["TTYPE1"] = "Axis ratio b/a, (0, 1]"
    hdu.header.comments["TTYPE2"] = "Position angle of major axis [rad]"
    hdu.header.comments["TTYPE3"] = "Reduced shear component 1"
    hdu.header.comments["TTYPE4"] = "Reduced shear component 2"
    hdu.header.comments["TTYPE5"] = "Circularized half-light radius [arcsec]"
    hdu.header.comments["TTYPE6"] = "Galaxy flux [count]"
    for key, (value, comment) in header_items.items():
        hdu.header[key] = (value, comment)
    return hdu


def build(parent: Path, out_dir: Path, *, n_augments=10, train_fraction=0.6,
          seed=42, min_hlr=None, min_resolution=None, psf_fwhm=None,
          report_only=False):
    q, phi, hlr, flux = read_parent(parent)
    n_parent = len(q)

    keep, floor = size_floor(hlr, min_hlr=min_hlr,
                             min_resolution=min_resolution, psf_fwhm=psf_fwhm)
    n_kept = int(keep.sum())

    print(f"parent          {parent.name}: {n_parent} source galaxies")
    print(f"  hlr spans     {hlr.min():.4f} to {hlr.max():.4f}\"")
    if floor is not None:
        print(f"  size floor    hlr >= {floor:.4f}\"")
    print(f"  surviving     {n_kept} ({100.0 * n_kept / n_parent:.1f}%)")

    # Split on the SOURCE galaxy, then augment. The other order puts rotations
    # of one galaxy on both sides of the split.
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_kept)
    n_train_gal = int(round(n_kept * train_fraction))
    split = {"train": order[:n_train_gal], "eval": order[n_train_gal:]}

    print(f"\nsplit by source galaxy (seed {seed}), then augment x{n_augments}:")
    for name, galaxies in split.items():
        print(f"  {name:<6} {len(galaxies):7d} galaxies -> "
              f"{len(galaxies) * n_augments:7d} rows")
    print("  no source galaxy appears on both sides, in any orientation")

    if report_only:
        return {}

    q_kept, phi_kept = q[keep], phi[keep]
    hlr_kept, flux_kept = hlr[keep], flux[keep]

    written = {}
    for name, galaxies in split.items():
        # Tile the orientation-independent quantities, redraw the angle.
        index = np.repeat(galaxies, n_augments)
        angles = rng.uniform(0, 2 * np.pi, size=len(index))
        g1, g2 = shears_from(q_kept[index], angles)
        hdu = _table(
            q_kept[index], angles, g1, g2, hlr_kept[index], flux_kept[index],
            {
                "SUBSET": (name.upper(), "train/eval split by SOURCE galaxy"),
                "RNDSD": (seed, "RNG seed for the split"),
                "NGAL": (len(galaxies), "distinct source galaxies here"),
                "NAUG": (n_augments, "orientations per source galaxy"),
                "NPARENT": (n_parent, "source galaxies before the size floor"),
                "HLRFLOOR": (floor if floor is not None else 0.0,
                             "arcsec; truth-space size floor"),
                "GRPSPLIT": (True, "split on source galaxy, not augmented row"),
                "ORIGIN": (parent.name, "source catalog"),
            },
        )
        path = out_dir / f"cosmos_catalog_{name}.fits"
        hdu.writeto(path, overwrite=True)
        written[name] = path
        print(f"wrote {path}  ({len(index)} rows)")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--n-augments", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-hlr", type=float, default=None)
    parser.add_argument("--min-resolution", type=float, default=None,
                        help="size floor in PSF half-widths")
    parser.add_argument("--psf-fwhm", type=float, default=0.5)
    parser.add_argument("--report-only", action="store_true",
                        help="print the counts a floor would give, write nothing")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build(args.parent, args.out_dir, n_augments=args.n_augments,
          train_fraction=args.train_fraction, seed=args.seed,
          min_hlr=args.min_hlr, min_resolution=args.min_resolution,
          psf_fwhm=args.psf_fwhm, report_only=args.report_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
