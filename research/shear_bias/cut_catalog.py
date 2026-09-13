"""Write a size-cut copy of a COSMOS catalog FITS.

The catalogs are consumed as plain FITS tables -- ``dataset._load_cosmos_cat``
reads ``hdul[1].data`` and indexes ``G1``, ``G2``, ``HLR``, ``FLUX`` row by row
-- so a resolution floor on the *training* set needs no code change at all.
Filter the table, point ``paths.train_catalog`` at the copy, and every renderer
downstream draws from the truncated distribution.

That is the whole mechanism behind the retrain: the question is whether feeding
the network galaxies it cannot resolve (hlr down to 0.0" against a 0.5" FWHM
PSF) costs it accuracy on the ones it can, and the only way to answer it is to
train on a catalog that does not contain them.

    python research/shear_bias/cut_catalog.py \\
        --in  cosmos_catalog_train.fits --out cosmos_catalog_train_r1.fits \\
        --min-resolution 1.0 --psf-fwhm 0.5

``--min-resolution`` is in PSF half-widths, so 1.0 keeps galaxies at least as
large as the PSF -- the truth-space analogue of SuperBIT's ``min_Tpsf = 1.0``.
Give ``--min-hlr`` instead to set the floor in arcsec directly.

TWO THINGS TO KNOW BEFORE USING THE OUTPUT
------------------------------------------
Rows are consumed by index, so a cut catalog holds fewer galaxies than the
original and the renderer refuses a ``samples`` larger than the table. The
count is printed; check it against the ``samples`` in your configs.

The cut also renumbers every row, so index *i* is a different galaxy than it
was. A run on the cut catalog is not the same population as a run on the full
one at the same seed, and the two are comparable only as ensembles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def cut_catalog(source: Path, destination: Path, *, min_hlr: float = None,
                min_resolution: float = None, psf_fwhm: float = None,
                max_hlr: float = None) -> int:
    """Copy ``source`` to ``destination`` keeping only large-enough rows."""
    with fits.open(source) as hdul:
        hdu = hdul[1]
        data = hdu.data
        hlr = np.asarray(data["HLR"], dtype=float)

        floor = min_hlr
        if min_resolution is not None:
            if psf_fwhm is None:
                raise SystemExit("--min-resolution needs --psf-fwhm")
            resolution_floor = min_resolution * (psf_fwhm / 2.0)
            floor = resolution_floor if floor is None else max(floor, resolution_floor)

        keep = np.isfinite(hlr)
        if floor is not None:
            keep &= hlr >= floor
        if max_hlr is not None:
            keep &= hlr <= max_hlr

        print(f"{source.name}: {len(hlr)} rows, "
              f"hlr {np.nanmin(hlr):.4f} to {np.nanmax(hlr):.4f}\"")
        if floor is not None:
            print(f"  floor      hlr >= {floor:.4f}\"")
        if max_hlr is not None:
            print(f"  ceiling    hlr <= {max_hlr:.4f}\"")
        print(f"  kept       {keep.sum()} ({100.0 * keep.sum() / len(hlr):.1f}%)")
        print(f"  discarded  {(~keep).sum()}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        # A new BinTableHDU from the filtered rows, carrying the original
        # header so anything else reading this file sees what it expects.
        out = fits.BinTableHDU(data=data[keep], header=hdu.header, name=hdu.name)
        out.header["HLRFLOOR"] = (floor if floor is not None else 0.0,
                                  "arcsec; size cut applied by cut_catalog.py")
        out.header["HLRNORIG"] = (len(hlr), "rows before the cut")
        fits.HDUList([fits.PrimaryHDU(), out]).writeto(destination, overwrite=True)

    print(f"wrote {destination}")
    return int(keep.sum())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", required=True, type=Path)
    parser.add_argument("--out", dest="destination", required=True, type=Path)
    parser.add_argument("--min-hlr", type=float, default=None,
                        help="keep HLR >= this, in arcsec")
    parser.add_argument("--max-hlr", type=float, default=None,
                        help="keep HLR <= this, in arcsec")
    parser.add_argument("--min-resolution", type=float, default=None,
                        help="keep HLR >= this many PSF half-widths")
    parser.add_argument("--psf-fwhm", type=float, default=0.5,
                        help="PSF FWHM in arcsec, for --min-resolution")
    args = parser.parse_args(argv)

    if args.min_hlr is None and args.min_resolution is None and args.max_hlr is None:
        raise SystemExit("nothing to cut: give --min-hlr, --min-resolution or --max-hlr")

    cut_catalog(args.source, args.destination, min_hlr=args.min_hlr,
                min_resolution=args.min_resolution, psf_fwhm=args.psf_fwhm,
                max_hlr=args.max_hlr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
