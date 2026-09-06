"""Rewrite an evaluation FITS under the schema policy of :mod:`catalog`.

Two uses. First, immediate relief on files already produced: a 1.2 GB
``evaluation.fits`` from a finished run becomes a few hundred MB at ``paper``
level, or kilobytes at ``summary``, without re-running anything. Second, a way
to check the policy against a real file before wiring it into the harness --
the derived tables are copied through untouched and compared, so a run that
changes ``SUMMARY`` by so much as a bit fails loudly instead of quietly.

    python -m research.shear_bias.shrink_fits evaluation.fits
    python -m research.shear_bias.shrink_fits evaluation.fits --level summary
    python -m research.shear_bias.shrink_fits eval.fits -o slim.fits --keep-float64

The derived tables (``SUMMARY``, ``BINNED``, ``LEAKSUM``) are never touched at
any level: they were computed from the full-precision arrays inside the harness,
so they are the authoritative answer regardless of what the per-object tables
were trimmed to.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:  # as a module inside the package, or as a script beside catalog.py
    from .catalog import (CATALOG_LEVELS, DEFAULT_LEVEL, SlimReport,
                          meta_table, resolve_level, slim_columns)
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from catalog import (CATALOG_LEVELS, DEFAULT_LEVEL, SlimReport,
                         meta_table, resolve_level, slim_columns)

logger = logging.getLogger("shrink_fits")

#: Tables holding one row per object. Everything else is a derived summary and
#: is copied through verbatim.
PER_OBJECT_PREFIXES = ("TAB_", "LEAKAGE")

#: Copied unchanged at every level. These are the numbers the paper quotes.
DERIVED_NAMES = ("SUMMARY", "BINNED", "LEAKSUM")


def _is_per_object(name: str) -> bool:
    return any(name.upper().startswith(prefix) for prefix in PER_OBJECT_PREFIXES)


def _columns_of(hdu) -> dict:
    """``{name: ndarray}`` from a BinTableHDU, preserving vector columns."""
    data = hdu.data
    return {name: np.asarray(data[name]) for name in data.columns.names}


def shrink(path: Path, out: Path, level: str, *, float_dtype=np.float32) -> Tuple[int, int]:
    """Rewrite ``path`` to ``out`` at ``level``. Returns ``(before, after)`` bytes."""
    from astropy.io import fits
    from astropy.table import Table

    level = resolve_level(level)
    before = path.stat().st_size

    reports: List[Tuple[str, SlimReport]] = []
    derived_checksums = {}
    out_hdus = []

    with fits.open(path, memmap=True) as hdul:
        primary = fits.PrimaryHDU(header=hdul[0].header.copy())
        primary.header["CATLEVEL"] = (level, "per-object catalog schema level")
        out_hdus.append(primary)

        for hdu in hdul[1:]:
            name = (hdu.name or "").upper()
            if not isinstance(hdu, fits.BinTableHDU):
                out_hdus.append(hdu.copy())
                continue

            if name in DERIVED_NAMES or not _is_per_object(name):
                # Copied verbatim, and checksummed so the caller can prove the
                # reported numbers did not move.
                copied = fits.BinTableHDU(hdu.data.copy(), header=hdu.header.copy(),
                                          name=hdu.name)
                derived_checksums[name] = _checksum(hdu.data)
                out_hdus.append(copied)
                continue

            columns = _columns_of(hdu)
            kept, report = slim_columns(columns, level, float_dtype=float_dtype)
            report.log(name)
            reports.append((name, report))
            if not kept:
                logger.info("%s: dropped entirely at level=%s", name, level)
                continue
            slim = fits.BinTableHDU(Table(kept), header=_stripped(hdu.header), name=hdu.name)
            out_hdus.append(slim)

    out_hdus.append(fits.BinTableHDU(meta_table(reports), name="SLIMMETA"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(out_hdus).writeto(out, overwrite=True)

    _verify_derived(out, derived_checksums)
    return before, out.stat().st_size


def _stripped(header):
    """Copy a table header without its column descriptors.

    ``TTYPEn``/``TFORMn``/``TUNITn`` describe the *old* column set; astropy
    regenerates them for the new table, and leaving stale ones behind produces a
    file that opens but mislabels columns.
    """
    clean = header.copy()
    for key in list(clean):
        if key[:5] in ("TTYPE", "TFORM", "TUNIT", "TDISP", "TDIM ") or key[:4] == "TDIM":
            del clean[key]
    for key in ("NAXIS1", "NAXIS2", "TFIELDS", "PCOUNT", "GCOUNT"):
        if key in clean:
            del clean[key]
    return clean


def _checksum(data) -> str:
    """A stable digest of a FITS table's bytes."""
    import hashlib

    digest = hashlib.sha256()
    for name in data.columns.names:
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(data[name]).tobytes())
    return digest.hexdigest()


def _verify_derived(path: Path, expected: dict) -> None:
    """Fail loudly if a derived table changed. It never should."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        for name, want in expected.items():
            hdu = hdul[name]
            got = _checksum(hdu.data)
            if got != want:
                raise RuntimeError(
                    f"{name} changed while slimming ({want[:12]} -> {got[:12]}). "
                    "The derived tables must be copied verbatim; refusing to "
                    "present this file as equivalent."
                )
    logger.info("verified %d derived tables unchanged: %s",
                len(expected), ", ".join(sorted(expected)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="evaluation FITS to shrink")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output path (default: <name>.slim.fits)")
    parser.add_argument("--level", default=DEFAULT_LEVEL, choices=CATALOG_LEVELS,
                        help=f"schema level (default: {DEFAULT_LEVEL})")
    parser.add_argument("--keep-float64", action="store_true",
                        help="do not downcast float64 columns")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="log the reason each column was dropped")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    out = args.output or args.path.with_suffix(".slim.fits")
    if out.resolve() == args.path.resolve():
        parser.error("refusing to overwrite the input in place; pass -o")

    before, after = shrink(
        args.path, out, args.level,
        float_dtype=np.float64 if args.keep_float64 else np.float32,
    )
    logger.info("%s -> %s", args.path, out)
    logger.info("%.1f MB -> %.1f MB (%.1fx smaller)",
                before / 1e6, after / 1e6, before / max(after, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
