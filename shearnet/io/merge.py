"""Join two complete estimator catalogs without changing their measurements.

The ShearNet job renders on the GPU and the ngmix job on CPUs, so their copies
of the shared ``TRUTH`` and ``STAMP`` tables agree to roundoff, not bit for bit.
The merge checks every shared column to :data:`TOLERANCE` (integers and
identifiers exactly), takes the shared tables from the ngmix catalog, and copies
both estimators' measurement tables byte for byte.
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..artifacts.recovery import exclusive
from .catalog_schema import Column
from .fits_catalog import PER_RECORD, CatalogWriter, key_value_hdu, validate_catalog

#: Relative agreement required of the two jobs' copies of a shared value. The
#: roundoff between devices is ~1e-7; a different galaxy, noise field or PSF
#: changes the values at order unity.
TOLERANCE = 1e-4

#: Absolute floor under that: values near zero (a round PSF's psf_g) differ by
#: ~1e-16 of roundoff, which no relative tolerance accepts. Every quantity
#: compared is in units where 1e-12 is far below anything meaningful.
ABSOLUTE = 1e-12

#: Header keys that must be identical in the two catalogs.
SHARED_HEADER = ("SCHEMA", "SCHEMAV", "OBSSHA", "IMPLSHA", "NOBJ", "NRECORD", "NSCENE",
                 "NROT", "MCALSTEP", "MCALPSF")


def table_columns(hdul):
    tables = {}
    for row in hdul["SCHEMA"].data:
        shape = () if row["shape"] == "1" else tuple(map(int, row["shape"].split("x")))
        tables.setdefault(row["extension"], []).append(
            Column(row["column"], row["dtype"], shape, row["unit"], row["description"]))
    return tables


#: Columns that are sums over the galaxy stamp: they can sit near zero while
#: their roundoff scales with the stamp, so they are compared against the
#: stamp's own sum |x| rather than their value.
IMAGE_SUMS = ("flux_stamp",)


def agree(name, x, y, stamp_scale=None):
    """Whether two copies of column ``name`` describe the same observations.

    Each value is judged on its own, so the answer never depends on the other
    rows compared with it. Integers and strings must match exactly; floats to
    :data:`TOLERANCE` of their own size, of their image's ``sum |x|`` for a
    fingerprint, and of ``stamp_scale`` (the row's galaxy ``sum |x|``) for the
    columns in :data:`IMAGE_SUMS` -- always with :data:`ABSOLUTE` as a floor.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        return False
    if x.dtype.kind != "f":
        return np.array_equal(x, y)
    x, y = x.astype(np.float64), y.astype(np.float64)
    if name.endswith("_fingerprint"):
        # every entry is bounded by the image's sum |x|, the first one
        scale = np.maximum(np.abs(x[..., :1]), np.abs(y[..., :1]))
    elif name in IMAGE_SUMS:
        scale = np.asarray(stamp_scale, dtype=np.float64)
    else:
        scale = np.maximum(np.abs(x), np.abs(y))
    close = np.abs(x - y) <= TOLERANCE * np.nan_to_num(scale) + ABSOLUTE
    return bool(np.all(close | (np.isnan(x) & np.isnan(y))))


def _check_input(path, hdul, estimator):
    if hdul[0].header.get("ESTIMATR") != estimator:
        raise ValueError(f"{path}: expected a standalone {estimator} catalog")
    tables = table_columns(hdul)
    if set(tables) != {"TRUTH", "STAMP", estimator.upper()}:
        raise ValueError(f"{path}: unexpected estimator tables")
    n = hdul[0].header["NRECORD"]
    validate_catalog(path, {t: [c.name for c in cols] for t, cols in tables.items()}, n)
    # canonical record order is required; the UIDs prove both inputs use it
    uids = hdul["TRUTH"].data["observation_uid"]
    if len(np.unique(uids)) != n or np.any(uids == ""):
        raise ValueError(f"{path}: duplicate or missing observation UID")
    return tables


def merge(shearnet, ngmix, output, chunk=8192):
    output = Path(output).resolve()
    if output in (Path(shearnet).resolve(), Path(ngmix).resolve()):
        raise ValueError("merge output must not replace an input")
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(output.with_suffix(".lock")), \
            fits.open(shearnet, memmap=True) as a, fits.open(ngmix, memmap=True) as b:
        inputs = ((a, "shearnet"), (b, "ngmix"))
        a_tables = _check_input(shearnet, a, "shearnet")
        b_tables = _check_input(ngmix, b, "ngmix")
        for key in SHARED_HEADER:
            if a[0].header[key] != b[0].header[key]:
                raise ValueError(f"catalogs disagree on {key}")
        for name in ("SCENES", "ROTATIONS"):
            if a[name].data.tobytes() != b[name].data.tobytes():
                raise ValueError(f"catalogs disagree on {name}")
        for name in ("TRUTH", "STAMP"):
            if a_tables[name] != b_tables[name]:
                raise ValueError(f"{name} columns differ between the catalogs")

        n = a[0].header["NRECORD"]
        tables = {"TRUTH": b_tables["TRUTH"], "STAMP": b_tables["STAMP"],
                  "SHEARNET": a_tables["SHEARNET"], "NGMIX": b_tables["NGMIX"]}
        source = {"TRUTH": b, "STAMP": b, "SHEARNET": a, "NGMIX": b}
        writer = CatalogWriter(output.parent / f".{output.name}.assembly", tables, n)
        for start in range(0, n, chunk):
            rows = slice(start, min(n, start + chunk))
            fingerprints = b["STAMP"].data["input_fingerprint"][rows]
            stamp_scale = np.abs(fingerprints[:, 0, 0].astype(np.float64))
            for name in ("TRUTH", "STAMP"):
                for column in tables[name]:
                    if not agree(column.name, a[name].data[column.name][rows],
                                 b[name].data[column.name][rows], stamp_scale):
                        raise ValueError(f"{name}.{column.name}: the two jobs observed "
                                         f"different things in records {rows.start}-"
                                         f"{rows.stop - 1}")
            for name in PER_RECORD:
                writer.fill(name, rows, {c.name: source[name][name].data[c.name][rows]
                                         for c in tables[name]})

        skip = ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "CHECKSUM", "DATASUM", "ESTIMATR",
                "BACKEND")
        header = {k: (a[0].header[k], a[0].header.comments[k]) for k in a[0].header
                  if k not in skip}
        header["ESTIMATR"] = ("shearnet,ngmix", "merged estimators")
        devices = {e: h[0].header["BACKEND"].split("/")[-1] for h, e in inputs}
        header["BACKEND"] = (f"jax-galsim; shearnet {devices['shearnet']}, ngmix "
                             f"{devices['ngmix']}", "renderer; device per job")
        extra = [a[name].copy() for name in ("SCENES", "ROTATIONS", "PSF_FILES", "BLOCKS")]
        for name in ("PROTOCOL", "CONFIG", "PROVENANCE", "TIMING"):
            combined = {f"{estimator}.{row['key']}": row["value"]
                        for hdul, estimator in inputs for row in hdul[name].data}
            if name == "CONFIG":
                # the model and evaluation configuration describe the merged catalog
                combined.update({row["key"]: row["value"] for row in a[name].data})
            if name == "PROTOCOL":
                combined["merge"] = (f"TRUTH and STAMP from the ngmix job; the ShearNet job's "
                                     f"copies agree to {TOLERANCE:g} relative")
            extra.append(key_value_hdu(name, combined))
        result = writer.write(output, header, extra)
        writer.cleanup()
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shearnet", required=True)
    parser.add_argument("--ngmix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(merge(args.shearnet, args.ngmix, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
