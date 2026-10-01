"""Write and read the evaluation catalog.

The per-record tables are filled block by block into memory-mapped,
big-endian structured arrays under the evaluation's ``.partial/`` directory, so
a 4-million-row catalog never has to sit in RAM, and the FITS writer can hand
the bytes straight through. The finished file is written next to its final
name, read back and checked (extensions, row counts, columns, keys), and only
then renamed into place. A file at the final name is therefore always a whole
catalog.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog_schema import SCHEMA_NAME, SCHEMA_VERSION, Column, schema_rows

__all__ = ["CatalogWriter", "read_table", "validate_catalog", "PER_RECORD"]

#: Extensions with one row per record, in the order they are written.
PER_RECORD = ("TRUTH", "STAMP", "SHEARNET", "NGMIX")


def _dtype(columns: Sequence[Column]) -> np.dtype:
    return np.dtype([(c.name, ">" + c.dtype, c.shape) if c.shape else (c.name, ">" + c.dtype)
                     for c in columns])


def _fill_value(column: Column):
    return np.nan if column.dtype.startswith("f") else (-1 if column.name.endswith("_id")
                                                         else 0)


class CatalogWriter:
    """Preallocated per-record tables, filled by row range, then written once."""

    def __init__(self, scratch: Path, tables: Mapping[str, List[Column]], nrows: int):
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.columns = {name: list(cols) for name, cols in tables.items()}
        self.nrows = int(nrows)
        self.arrays = {}
        for name, cols in self.columns.items():
            names = [c.name for c in cols]
            if len(set(names)) != len(names):
                raise ValueError(f"{name}: duplicate column names")
            array = np.lib.format.open_memmap(self.scratch / f"{name}.npy", mode="w+",
                                              dtype=_dtype(cols), shape=(self.nrows,))
            for column in cols:
                array[column.name] = _fill_value(column)
            self.arrays[name] = array
        self.filled = {name: np.zeros(self.nrows, dtype=bool) for name in self.columns}

    def fill(self, table: str, rows: slice, values: Mapping[str, np.ndarray]) -> None:
        """Write ``values`` (column -> array of len(rows)) into ``rows`` of ``table``."""
        array = self.arrays[table]
        known = {c.name for c in self.columns[table]}
        unknown = sorted(set(values) - known)
        if unknown:
            raise KeyError(f"{table} has no columns {unknown}")
        for name, value in values.items():
            array[name][rows] = value
        self.filled[table][rows] = True

    def write(self, path: Path, header: Mapping[str, Tuple], extra_hdus: Iterable) -> Path:
        """Write the FITS next to ``path``, validate it, then move it into place."""
        from astropy.io import fits

        missing = {name: int((~done).sum()) for name, done in self.filled.items()
                   if not done.all()}
        if missing:
            raise RuntimeError(f"refusing to write a partial catalog; unfilled rows: {missing}")
        for array in self.arrays.values():
            array.flush()

        primary = fits.PrimaryHDU()
        for key, (value, comment) in header.items():
            primary.header[key] = (value, comment)
        hdus = [primary]
        for name in PER_RECORD:
            if name not in self.arrays:
                continue
            hdu = fits.BinTableHDU(data=self.arrays[name], name=name)
            for i, column in enumerate(self.columns[name], start=1):
                if column.unit:
                    hdu.header[f"TUNIT{i}"] = column.unit
            hdus.append(hdu)
        hdus.append(_table_hdu("SCHEMA", ["extension", "column", "dtype", "shape", "unit",
                                          "description"],
                               schema_rows({n: self.columns[n] for n in PER_RECORD
                                            if n in self.columns})))
        hdus.extend(extra_hdus)

        path = Path(path)
        tmp = path.with_name(f".{path.name}.writing")
        if tmp.exists():
            tmp.unlink()
        fits.HDUList(hdus).writeto(tmp, checksum=True)
        validate_catalog(tmp, {n: [c.name for c in cols] for n, cols in self.columns.items()},
                         self.nrows)
        os.replace(tmp, path)
        return path

    def cleanup(self) -> None:
        self.arrays.clear()
        shutil.rmtree(self.scratch, ignore_errors=True)


def _table_hdu(name: str, names: Sequence[str], rows: Sequence[Sequence]):
    """A small all-text table (one ``A`` column per field)."""
    from astropy.io import fits

    columns = []
    for i, field in enumerate(names):
        values = [str(row[i]) for row in rows]
        width = max([len(v.encode()) for v in values] + [1])
        columns.append(fits.Column(name=field, format=f"{width}A", array=values))
    return fits.BinTableHDU.from_columns(columns, name=name)


def key_value_hdu(name: str, items: Mapping[str, object]):
    """``key``/``value`` text table; non-string values are JSON."""
    rows = [(key, value if isinstance(value, str) else json.dumps(value, default=str))
            for key, value in items.items()]
    return _table_hdu(name, ["key", "value"], rows)


def numeric_hdu(name: str, columns: Mapping[str, np.ndarray]):
    """A small table from ``{name: array}`` (numbers or strings)."""
    from astropy.io import fits

    out = []
    for field, values in columns.items():
        values = np.asarray(values)
        if values.dtype.kind in "US":
            width = max([len(str(v).encode()) for v in values] + [1])
            out.append(fits.Column(name=field, format=f"{width}A", array=values.astype(str)))
        elif values.dtype.kind in "iu":
            out.append(fits.Column(name=field, format="K", array=values.astype(np.int64)))
        else:
            out.append(fits.Column(name=field, format="D", array=values.astype(float)))
    return fits.BinTableHDU.from_columns(out, name=name)


def validate_catalog(path, expected: Mapping[str, Sequence[str]], nrows: int) -> None:
    """Structural check of a written catalog. Raises ``ValueError`` on any problem."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        if header.get("SCHEMA") != SCHEMA_NAME or header.get("SCHEMAV") != SCHEMA_VERSION:
            raise ValueError(f"{path}: not a {SCHEMA_NAME} v{SCHEMA_VERSION} catalog")
        names = [h.name for h in hdul[1:]]
        for extension, columns in expected.items():
            if extension not in names:
                raise ValueError(f"{path}: missing extension {extension}")
            hdu = hdul[extension]
            if hdu.header["NAXIS2"] != nrows:
                raise ValueError(f"{path}: {extension} has {hdu.header['NAXIS2']} rows, "
                                 f"expected {nrows}")
            if list(hdu.columns.names) != list(columns):
                raise ValueError(f"{path}: {extension} columns differ from the schema")
        if "TRUTH" in names:
            ids = hdul["TRUTH"].data["record_id"]
            if not np.array_equal(ids, np.arange(nrows)):
                raise ValueError(f"{path}: record_id is not 0..{nrows - 1} in order")
            for extension in expected:
                if extension != "TRUTH" and not np.array_equal(
                        hdul[extension].data["record_id"], ids):
                    raise ValueError(f"{path}: {extension} rows are not aligned with TRUTH")
        for required in ("SCHEMA", "SCENES", "ROTATIONS", "PROTOCOL", "CONFIG", "PROVENANCE"):
            if required not in names:
                raise ValueError(f"{path}: missing extension {required}")


def read_table(path, extension: str, columns: Optional[Sequence[str]] = None):
    """One extension as an ``astropy.table.Table`` (optionally only some columns)."""
    from astropy.table import Table

    table = Table.read(path, hdu=extension, memmap=True)
    return table if columns is None else table[list(columns)]


def describe(path) -> Dict[str, int]:
    """``{extension: rows}`` of a catalog -- for logs and tests."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        return {h.name: int(h.header.get("NAXIS2", 0)) for h in hdul[1:]}
