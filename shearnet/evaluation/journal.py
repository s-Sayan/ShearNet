"""Committed FITS shards are the evaluation's recovery record."""
import json
import os
from pathlib import Path
import numpy as np
from astropy.io import fits
from ..artifacts.runs import atomic_write
from ..artifacts.recovery import sync_directory
from ..io.fits_catalog import _dtype
from .identity import digest


class Journal:
    def __init__(self, root, identity, tables, nrecords, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tables = tables
        self.nrecords = nrecords
        self.done = np.zeros(nrecords, bool)
        self.paths = []
        self.signature = digest(identity)
        manifest = self.root / "identity.json"
        if manifest.exists():
            if not resume:
                raise FileExistsError(f"{root} already exists; use --resume")
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("resume inputs, implementation or estimator differ")
        else:
            if any(self.root.iterdir()):
                raise ValueError(f"{root}: shards exist without an identity manifest")
            atomic_write(manifest, (json.dumps(identity, sort_keys=True) + "\n").encode())
        for path in sorted(self.root.glob("*.fits")):
            with self.open(path) as hdul:
                self._claim(hdul["TRUTH"].data["record_id"])
            self.paths.append(path)

    def _claim(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if (np.any(ids < 0) or np.any(ids >= self.nrecords)
                or len(np.unique(ids)) != len(ids) or np.any(self.done[ids])):
            raise ValueError("duplicate or out-of-range shard records")
        self.done[ids] = True

    def open(self, path):
        hdul = fits.open(path, memmap=True)
        try:
            if hdul[0].header.get("IDENTITY") != self.signature:
                raise ValueError(f"{path}: shard belongs to another evaluation")
            for hdu in hdul:
                if hdu.verify_checksum() != 1 or hdu.verify_datasum() != 1:
                    raise ValueError(f"{path}: invalid FITS checksum")
            ids = hdul["TRUTH"].data["record_id"]
            for name, columns in self.tables.items():
                if hdul[name].data.dtype != _dtype(columns):
                    raise ValueError(f"{path}: {name} schema differs")
                if not np.array_equal(hdul[name].data["record_id"], ids):
                    raise ValueError(f"{path}: tables are not aligned")
            return hdul
        except BaseException:
            hdul.close()
            raise

    def commit(self, records, timing=None):
        """Publish ``records`` as one shard. ``timing`` (FITS keyword -> number)
        is the work spent on exactly these records; :meth:`timing` sums it."""
        if not records:
            return
        records.sort(key=lambda row: row["TRUTH"]["record_id"])
        ids = np.array([r["TRUTH"]["record_id"] for r in records])
        if np.any(self.done[ids]):
            raise ValueError("attempted to write a completed record")
        primary = fits.PrimaryHDU()
        primary.header["IDENTITY"] = self.signature
        for key, value in (timing or {}).items():
            primary.header[key] = value
        hdus = [primary]
        for name, columns in self.tables.items():
            array = np.empty(len(records), dtype=_dtype(columns))
            for i, record in enumerate(records):
                if set(record[name]) != set(array.dtype.names):
                    raise ValueError(f"{name}: incomplete record {ids[i]}")
                for key, value in record[name].items():
                    array[key][i] = value
            hdus.append(fits.BinTableHDU(array, name=name))
        index = max([int(p.stem) for p in self.paths], default=-1) + 1
        path = self.root / f"{index:08d}.fits"
        tmp = path.with_suffix(".writing")
        fits.HDUList(hdus).writeto(tmp, overwrite=True, checksum=True)
        with self.open(tmp):
            pass
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        sync_directory(path.parent)
        self._claim(ids)
        self.paths.append(path)

    def timing(self, keys):
        """Each of ``keys`` summed over every committed shard."""
        totals = dict.fromkeys(keys, 0.0)
        for path in self.paths:
            header = fits.getheader(path, 0)
            for key in keys:
                totals[key] += float(header.get(key, 0.0))
        return totals
