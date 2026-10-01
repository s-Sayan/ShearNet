"""The per-epoch training record, written as it grows.

Both training loops call a :class:`History` once per epoch with a plain dict.
Every call rewrites ``history.csv`` and ``history.npz``, so a run that dies at
epoch 40 of 60 still leaves 40 epochs on disk. Validation that did not run at an
epoch (``eval_interval > 1``) is left empty, not filled in.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..artifacts.runs import atomic_write

__all__ = ["History"]


class History:
    """Accumulates epoch records and persists them to ``csv_path`` / ``npz_path``."""

    def __init__(self, output_keys: Sequence[str], csv_path=None, npz_path=None):
        self.output_keys = tuple(output_keys)
        self.csv_path = csv_path
        self.npz_path = npz_path
        self.records: List[Dict[str, Any]] = []

    # -- recording -----------------------------------------------------------
    def __call__(self, record: Dict[str, Any]) -> None:
        self.records.append(self._flat(record))
        self.save()

    def _flat(self, record: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "epoch": int(record["epoch"]),
            "train_loss": float(record["train_loss"]),
            "val_loss": _maybe_float(record.get("val_loss")),
        }
        per_key = record.get("val_per_key")
        for i, key in enumerate(self.output_keys):
            row[f"val_mse_{key}"] = None if per_key is None else float(per_key[i])
        row["best"] = bool(record.get("best", False))
        row["seconds"] = _maybe_float(record.get("seconds"))
        for name, value in (record.get("response") or {}).items():
            row[f"response_{name}"] = float(value)
        return row

    # -- queries -------------------------------------------------------------
    @property
    def columns(self) -> List[str]:
        names: List[str] = []
        for row in self.records:
            for key in row:
                if key not in names:
                    names.append(key)
        return names

    @property
    def best_epoch(self) -> Optional[int]:
        best = [r["epoch"] for r in self.records if r["best"]]
        return best[-1] if best else None

    @property
    def best_val_loss(self) -> Optional[float]:
        losses = [r["val_loss"] for r in self.records if r["best"]]
        return losses[-1] if losses else None

    def column(self, name: str) -> np.ndarray:
        """One column as floats; NaN where it was not measured."""
        return np.array([np.nan if r.get(name) is None else float(r[name])
                         for r in self.records], dtype=float)

    # -- persistence ---------------------------------------------------------
    def save(self) -> None:
        if self.csv_path is not None:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=self.columns, restval="")
            writer.writeheader()
            for row in self.records:
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
            atomic_write(self.csv_path, buffer.getvalue().encode())
        if self.npz_path is not None:
            buffer = io.BytesIO()
            arrays = {name: self.column(name) for name in self.columns}
            arrays["epoch"] = arrays["epoch"].astype(int)
            arrays["best"] = arrays["best"].astype(bool)
            np.savez(buffer, output_keys=np.asarray(self.output_keys), **arrays)
            atomic_write(self.npz_path, buffer.getvalue())

    @classmethod
    def read_csv(cls, path, output_keys: Sequence[str]) -> "History":
        history = cls(output_keys)
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                parsed: Dict[str, Any] = {}
                for key, value in row.items():
                    if key == "epoch":
                        parsed[key] = int(value)
                    elif key == "best":
                        parsed[key] = value == "True"
                    else:
                        parsed[key] = None if value == "" else float(value)
                history.records.append(parsed)
        return history


def _maybe_float(value) -> Optional[float]:
    return None if value is None else float(value)
