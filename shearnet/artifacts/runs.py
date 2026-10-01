"""The run directory: one place for everything a training run produces.

::

    <run>/
      config.input.yaml        the YAML as given
      config.resolved.yaml     every setting, defaults filled, paths absolute
      manifest.json            identity and provenance, checkpoint hash
      status.json              pending / running / completed / failed
      model/best.msgpack       the parameters to evaluate (EMA ones if EMA)
      normalizers/labels.npz   always (identity when label normalization is off)
      normalizers/images.npz   only when image normalization is on
      training/history.csv     one row per epoch
      training/history.npz     the same numbers as arrays
      training/learning_curve.png
      logs/train.log
      evaluations/<name>/      one per shearnet-eval run, see EvaluationDir

A run is found by its directory, never by searching for a name prefix.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["RunDir", "EvaluationDir", "RunError", "atomic_write", "write_json"]

STATES = ("pending", "running", "completed", "failed")


class RunError(RuntimeError):
    """A run directory that cannot be used as asked."""


def atomic_write(path, data: bytes) -> None:
    """Write ``data`` to ``path`` so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path, payload: Dict[str, Any]) -> None:
    atomic_write(path, (json.dumps(payload, indent=2, sort_keys=False, default=str)
                        + "\n").encode())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class _StatusMixin:
    status_path: Path

    def write_status(self, state: str, **info) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        current = self.read_status() or {}
        record = dict(current, state=state, updated=_now(), **info)
        if state == "running" and "started" not in current:
            record["started"] = record["updated"]
        if state in ("completed", "failed"):
            record["finished"] = record["updated"]
        write_json(self.status_path, record)

    def read_status(self) -> Optional[Dict[str, Any]]:
        if not self.status_path.is_file():
            return None
        with open(self.status_path) as handle:
            return json.load(handle)

    @property
    def state(self) -> Optional[str]:
        status = self.read_status()
        return None if status is None else status.get("state")


class RunDir(_StatusMixin):
    """Paths inside one training run, and its state."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    # -- layout --------------------------------------------------------------
    config_input = property(lambda self: self.root / "config.input.yaml")
    config_resolved = property(lambda self: self.root / "config.resolved.yaml")
    manifest_path = property(lambda self: self.root / "manifest.json")
    status_path = property(lambda self: self.root / "status.json")
    model_dir = property(lambda self: self.root / "model")
    best_params = property(lambda self: self.root / "model" / "best.msgpack")
    normalizers_dir = property(lambda self: self.root / "normalizers")
    label_normalizer = property(lambda self: self.root / "normalizers" / "labels.npz")
    image_normalizer = property(lambda self: self.root / "normalizers" / "images.npz")
    training_dir = property(lambda self: self.root / "training")
    history_csv = property(lambda self: self.root / "training" / "history.csv")
    history_npz = property(lambda self: self.root / "training" / "history.npz")
    learning_curve = property(lambda self: self.root / "training" / "learning_curve.png")
    logs_dir = property(lambda self: self.root / "logs")
    train_log = property(lambda self: self.root / "logs" / "train.log")
    evaluations_dir = property(lambda self: self.root / "evaluations")

    def __repr__(self) -> str:
        return f"RunDir({str(self.root)!r})"

    # -- lifecycle -----------------------------------------------------------
    def create(self, overwrite: bool = False) -> None:
        """Make a fresh run directory.

        An existing non-empty directory is refused: a completed run is never
        silently replaced, and a half-written one should be looked at before it
        is thrown away. ``overwrite`` deletes it first, evaluations included --
        they measured the model being replaced.
        """
        if self.root.exists() and any(self.root.iterdir()):
            if not overwrite:
                state = self.state or "no status"
                raise RunError(
                    f"{self.root} already holds a run ({state}). Pick another run "
                    "directory, or pass --overwrite to delete it and start again.")
            shutil.rmtree(self.root)
        for path in (self.root, self.model_dir, self.normalizers_dir, self.training_dir,
                     self.logs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def require_completed(self) -> None:
        """Refuse a directory that is not a finished training run."""
        if not self.root.is_dir():
            raise RunError(f"no run directory at {self.root}")
        state = self.state
        if state != "completed":
            raise RunError(
                f"{self.root} is not a completed training run (status: {state or 'missing'}). "
                "Evaluating it would measure a partial model or none at all.")
        for path in (self.config_resolved, self.manifest_path, self.best_params,
                     self.label_normalizer):
            if not path.is_file():
                raise RunError(f"{self.root} is marked completed but {path.name} is missing")

    def read_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path) as handle:
            return json.load(handle)

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        write_json(self.manifest_path, manifest)

    def evaluation(self, name: str) -> "EvaluationDir":
        return EvaluationDir(self, name)


class EvaluationDir(_StatusMixin):
    """One ``shearnet-eval`` of a run: ``<run>/evaluations/<name>/``."""

    def __init__(self, run: RunDir, name: str):
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"bad evaluation name {name!r}")
        self.run = run
        self.name = name
        self.root = run.evaluations_dir / name

    config_input = property(lambda self: self.root / "config.input.yaml")
    config_resolved = property(lambda self: self.root / "config.resolved.yaml")
    status_path = property(lambda self: self.root / "status.json")
    log = property(lambda self: self.root / "eval.log")
    scratch = property(lambda self: self.root / ".partial")

    def catalog_path(self, run_name: str) -> Path:
        return self.root / f"{run_name}_{self.name}.fits"

    def create(self, overwrite: bool = False) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            if not overwrite:
                raise RunError(
                    f"{self.root} already holds an evaluation ({self.state or 'no status'}). "
                    "Give this one another --eval-name, or pass --overwrite.")
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
