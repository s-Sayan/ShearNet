"""Load, validate and query a ShearNet configuration.

A :class:`Config` is always complete: every field of
:mod:`shearnet.config.schema` has a value, every path is absolute, and every
cross-field rule has passed. Building one reads a YAML file and nothing else --
no directory is created and no simulation runs -- so a typo costs a second.

Values are read with dotted keys::

    config.get("training.epochs")
    config.get("training.response")      # a whole block, as a dict

A key the schema does not have raises :class:`KeyError`. That is deliberate:
reading a misspelled key and getting ``None`` back is how settings used to be
ignored without anyone noticing.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..logging_utils import get_logger
from . import legacy
from .loader import dump_yaml, read_yaml
from .schema import (
    EVALUATION_OVERRIDABLE,
    FIELDS,
    ConfigError,
    check_keys,
    flatten,
    resolve,
)

logger = get_logger(__name__)

__all__ = ["Config", "ConfigError", "load_config"]

_PREFIXES = frozenset(
    ".".join(key.split(".")[:i]) for key in FIELDS for i in range(1, key.count(".") + 1)
)


class Config:
    """A resolved configuration. Build it with :meth:`from_file` or :meth:`from_dict`."""

    def __init__(self, resolved: Mapping[str, Any], source: Optional[str] = None,
                 notes: Iterable[str] = ()):
        self._data = copy.deepcopy(dict(resolved))
        self.source = source
        #: What a legacy translation changed; empty for a current-schema file.
        self.notes: List[str] = list(notes)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any], base_dir: Optional[str] = None,
                  source: Optional[str] = None) -> "Config":
        """Validate ``mapping`` (current schema, or either legacy dialect)."""
        notes: List[str] = []
        mapping = dict(mapping)
        if legacy.is_legacy(mapping):
            mapping, notes = legacy.migrate(mapping)
            where = source or "a config"
            logger.warning("%s uses the pre-schema layout; translated it. Convert the "
                           "file with `python -m shearnet.config.legacy`.", where)
            for note in notes:
                logger.warning("  %s", note)
        return cls(resolve(mapping, base_dir=base_dir), source=source, notes=notes)

    @classmethod
    def from_file(cls, path) -> "Config":
        """Read and validate a YAML file. Relative paths resolve against its directory."""
        path = os.path.abspath(os.fspath(path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"config file not found: {path}")
        try:
            raw = read_yaml(path)
        except Exception as exc:  # YAML syntax, duplicate keys
            raise ConfigError(f"{path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{path}: the top level must be a mapping")
        try:
            return cls.from_dict(raw, base_dir=os.path.dirname(path), source=path)
        except ConfigError as exc:
            raise ConfigError(f"{path}: {exc}") from None

    def with_overrides(self, mapping: Mapping[str, Any], base_dir: Optional[str] = None,
                       allowed: Optional[Iterable[str]] = None) -> "Config":
        """A new config with ``mapping`` (current schema) layered on top.

        ``allowed`` restricts which keys may change; anything else is an error
        rather than a silent second meaning of the same run.
        """
        mapping = dict(mapping)
        mapping.pop("schema_version", None)
        flat = flatten(mapping)
        check_keys(flat, allowed if allowed is not None else FIELDS)
        resolved = resolve(mapping, base_dir=base_dir, base=self._data)
        return Config(resolved, source=self.source, notes=self.notes)

    def evaluation_override(self, path) -> "Config":
        """Layer an evaluation-only YAML on top. Only evaluation settings may change."""
        path = os.path.abspath(os.fspath(path))
        raw = read_yaml(path)
        if not isinstance(raw, Mapping):
            raise ConfigError(f"{path}: the top level must be a mapping")
        stray = sorted(k for k in flatten({k: v for k, v in raw.items()
                                           if k != "schema_version"})
                       if k not in EVALUATION_OVERRIDABLE)
        if stray:
            raise ConfigError(
                f"{path}: an evaluation override may only change evaluation settings, "
                f"the evaluation catalog and run_options.ncores; it sets {stray}. The "
                "model, renderer and training population belong to the run.")
        return self.with_overrides(raw, base_dir=os.path.dirname(path),
                                   allowed=EVALUATION_OVERRIDABLE)

    # -- access ------------------------------------------------------------
    def get(self, key: str) -> Any:
        """The value at dotted ``key`` (a field or a whole block)."""
        if key not in FIELDS and key not in _PREFIXES:
            raise KeyError(f"{key!r} is not a config key")
        node: Any = self._data
        for part in key.split("."):
            node = node[part]
        return copy.deepcopy(node)

    def to_dict(self) -> Dict[str, Any]:
        """The full resolved config as nested plain data."""
        return copy.deepcopy(self._data)

    def to_yaml(self) -> str:
        return dump_yaml(self._data)

    def save(self, path) -> None:
        """Write the resolved config as YAML (does not create directories)."""
        with open(path, "w") as handle:
            handle.write(self.to_yaml())

    def __eq__(self, other) -> bool:
        return isinstance(other, Config) and self._data == other._data

    def __repr__(self) -> str:
        name = self._data.get("run_options", {}).get("run_name")
        return f"Config(run_name={name!r}, source={self.source!r})"


def load_config(path) -> Config:
    """Shorthand for :meth:`Config.from_file`."""
    return Config.from_file(path)
