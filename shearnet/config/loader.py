"""YAML in and out, without PyYAML's two traps.

* ``1e-4`` is a float. PyYAML implements YAML 1.1, where a float needs a dot,
  so its default loader parses ``learning_rate: 1e-4`` as the *string* ``"1e-4"``. The
  resolver below is the one SuperBIT's ``utils.read_yaml`` installs, on a
  private loader class rather than on ``yaml.SafeLoader`` itself.
* A key written twice is an error. PyYAML keeps the last one silently, which
  turns a copy-paste slip into a different experiment.
"""

from __future__ import annotations

import re
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """``SafeLoader`` with scientific-notation floats and duplicate-key errors."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
        [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _construct_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            mark = key_node.start_mark
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r} (line {mark.line + 1})", mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(text: str) -> Any:
    """Parse YAML text with the rules above. An empty document is ``{}``."""
    data = yaml.load(text, Loader=_Loader)
    return {} if data is None else data


def read_yaml(path) -> Any:
    """:func:`load_yaml` on a file."""
    with open(path, "r") as handle:
        return load_yaml(handle.read())


class _Dumper(yaml.SafeDumper):
    """Block style, insertion order, and lists of scalars kept on one line."""


def _represent_list(dumper, data):
    flow = all(not isinstance(v, (dict, list)) for v in data) and len(data) <= 8
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _represent_list)
_Dumper.add_representer(tuple, _represent_list)


def dump_yaml(data) -> str:
    """Readable YAML that :func:`load_yaml` reads back to the same values."""
    return yaml.dump(data, Dumper=_Dumper, default_flow_style=False, sort_keys=False)
