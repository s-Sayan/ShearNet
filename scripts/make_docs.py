#!/usr/bin/env python
"""Regenerate the generated parts of docs/config.md and docs/catalog.md.

The tables between the ``<!-- generated:start -->`` and ``<!-- generated:end -->``
markers come from the code (the config schema and the catalog schema), so they
cannot drift from it; ``tests/test_docs.py`` fails when they are stale.

    python scripts/make_docs.py           # rewrite
    python scripts/make_docs.py --check   # verify only
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START, END = "<!-- generated:start -->", "<!-- generated:end -->"


def generated():
    from shearnet.config.schema import markdown as config_markdown
    from shearnet.io.catalog_schema import markdown as catalog_markdown

    return {REPO / "docs" / "config.md": config_markdown(),
            REPO / "docs" / "catalog.md": catalog_markdown()}


def splice(text: str, body: str) -> str:
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n\n{body.strip()}\n\n{END}{tail}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    stale = []
    for path, body in generated().items():
        text = path.read_text()
        new = splice(text, body)
        if new != text:
            stale.append(path.name)
            if not args.check:
                path.write_text(new)
    if args.check and stale:
        print("stale:", ", ".join(stale), "-- run python scripts/make_docs.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
