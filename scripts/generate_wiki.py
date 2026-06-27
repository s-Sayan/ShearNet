#!/usr/bin/env python
"""Generate GitHub-wiki Markdown for the ShearNet API using pydoc-markdown.

Parses the ``shearnet`` package statically (no heavy imports required) and writes
one Markdown page per module, plus a ``Home.md`` landing page and a ``_Sidebar.md``
navigation file, ready to be copied into the ShearNet.wiki repository.

Usage:
    python scripts/generate_wiki.py [output_dir]   # default: ./wiki
"""

import os
import re
import sys

import docspec
from pydoc_markdown.interfaces import Context
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.smart import SmartProcessor
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "shearnet"

# Human-readable section grouping for the sidebar / home page.
SECTIONS = [
    ("Core", "shearnet.core"),
    ("Command-line interface", "shearnet.cli"),
    ("Configuration", "shearnet.config"),
    ("Methods (baselines)", "shearnet.methods"),
    ("Utilities", "shearnet.utils"),
]


# Sphinx/RST cross-reference roles like :func:`foo` or :class:`~pkg.Bar` render
# literally on a Markdown wiki; convert them to plain inline code.
_RST_ROLE = re.compile(r":[a-zA-Z]+:`~?([^`]+)`")


def clean_md(text: str) -> str:
    """Convert leftover RST cross-reference roles to plain Markdown inline code."""
    return _RST_ROLE.sub(r"`\1`", text)


def page_name(module_name: str) -> str:
    """Wiki page slug for a module, e.g. ``shearnet.core.dataset`` -> ``API-core-dataset``."""
    if module_name == PACKAGE:
        return "API-Overview"
    return "API-" + module_name[len(PACKAGE) + 1 :].replace(".", "-")


def new_renderer() -> MarkdownRenderer:
    """A MarkdownRenderer configured for GitHub-flavoured wiki pages."""
    return MarkdownRenderer(
        render_module_header=False,   # we write our own H1 per page
        insert_header_anchors=True,
        descriptive_class_title=False,
        signature_with_def=True,
        signature_in_header=False,
        code_headers=True,
        render_toc=False,
        add_method_class_prefix=False,
        add_member_class_prefix=False,
    )


def load_modules():
    """Load and process the package, returning the list of docspec modules."""
    context = Context(directory=REPO_ROOT)
    loader = PythonLoader(search_path=[REPO_ROOT], packages=[PACKAGE])
    loader.init(context)
    modules = list(loader.load())

    processors = (
        FilterProcessor(
            documented_only=True,
            exclude_private=True,
            exclude_special=True,
            do_not_filter_modules=True,
            skip_empty_modules=True,
        ),
        SmartProcessor(),       # auto-detect Google/Numpy/plain docstrings
        CrossrefProcessor(),
    )
    for proc in processors:
        proc.init(context)
        proc.process(modules, None)
    return modules


def has_content(module: docspec.Module) -> bool:
    """True if the module has a docstring or at least one documented member."""
    return bool(module.docstring) or bool(getattr(module, "members", None))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "wiki")
    os.makedirs(out_dir, exist_ok=True)

    modules = load_modules()
    by_name = {m.name: m for m in modules}
    renderer = new_renderer()
    renderer.init(Context(directory=REPO_ROOT))

    # --- one page per module ---------------------------------------------
    written = {}  # module_name -> page_name
    for name in sorted(by_name):
        module = by_name[name]
        if not has_content(module):
            continue
        body = renderer.render_to_string([module])
        slug = page_name(name)
        title = "Overview" if name == PACKAGE else f"`{name}`"
        page = f"# {title}\n\n"
        if name != PACKAGE:
            page += f"> Module: `{name}`\n\n"
        page += body.strip() + "\n"
        with open(os.path.join(out_dir, slug + ".md"), "w", encoding="utf-8") as f:
            f.write(clean_md(page))
        written[name] = slug

    # --- grouped index used by Home and Sidebar --------------------------
    grouped = []  # (section_title, [(module_name, slug), ...])
    seen = set()
    for section_title, prefix in SECTIONS:
        entries = [
            (n, written[n])
            for n in sorted(written)
            if n == prefix or n.startswith(prefix + ".")
        ]
        if entries:
            grouped.append((section_title, entries))
            seen.update(n for n, _ in entries)
    leftovers = [(n, written[n]) for n in sorted(written) if n not in seen]
    if leftovers:
        grouped.append(("Other", leftovers))

    # --- _Sidebar.md ------------------------------------------------------
    sidebar = ["## ShearNet API\n", "* [API Reference Home](Home)\n"]
    for section_title, entries in grouped:
        sidebar.append(f"\n**{section_title}**\n")
        for name, slug in entries:
            label = "Overview" if name == PACKAGE else name
            sidebar.append(f"* [{label}]({slug})\n")
    with open(os.path.join(out_dir, "_Sidebar.md"), "w", encoding="utf-8") as f:
        f.write("".join(sidebar))

    # --- Home.md ----------------------------------------------------------
    pkg_doc = (by_name.get(PACKAGE).docstring.content
               if PACKAGE in by_name and by_name[PACKAGE].docstring else "")
    home = ["# ShearNet API Reference\n"]
    if pkg_doc:
        home.append("\n" + clean_md(pkg_doc.strip()) + "\n")
    home.append(
        "\nThis wiki is generated from the in-code docstrings with "
        "[pydoc-markdown](https://niklasrosenstein.github.io/pydoc-markdown/). "
        "Regenerate it with `python scripts/generate_wiki.py`.\n"
    )
    for section_title, entries in grouped:
        home.append(f"\n## {section_title}\n\n")
        for name, slug in entries:
            label = "Overview" if name == PACKAGE else f"`{name}`"
            summary = ""
            mod = by_name[name]
            if mod.docstring and mod.docstring.content:
                summary = " — " + clean_md(mod.docstring.content.strip().splitlines()[0])
            home.append(f"- [{label}]({slug}){summary}\n")
    with open(os.path.join(out_dir, "Home.md"), "w", encoding="utf-8") as f:
        f.write("".join(home))

    print(f"Wrote {len(written)} module pages + Home.md + _Sidebar.md to {out_dir}/")
    for name in sorted(written):
        print(f"  {written[name]}.md  <-  {name}")


if __name__ == "__main__":
    main()
