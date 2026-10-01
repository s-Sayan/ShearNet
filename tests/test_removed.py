"""Things that were taken out stay out."""

import importlib.util
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The legacy-config translator has to name the keys it drops.
ALLOWED = {REPO / "shearnet" / "config" / "legacy.py"}


def _sources():
    for top in ("shearnet", "configs", "scripts", "research", "docs"):
        for path in (REPO / top).rglob("*"):
            if path.suffix in (".py", ".yaml", ".md", ".sbatch", ".toml") and path.is_file():
                yield path
    yield REPO / "pyproject.toml"
    yield REPO / "README.md"


def test_no_anacal_or_fpfs():
    pattern = re.compile(r"anacal|fpfs", re.IGNORECASE)
    hits = [str(p.relative_to(REPO)) for p in _sources()
            if p not in ALLOWED and pattern.search(p.read_text(errors="ignore"))]
    assert not hits, hits
    for module in ("shearnet.methods.anacal", "shearnet.methods.anacal_fit"):
        assert importlib.util.find_spec(module) is None, module


def test_no_old_harness():
    assert importlib.util.find_spec("shearnet.benchmarking") is None
    for path in ("research/shear_bias/run.py", "scripts/post_installation.py",
                 "shearnet/config/default_config.yaml"):
        assert not (REPO / path).exists(), path
