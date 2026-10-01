"""The generated parts of the docs match the code."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_generated_docs_are_current():
    result = subprocess.run([sys.executable, str(REPO / "scripts" / "make_docs.py"), "--check"],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_config_key_is_documented():
    from shearnet.config.schema import FIELDS

    text = (REPO / "docs" / "config.md").read_text()
    for key in FIELDS:
        assert f"`{key.split('.', 1)[1]}`" in text, key
