"""No undefined name reaches a submitted job.

`shearnet-train` died with `NameError: name 'is_fork_model' is not defined` on
its first real run: the call sat inside `_run_inloop_training`, which no unit
test executes, so importing the module proved nothing. Three jobs burned a
scheduler slot to discover a one-line import.

pyflakes decides this statically in under a second, so it is a test rather than
a habit.
"""

import io
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pyflakes_api = pytest.importorskip("pyflakes.api")
from pyflakes.reporter import Reporter  # noqa: E402


def _undefined_names(root: Path):
    found = []
    for module in sorted(root.rglob("*.py")):
        out, err = io.StringIO(), io.StringIO()
        pyflakes_api.checkPath(str(module), Reporter(out, err))
        for line in out.getvalue().splitlines():
            # The real diagnostic is "undefined name 'x'". Not to be confused
            # with the star-import advisory "unable to detect undefined names",
            # which reports pyflakes' own limits rather than a defect.
            if "undefined name '" in line:
                found.append(line.replace(str(REPO) + "/", ""))
    return found


def test_the_package_has_no_undefined_names():
    problems = _undefined_names(REPO / "shearnet")
    assert not problems, "\n".join(problems)


def test_the_research_scripts_have_no_undefined_names():
    """These are what the SLURM jobs actually invoke."""
    problems = _undefined_names(REPO / "research")
    assert not problems, "\n".join(problems)
