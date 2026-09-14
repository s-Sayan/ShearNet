"""The timing pass must record ngmix, or tab:timing cannot be filled.

``_timing_pass`` measured the render and ShearNet's inference only, so
``NGMIX_SE`` was absent from every evaluation FITS ever written. The paper's
timing table compares GPU inference against the ngmix CPU pool and quotes the
ratio, so its second row and its ratio were not pending a measurement -- they
were pending one the harness never took, which no amount of re-measuring would
have produced.

There is no flag to enable it: the ngmix leg runs whenever ngmix is among the
estimators. These pin that, and pin the header key the plotting repo reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "shear_bias"))


class _Stamps:
    def __init__(self, n):
        self.galaxy_images = np.zeros((n, 5, 5))
        self.psf_images = np.zeros((n, 5, 5))
        self.labels = np.zeros((n, 4))


class _Renderer:
    def render(self, samples, **kwargs):
        return _Stamps(samples)


def _timing():
    from run import _timing_pass

    return _timing_pass


def test_ngmix_is_timed_when_it_is_an_estimator():
    calls = []

    def ngmix_measure(galaxy, psf):
        calls.append(len(galaxy))
        return np.zeros((len(galaxy), 2))

    result = _timing()(_Renderer(), None, samples=32, seed=1, batch=8,
                       measures={"ngmix": ngmix_measure})

    assert "ngmix_seconds" in result, (
        "no ngmix timing recorded; tab:timing's second row and its ratio "
        "cannot be filled from the resulting FITS"
    )
    assert result["ngmix_seconds"] >= 0.0
    # Timed on the whole population, once -- the same stamps the network saw.
    assert calls == [32]


def test_no_ngmix_leg_when_ngmix_is_not_measured():
    result = _timing()(_Renderer(), None, samples=16, seed=1, batch=8,
                       measures={"anacal": lambda g, p: np.zeros((len(g), 2))})
    assert "ngmix_seconds" not in result


def test_measures_is_optional():
    """Older callers pass no measures at all; they must not break."""
    result = _timing()(_Renderer(), None, samples=8, seed=1, batch=4)
    assert "render_seconds" in result
    assert "ngmix_seconds" not in result


def test_the_call_site_passes_measures():
    """The keyword has to actually be threaded, not merely accepted."""
    import ast

    source = (REPO / "research/shear_bias/run.py").read_text()
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_timing_pass"
    ]
    assert calls, "_timing_pass is never called"
    for call in calls:
        assert any(k.arg == "measures" for k in call.keywords), (
            f"_timing_pass call at line {call.lineno} does not pass measures, "
            "so ngmix is silently not timed"
        )


def test_ngmix_seconds_reaches_the_fits_header():
    """The header loop is an explicit allow-list; a key not in it is dropped."""
    source = (REPO / "research/shear_bias/run.py").read_text()
    header_block = source[source.index('for key in ("backend", "generation"'):]
    header_block = header_block[: header_block.index("):")]
    assert '"ngmix_seconds"' in header_block, (
        "ngmix_seconds is measured but never written to the primary header"
    )


def test_the_header_key_is_the_one_the_plots_repo_reads():
    """FITS keywords truncate to 8 characters; NGMIX_SE must not collide."""
    keys = {}
    for name in ("ngmix_nproc", "ngmix_seconds", "render_seconds",
                 "inference_seconds"):
        keys.setdefault(name[:8].upper(), []).append(name)
    collisions = {k: v for k, v in keys.items() if len(v) > 1}
    assert not collisions, f"truncated header keys collide: {collisions}"
    assert "ngmix_seconds"[:8].upper() == "NGMIX_SE"
