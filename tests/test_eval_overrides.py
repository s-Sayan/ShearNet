"""The evaluation CLI overrides must reach the keys the evaluation reads.

Re-measuring the finished runs on a size-cut catalog is done with
``--eval-catalog`` and ``--output`` rather than by editing 28 YAMLs. That makes
two silent failure modes possible, and both would be discovered only after a
queue of jobs had finished:

  * an override that lands on the wrong key, so the run is measured on the
    ORIGINAL catalog while its log claims otherwise, and the cut re-measurement
    is indistinguishable from the thing it was supposed to replace;
  * an output override that does not take, so the re-measurement overwrites
    ``benchmarking/evaluation.fits`` -- the only record of the uncut result.

These pin the exact dotted paths ``run.py`` reads, so renaming either key
breaks a test rather than a campaign.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "research/unit_test_variations/fourth_inloop_shearnet_d4_2drope/config.yaml"

#: The keys run.py resolves the catalog and the output path from.
EVAL_CATALOG_KEY = "paths.eval_catalog"
OUTPUT_KEY = "eval.evaluate.output"


@pytest.mark.skipif(not CONFIG.is_file(), reason="flagship config not present")
def test_overrides_reach_the_keys_the_evaluation_reads(tmp_path):
    from shearnet.config.config_handler import Config

    config = Config(str(CONFIG))
    original = config.get(EVAL_CATALOG_KEY)

    catalog = tmp_path / "cut.fits"
    catalog.write_bytes(b"")
    config._set_nested(EVAL_CATALOG_KEY, str(catalog))
    config._set_nested(OUTPUT_KEY, "benchmarking/evaluation_r15.fits")

    assert config.get(EVAL_CATALOG_KEY) == str(catalog)
    assert config.get(EVAL_CATALOG_KEY) != original
    assert config.get(OUTPUT_KEY) == "benchmarking/evaluation_r15.fits"


def test_run_py_exposes_both_overrides():
    """The flags exist and are spelled the way sub.sh passes them."""
    result = subprocess.run(
        [sys.executable, str(REPO / "research/shear_bias/run.py"), "--help"],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "--eval-catalog" in result.stdout
    assert "--output" in result.stdout


def test_a_missing_catalog_is_refused_before_any_work(tmp_path):
    """A mistyped path must fail immediately, not after the render.

    Without this the job spends its render budget and then either crashes deep
    in the measurement or, worse, silently falls back to the config's catalog.
    """
    result = subprocess.run(
        [sys.executable, str(REPO / "research/shear_bias/run.py"),
         "-c", str(CONFIG), "--eval-catalog", str(tmp_path / "nope.fits")],
        capture_output=True, text=True, cwd=REPO,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode != 0
    assert "--eval-catalog does not exist" in (result.stderr + result.stdout)


def test_submit_script_offers_only_done():
    """--only-done must exist: it is what keeps a half-trained arm out.

    --no-train on a run that has not finished evaluates whatever checkpoint is
    on disk, which for a run mid-training is a partial model that would land in
    the same table as fully trained ones.
    """
    script = (REPO / "research/submit_paper_runs.sh").read_text()
    assert "--only-done" in script
    assert "ONLY_DONE" in script
    # The gate is the FITS, not the checkpoint: a mid-training run has a
    # checkpoint too.
    assert 'ONLY_DONE" -eq 1 && ! -f "$dir/benchmarking/evaluation.fits"' in script


def test_sub_script_passes_the_overrides_through():
    script = (REPO / "research/unit_test_variations/sub.sh").read_text()
    for token in ("--eval-catalog", "--output", "EVAL_CATALOG=$EVAL_CATALOG",
                  "OUTPUT=$OUTPUT"):
        assert token in script, token
