"""scripts/shearnet.sbatch, run with bash and stub shearnet-train / shearnet-eval."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "shearnet.sbatch"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

STUB = """#!/bin/bash
echo "$(basename "$0") $*" >> "$STUB_LOG"
if [[ "$(basename "$0")" == shearnet-train && -n "${STUB_TRAIN_FAILS:-}" ]]; then exit 3; fi
"""


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("shearnet-train", "shearnet-eval"):
        stub = bin_dir / name
        stub.write_text(STUB)
        stub.chmod(0o755)
    log = tmp_path / "calls.log"
    environ = {k: v for k, v in os.environ.items()
               if not k.startswith(("SLURM_", "SHEARNET_"))}
    environ.update(PATH=f"{bin_dir}{os.pathsep}{environ['PATH']}", STUB_LOG=str(log),
                   SLURM_SUBMIT_DIR=str(tmp_path))
    return environ, log, tmp_path


def run(environ, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=environ,
                          capture_output=True, text=True)


def calls(log):
    return log.read_text().splitlines() if log.exists() else []


def test_train_then_eval(env):
    environ, log, _ = env
    out = run(environ, "cfg.yaml", "runs/a")
    assert out.returncode == 0, out.stderr
    assert calls(log) == ["shearnet-train --config cfg.yaml --run runs/a",
                          "shearnet-eval --run runs/a"]


def test_run_dir_defaults_to_the_config_outdir(env):
    environ, log, tmp = env
    (tmp / "cfg.yaml").write_text("run_options:\n  run_name: a\n  outdir: runs/a\n")
    out = run(environ, "cfg.yaml")
    assert out.returncode == 0, out.stderr
    assert calls(log)[0] == f"shearnet-train --config cfg.yaml --run {tmp / 'runs' / 'a'}"


def test_training_failure_skips_the_evaluation(env):
    environ, log, _ = env
    out = run(dict(environ, STUB_TRAIN_FAILS="1"), "cfg.yaml", "runs/a")
    assert out.returncode == 3
    assert calls(log) == ["shearnet-train --config cfg.yaml --run runs/a"]


def test_train_only_and_eval_only(env):
    environ, log, _ = env
    assert run(environ, "--train-only", "cfg.yaml", "runs/a").returncode == 0
    assert run(environ, "--eval-only", "runs/a").returncode == 0
    assert run(environ, "--eval-only", "runs/a", "deep.yaml", "deep").returncode == 0
    assert calls(log) == ["shearnet-train --config cfg.yaml --run runs/a",
                          "shearnet-eval --run runs/a",
                          "shearnet-eval --run runs/a --config deep.yaml --eval-name deep"]
    # an override without a name is refused before anything runs
    assert run(environ, "--eval-only", "runs/a", "deep.yaml").returncode != 0
    assert len(calls(log)) == 3


def test_list_picks_the_array_task_line(env):
    environ, log, tmp = env
    (tmp / "runs.txt").write_text("# header\n\none.yaml runs/one\n"
                                  "  # indented comment\n--eval-only runs/two\n")
    out = run(dict(environ, SLURM_ARRAY_TASK_ID="1"), "--list", "runs.txt")
    assert out.returncode == 0, out.stderr
    assert calls(log) == ["shearnet-eval --run runs/two"]
    assert run(dict(environ, SLURM_ARRAY_TASK_ID="2"), "--list", "runs.txt").returncode != 0
    assert run(environ, "--list", "runs.txt").returncode != 0  # not an array job


def test_refuses_without_the_environment(env):
    environ, log, _ = env
    bare = "/usr/bin:/bin"
    if shutil.which("shearnet-train", path=bare) or shutil.which("shearnet-eval", path=bare):
        pytest.skip("shearnet is installed system-wide")
    out = run(dict(environ, PATH=bare), "cfg.yaml", "runs/a")
    assert out.returncode != 0 and not calls(log)
    assert "activate the environment" in out.stderr


def test_thread_and_x64_defaults_are_exported(env):
    environ, log, tmp = env
    (tmp / "bin" / "shearnet-train").write_text(
        '#!/bin/bash\necho "$OMP_NUM_THREADS $OPENBLAS_NUM_THREADS $JAX_ENABLE_X64" >> "$STUB_LOG"\n')
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "JAX_ENABLE_X64"):
        environ.pop(name, None)
    assert run(environ, "--train-only", "cfg.yaml", "runs/a").returncode == 0
    assert calls(log) == ["1 1 1"]
    # an explicit setting wins
    assert run(dict(environ, OMP_NUM_THREADS="4"), "--train-only", "cfg.yaml",
               "runs/a").returncode == 0
    assert calls(log)[-1] == "4 1 1"


def test_extra_args_pass_through(env):
    environ, log, _ = env
    out = run(dict(environ, SHEARNET_TRAIN_ARGS="--overwrite -v",
                   SHEARNET_EVAL_ARGS="--eval-name x"), "cfg.yaml", "runs/a")
    assert out.returncode == 0, out.stderr
    assert calls(log) == ["shearnet-train --config cfg.yaml --run runs/a --overwrite -v",
                          "shearnet-eval --run runs/a --eval-name x"]


def test_usage_is_the_header(env):
    environ, log, _ = env
    out = run(environ, "--help")
    assert out.returncode == 1 and not calls(log)
    assert out.stdout.startswith("The one Slurm script")
    assert "--list" in out.stdout and "SHEARNET_EVAL_ARGS" in out.stdout


def test_paper_run_list_lines_are_real_configs():
    lines = [line.split() for line in (REPO / "configs/paper/runs.txt").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    assert len(lines) == 4
    for fields in lines:
        assert len(fields) == 1 and (REPO / fields[0]).is_file(), fields
