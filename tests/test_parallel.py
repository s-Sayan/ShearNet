"""shearnet.parallel.spawn_map: ordered results, and a dead worker fails the job."""

import os
import subprocess
import sys
import textwrap

import pytest

from shearnet.parallel import spawn_map


def _square(x):
    return x * x


def _die():
    os._exit(3)


def test_results_come_back_in_order():
    assert list(spawn_map(_square, range(50), 3, chunksize=4)) == [x * x for x in range(50)]


def test_a_dying_worker_raises_instead_of_respawning():
    with pytest.raises(RuntimeError, match="worker process"):
        list(spawn_map(_square, range(10), 2, initializer=_die))


def test_checkout_changing_under_a_running_job(tmp_path):
    """The script a job runs disappears after it starts (a git checkout did this).

    Spawned workers re-run the main script by path, so none of them can start.
    multiprocessing.Pool replaced them forever and wrote 38 GB of tracebacks;
    this has to fail, promptly, with a handful of them.
    """
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent("""
        import os, sys
        from shearnet.parallel import spawn_map

        def square(x):
            return x * x

        if __name__ == "__main__":
            os.remove(sys.argv[0])
            try:
                list(spawn_map(square, range(100), 4))
            except RuntimeError as err:
                print("FAILED CLEANLY:", err)
                sys.exit(7)
    """))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            timeout=120, cwd=tmp_path)
    assert result.returncode == 7, result.stdout + result.stderr[-2000:]
    assert "FAILED CLEANLY" in result.stdout
    assert result.stderr.count("FileNotFoundError") <= 8
