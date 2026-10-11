"""Recovery accepts committed shards and rejects damaged results."""
import numpy as np
import pytest
from shearnet.evaluation.journal import Journal
from shearnet.io.catalog_schema import Column
from shearnet.evaluation.workers import Workers

TABLES = {name: [Column("record_id", "i8"), Column("measurement", "f8")]
          for name in ("TRUTH", "STAMP")}


def row(index):
    return {name: {"record_id": index, "measurement": np.nan} for name in TABLES}


def test_resume_ignores_uncommitted_writes_and_keeps_flagged_rows(tmp_path):
    journal = Journal(tmp_path, {"seed": 3}, TABLES, 3)
    journal.commit([row(1)])
    (tmp_path / "00000001.writing").write_bytes(b"interrupted write")
    resumed = Journal(tmp_path, {"seed": 3}, TABLES, 3, resume=True)
    assert resumed.done.tolist() == [False, True, False]
    resumed.commit([row(2)])
    assert [p.name for p in resumed.paths] == ["00000000.fits", "00000001.fits"]
    with pytest.raises(ValueError, match="completed record"):
        resumed.commit([row(1)])
    with pytest.raises(ValueError, match="resume inputs"):
        Journal(tmp_path, {"seed": 4}, TABLES, 3, resume=True)


def test_corrupt_committed_shard_is_not_silently_recomputed(tmp_path):
    journal = Journal(tmp_path, {}, TABLES, 1)
    journal.commit([row(0)])
    path = journal.paths[0]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="invalid FITS checksum"):
        Journal(tmp_path, {}, TABLES, 1, resume=True)


def test_dead_worker_raises_without_waiting_for_a_missing_result():
    class Stop:
        def check(self):
            pass
    with Workers(1) as pool:
        pool.processes[0].kill()
        pool.processes[0].join(timeout=5)
        with pytest.raises(RuntimeError, match="worker exited"):
            list(pool.map([(0, None)], Stop()))
