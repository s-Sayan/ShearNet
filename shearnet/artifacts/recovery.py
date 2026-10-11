"""Atomic checkpoints, exclusive writers, and cooperative interruption."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal


class Interrupted(RuntimeError):
    """The allocation is ending; durable work can be resumed."""


class StopRequest:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        def request(signum, frame):
            self.requested = True
        for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            self.previous[sig] = signal.signal(sig, request)
        return self

    def check(self):
        if self.requested:
            raise Interrupted("interrupted; resume this stage to continue")

    def __exit__(self, *args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another process owns {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
