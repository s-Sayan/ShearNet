"""Worker-count resolution

``os.cpu_count()`` reports the node's cores, not the allocation. Every parallel
path in this package resolves its worker count here.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Iterable, Optional

__all__ = ["cpu_only_children", "resolve_nproc", "slurm_cpus", "spawn_map"]


#: Environment that keeps a child process off the accelerator.
_CPU_ONLY = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}


@contextmanager
def cpu_only_children():
    """Spawn workers with JAX pinned to the CPU, for the duration of the block.

    ``import shearnet.methods.ngmix`` pulls in JAX, and a ``spawn``-ed worker
    performs that import while unpickling the function it was given -- before
    any initializer of ours can run. So an initializer cannot fix this; the
    environment has to be right at spawn time, which means setting it in the
    parent and letting the children inherit it. Eighteen ngmix workers each
    creating a GPU context would otherwise trade a host OOM for a device one.

    The parent's own JAX is unaffected: its context already exists, and these
    variables are only read at initialisation.
    """
    saved = {key: os.environ.get(key) for key in _CPU_ONLY}
    os.environ.update(_CPU_ONLY)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def slurm_cpus(default: int = 1) -> int:
    """CPUs this process was actually allocated.

    ``SLURM_CPUS_PER_TASK`` when running under SLURM, else ``default``. Never
    ``os.cpu_count()``: off-cluster that is the developer's laptop and on a
    shared node it is somebody else's cores.
    """
    try:
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
    except (KeyError, ValueError):
        return max(1, int(default))


def resolve_nproc(nproc: Optional[int] = None, n_tasks: Optional[int] = None) -> int:
    """Number of workers to use.

    ``nproc=None`` means auto: the SLURM allocation, or 1 (serial) off-cluster.
    An explicit integer forces that many. The result is never larger than
    ``n_tasks`` -- spawning eight interpreters to fit three objects costs more
    than it saves, and matters for the tiny runs in the test suite.
    """
    resolved = slurm_cpus() if nproc is None else max(1, int(nproc))
    if n_tasks is not None:
        resolved = max(1, min(resolved, int(n_tasks)))
    return resolved


def spawn_map(fn: Callable, items: Iterable, workers: int, *, initializer=None,
              initargs=(), chunksize: int = 1):
    """``fn`` over ``items`` on ``workers`` spawned processes, in order (an iterator).

    ``concurrent.futures`` rather than ``multiprocessing.Pool``, on purpose. A
    ``Pool`` whose workers die -- killed, or unable to even start because the
    code they import changed on disk under a running job -- quietly starts
    replacements forever, each printing the same traceback, while the map never
    returns. That is how one evaluation once wrote 38 GB of identical
    tracebacks. Here the first dead worker breaks the pool and the job fails.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=initializer, initargs=initargs) as pool:
        try:
            yield from pool.map(fn, items, chunksize=chunksize)
        except BrokenProcessPool as exc:
            raise RuntimeError(
                f"a worker process running {getattr(fn, '__name__', fn)} died; its "
                "traceback is above. If it could not import its code, the checkout "
                "this job runs from changed while it was running.") from exc
