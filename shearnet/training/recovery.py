"""Complete training state, published with its history and best checkpoint."""
import time
import hashlib
from flax import serialization
import jax
from ..artifacts.runs import atomic_write


class TrainingRecovery:
    def __init__(self, run, history, best, stop, interval=300):
        self.run, self.history, self.best = run, history, best
        self.stop, self.interval = stop, interval
        self.path = run.training_dir / "latest.msgpack"
        self.previous = run.training_dir / "previous.msgpack"
        self.last = time.monotonic()

    def load(self, state):
        if not self.path.exists() and not self.previous.exists():
            return state, None
        failure = None
        for path in (self.path, self.previous):
            if not path.exists():
                continue
            try:
                saved = self.decode(path.read_bytes())
                restored = serialization.from_bytes(state, saved["state"])
            except Exception as exc:
                failure = exc
                continue
            self.history.records = saved["history"]
            if self.history.records:
                self.history.save()
            self.best.epoch, self.best.sha256 = saved["best_epoch"], saved["best_sha256"]
            if saved["best"] is not None:
                atomic_write(self.run.best_params, saved["best"])
            elif self.run.best_params.exists():
                self.run.best_params.unlink()
            return restored, saved["progress"]
        raise ValueError("both training resume checkpoints are unreadable") from failure

    def due(self):
        return self.stop.requested or time.monotonic() - self.last >= self.interval

    def save(self, state, progress, force=False):
        if not force and not self.due():
            return
        payload = {"state": serialization.to_bytes(state),
                   "progress": jax.device_get(progress), "history": self.history.records,
                   "best_epoch": self.best.epoch, "best_sha256": self.best.sha256,
                   "best": self.run.best_params.read_bytes() if self.best.epoch else None}
        content = serialization.msgpack_serialize(payload)
        data = serialization.msgpack_serialize({"sha256": hashlib.sha256(content).hexdigest(),
                                                "payload": content})
        if self.path.exists():
            # A damaged latest checkpoint must not replace the last valid fallback.
            old = self.path.read_bytes()
            try:
                self.decode(old)
            except Exception:
                pass
            else:
                atomic_write(self.previous, old)
        atomic_write(self.path, data)
        self.last = time.monotonic()
        self.stop.check()

    @staticmethod
    def decode(data):
        envelope = serialization.msgpack_restore(data)
        payload = envelope["payload"]
        if hashlib.sha256(payload).hexdigest() != envelope["sha256"]:
            raise ValueError("training checkpoint checksum failed")
        return serialization.msgpack_restore(payload)
