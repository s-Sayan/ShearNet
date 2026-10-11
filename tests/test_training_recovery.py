"""Resumed optimizers and RNG streams follow the uninterrupted trajectory."""
import numpy as np
import pytest
import jax
from shearnet.artifacts import RunDir
from shearnet.artifacts.recovery import Interrupted
from shearnet.core.train import train_model
from shearnet.training.history import History
from shearnet.training.service import _Checkpointer
from shearnet.training.recovery import TrainingRecovery


class StopAfterSave:
    requested = False
    def check(self):
        raise Interrupted("test interruption")


class Continue:
    requested = False
    def check(self):
        pass


def recovery(root, stop):
    run = RunDir(root)
    if not run.root.exists():
        run.create()
    history = History(("g1", "g2"), run.history_csv, run.history_npz)
    best = _Checkpointer(run)
    return TrainingRecovery(run, history, best, stop, interval=0)


@pytest.mark.parametrize("ema", [None, .9])
def test_mid_batch_resume_is_exact(tmp_path, ema):
    images = np.random.RandomState(3).normal(size=(32, 9, 9)).astype("f4")
    labels = np.random.RandomState(4).normal(size=(32, 2)).astype("f4")
    kw = dict(epochs=2, batch_size=8, nn="mlp", ema_decay=ema,
              resample_noise=True, resample_noise_sd=.01)
    reference, *_ = train_model(images, labels, jax.random.PRNGKey(5), **kw)
    saved = recovery(tmp_path / "run", StopAfterSave())
    with pytest.raises(Interrupted):
        train_model(images, labels, jax.random.PRNGKey(5), recovery=saved,
                    checkpoint_fn=saved.best, history_fn=saved.history, **kw)
    saved = recovery(tmp_path / "run", Continue())
    resumed, *_ = train_model(images, labels, jax.random.PRNGKey(5), recovery=saved,
                             checkpoint_fn=saved.best, history_fn=saved.history, **kw)
    for x, y in zip(jax.tree_util.tree_leaves(reference), jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_array_equal(x, y)
    assert len(saved.history.records) == 2
    # A second resume after the final epoch does no extra optimizer steps.
    repeat = recovery(tmp_path / "run", Continue())
    finished, *_ = train_model(images, labels, jax.random.PRNGKey(5), recovery=repeat,
                              checkpoint_fn=repeat.best, history_fn=repeat.history, **kw)
    assert int(finished.step) == int(reference.step)


def test_inloop_resume_restores_noise_shuffle_and_ema(tmp_path):
    from shearnet.core.dataset_jax import JaxRenderConfig, sample_truth
    from shearnet.core.inloop import InLoopGenerator
    from shearnet.core.train_inloop import train_model_inloop
    cfg = JaxRenderConfig(npix=21, scale=.141, fft_size=64, batch_size=2)
    gen = InLoopGenerator(sample_truth(8, cfg, seed=3, add_noise=False), cfg, 2)
    kw = dict(epochs=2, nn="mlp", val_split=.25, nse_sd=1., ema_decay=.9,
              output_keys=("g1", "g2"), eval_interval=2)
    reference, *_ = train_model_inloop(gen, jax.random.PRNGKey(5), **kw)
    saved = recovery(tmp_path / "inloop", StopAfterSave())
    with pytest.raises(Interrupted):
        train_model_inloop(gen, jax.random.PRNGKey(5), recovery=saved,
                           checkpoint_fn=saved.best, history_fn=saved.history, **kw)
    saved = recovery(tmp_path / "inloop", Continue())
    resumed, *_ = train_model_inloop(gen, jax.random.PRNGKey(5), recovery=saved,
                                    checkpoint_fn=saved.best, history_fn=saved.history, **kw)
    for x, y in zip(jax.tree_util.tree_leaves(reference), jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_array_equal(x, y)


def test_corrupt_latest_checkpoint_uses_previous(tmp_path):
    saved = recovery(tmp_path / "run", Continue())
    state = {"step": np.array(1), "rng": np.array([2, 3], dtype="uint32")}
    saved.save(state, {"epoch": 0}, force=True)
    previous = saved.path.read_bytes()
    state["step"] = np.array(2)
    saved.save(state, {"epoch": 1}, force=True)
    assert saved.previous.read_bytes() == previous
    saved.path.write_bytes(b"truncated write")
    restored, progress = saved.load(state)
    assert int(restored["step"]) == 1 and progress["epoch"] == 0
    saved.save(restored, progress, force=True)
    assert saved.previous.read_bytes() == previous
    saved.previous.write_bytes(b"corrupt fallback")
    saved.path.write_bytes(b"corrupt latest")
    with pytest.raises(ValueError, match="both training resume checkpoints"):
        saved.load(state)
