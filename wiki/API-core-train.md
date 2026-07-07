# `shearnet.core.train`

> Module: `shearnet.core.train`

Core training functions for ShearNet models.

<a id="shearnet.core.train.save_checkpoint"></a>

#### `save_checkpoint`

```python
def save_checkpoint(state, step, checkpoint_dir, model_name, overwrite=True)
```

Save the model checkpoint.

<a id="shearnet.core.train.loss_fn"></a>

#### `loss_fn`

```python
def loss_fn(state, params, images, labels, gap, weights)
```

Weighted mean-squared-error loss for single-branch models.

<a id="shearnet.core.train.fork_loss_fn"></a>

#### `fork_loss_fn`

```python
def fork_loss_fn(state, params, galaxy_images, psf_images, labels, output_keys,
                 gap, weights)
```

Weighted MSE loss for the two-branch ``fork-like`` model.

<a id="shearnet.core.train.loss_fn_per_key"></a>

#### `loss_fn_per_key`

```python
def loss_fn_per_key(state, params, images, labels, gap, weights)
```

Like `loss_fn` but also returns the per-output-key MSE.

<a id="shearnet.core.train.fork_loss_fn_per_key"></a>

#### `fork_loss_fn_per_key`

```python
def fork_loss_fn_per_key(state, params, galaxy_images, psf_images, labels,
                         output_keys, gap, weights)
```

Like `fork_loss_fn` but also returns the per-output-key MSE.

<a id="shearnet.core.train.train_step"></a>

#### `train_step`

```python
@functools.partial(jax.jit, static_argnums=(3, ))
def train_step(state, images, labels, gap, weights)
```

One JIT-compiled gradient-descent step for single-branch models.

<a id="shearnet.core.train.fork_train_step"></a>

#### `fork_train_step`

```python
@functools.partial(jax.jit, static_argnums=(4, 5))
def fork_train_step(state, galaxy_images, psf_images, labels, output_keys, gap,
                    weights)
```

One JIT-compiled gradient-descent step for the ``fork-like`` model.

<a id="shearnet.core.train.eval_step"></a>

#### `eval_step`

```python
@functools.partial(jax.jit, static_argnums=(3, ))
def eval_step(state, images, labels, gap, weights)
```

JIT-compiled validation loss for single-branch models (no gradient step).

<a id="shearnet.core.train.fork_eval_step"></a>

#### `fork_eval_step`

```python
@functools.partial(jax.jit, static_argnums=(4, 5))
def fork_eval_step(state, galaxy_images, psf_images, labels, output_keys, gap,
                   weights)
```

JIT-compiled validation loss for the ``fork-like`` model.

<a id="shearnet.core.train.eval_step_per_key"></a>

#### `eval_step_per_key`

```python
@functools.partial(jax.jit, static_argnums=(3, ))
def eval_step_per_key(state, images, labels, gap, weights)
```

Return validation loss plus per-output-key MSE for single-branch models.

<a id="shearnet.core.train.fork_eval_step_per_key"></a>

#### `fork_eval_step_per_key`

```python
@functools.partial(jax.jit, static_argnums=(4, 5))
def fork_eval_step_per_key(state, galaxy_images, psf_images, labels,
                           output_keys, gap, weights)
```

Return validation loss plus per-output-key MSE for the ``fork-like`` model.

<a id="shearnet.core.train.train_model"></a>

#### `train_model`

```python
def train_model(galaxy_images,
                labels,
                rng_key,
                psf_images=None,
                epochs=10,
                batch_size=32,
                nn="simple",
                galaxy_type="cnn",
                psf_type="cnn",
                save_path=None,
                model_name="my_model",
                val_split=0.2,
                eval_interval=1,
                patience=5,
                lr=1e-3,
                weight_decay=1e-4,
                output_keys=("g1", "g2"),
                gap=False,
                weights=None,
                fusion="concat")
```

Train a ShearNet model with validation and early stopping.

Builds the requested architecture, trains it with an AdamW optimizer and a
warmup + cosine-decay learning-rate schedule, and (if ``save_path`` is given)
saves only the best checkpoint by validation loss — the checkpoint on disk is
always the best epoch, never the final one.

**Arguments**:

- `galaxy_images` - Galaxy stamps, shape ``(N, npix, npix)``.
- `labels` - Targets, shape ``(N, len(output_keys))``.
- `rng_key` - A ``jax.random.PRNGKey`` for parameter init and shuffling.
- `psf_images` - PSF stamps, shape ``(N, npix, npix)``. Required only for the
  ``fork-like`` architecture; ignored (and optional) otherwise.
- `epochs` - Maximum number of training epochs.
- `batch_size` - Mini-batch size.
- `nn` - Architecture name — one of ``'mlp'``, ``'cnn'``, ``'resnet'``,
  ``'research_backed'``, ``'forklens_psfnet'``, or ``'fork-like'``.
  galaxy_type, psf_type: Sub-model types for the two ``fork-like`` branches.
- `save_path` - Directory to write the best checkpoint to (no save if ``None``).
- `model_name` - Checkpoint filename prefix.
- `val_split` - Fraction of the data held out for validation.
- `eval_interval` - Validate every this many epochs.
- `patience` - Stop after this many evals without validation improvement.
- `lr` - Peak learning rate.
- `weight_decay` - AdamW weight decay.
- `output_keys` - Names of the predicted parameters.
- `gap` - Use global-average-pooling in the model head where supported.
- `weights` - Optional per-key loss weights (defaults to all ones).
- `fusion` - ``fork-like`` fusion strategy, ``'concat'`` or ``'transformer'``.
  

**Returns**:

  ``(state, train_losses, val_losses, val_losses_per_key)`` where ``state``
  is the final ``TrainState`` and the remaining items are per-epoch loss
  histories.
