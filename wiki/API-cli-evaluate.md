# `shearnet.cli.evaluate`

> Module: `shearnet.cli.evaluate`

Command-line interface for evaluating trained ShearNet models.

Loads a saved checkpoint, regenerates a matching test set, runs the network
(optionally alongside NGmix metacalibration for comparison), and prints an
MSE/bias/timing summary.

<a id="shearnet.cli.evaluate.create_parser"></a>

#### `create_parser`

```python
def create_parser()
```

Build the ``shearnet-eval`` argument parser.

<a id="shearnet.cli.evaluate.load_config"></a>

#### `load_config`

```python
def load_config(args)
```

Resolve evaluation settings from CLI args and the saved training config.

Falls back to the ``training_config.yaml`` written next to the model's plots
(so evaluation matches how the model was trained), then to package defaults.

**Returns**:

- `dict` - A flat settings dictionary consumed by the rest of the pipeline.

<a id="shearnet.cli.evaluate.generate_test_data"></a>

#### `generate_test_data`

```python
def generate_test_data(config)
```

Simulate the test set, returning ``(gal_images, psf_images, labels, obs)``.

<a id="shearnet.cli.evaluate.load_model"></a>

#### `load_model`

```python
def load_model(config, gal_images, psf_images)
```

Instantiate the configured architecture and restore its best checkpoint.

**Returns**:

- `flax.training.train_state.TrainState` - State with the restored parameters.

<a id="shearnet.cli.evaluate.run_shearnet"></a>

#### `run_shearnet`

```python
def run_shearnet(state,
                 gal_images,
                 psf_images,
                 labels,
                 config,
                 batch_size=128)
```

Batch-predict with the network and report MSE/bias/time.

Predictions are inverse-transformed with the saved label normalizer (if
present) before metrics are computed.

**Returns**:

- `tuple` - ``(preds, mse, bias, elapsed_seconds)``.

<a id="shearnet.cli.evaluate.run_ngmix"></a>

#### `run_ngmix`

```python
def run_ngmix(obs, labels, config)
```

Run NGmix metacalibration on the same observations for comparison.

**Returns**:

- `tuple` - ``(preds, mse, bias, elapsed_seconds)`` over the non-NaN fits.

<a id="shearnet.cli.evaluate.print_summary"></a>

#### `print_summary`

```python
def print_summary(config,
                  sn_mse,
                  sn_bias,
                  sn_time,
                  ngmix_mse=None,
                  ngmix_bias=None,
                  ngmix_time=None)
```

Print the final evaluation summary, including the NGmix speedup if run.

<a id="shearnet.cli.evaluate.main"></a>

#### `main`

```python
def main()
```

Entry point for the ``shearnet-eval`` command.
