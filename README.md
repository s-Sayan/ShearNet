# ShearNet

A JAX-based neural network library for galaxy **shear estimation**. ShearNet
simulates galaxy images with [GalSim](https://github.com/GalSim-developers/GalSim),
trains neural networks to recover their shear (`g1`, `g2`) and other parameters,
and benchmarks them against traditional moment- and likelihood-based methods
([NGmix](https://github.com/esheldon/ngmix) metacalibration).

---

## Installation

### Quick install (recommended)

```bash
git clone https://github.com/s-Sayan/ShearNet.git
cd ShearNet

make install        # CPU version       -> conda env "shearnet"
# or
make install-gpu    # GPU version (CUDA 12) -> conda env "shearnet_gpu"

conda activate shearnet      # or shearnet_gpu
```

Run `make help` to see all installation targets (`install-dev`, `install-all`,
`clean`, `uninstall`).

### Manual install

```bash
conda create -n shearnet python=3.11
conda activate shearnet
pip install -e .                  # or  pip install -e ".[gpu]"  for GPU
pip install git+https://github.com/esheldon/ngmix.git
python scripts/post_installation.py
```

The post-installation step asks for a directory to store trained models and
plots and writes it to your shell profile as the `SHEARNET_DATA_PATH`
environment variable (see [Data & paths](#data--paths)).

---

## Quick start

```bash
# Train a CNN on 10,000 simulated galaxies
shearnet-train --epochs 10 --batch_size 64 --samples 10000 \
               --psf_fwhm 0.25 --model_name my_cnn --nn cnn --plot

# Evaluate it (optionally comparing against NGmix metacalibration)
shearnet-eval --model_name my_cnn --test_samples 5000 --mcal
```

Or drive everything from a YAML config:

```bash
shearnet-train --config configs/example.yaml
shearnet-eval  --model_name original_high_noise --mcal
```

A tiny end-to-end example used by CI lives in `configs/dry_run.yaml`:

```bash
shearnet-train --config configs/dry_run.yaml
shearnet-eval  --model_name dry_run
```

---

## Command-line interface

### `shearnet-train`

Trains a model and saves the best checkpoint (by validation loss), the training
config, the label normalizer, and a copy of the model architecture under
`$SHEARNET_DATA_PATH`.

Common options:

| Option | Description |
|---|---|
| `--config` | Path to a YAML config (CLI flags override individual values) |
| `--nn` | Architecture: `mlp`, `cnn`, `resnet`, `research_backed`, `forklens_psf`, `fork-like` |
| `--samples` | Number of training galaxies to simulate |
| `--psf_fwhm` | Gaussian PSF FWHM in arcsec (for `--exp ideal`) |
| `--exp` | Simulation mode: `ideal` (analytic PSF) or `superbit` (empirical PSFEx PSF) |
| `--epochs`, `--batch_size`, `--patience` | Training schedule and early stopping |
| `--process_psf` | Feed PSF stamps through a separate branch (implies the `fork-like` model) |
| `--plot` | Save a learning-curve plot |

Run `shearnet-train --help` for the full list.

### `shearnet-eval`

Loads a trained model, regenerates a matching test set, runs predictions, and
prints an MSE/bias/timing summary.

| Option | Description |
|---|---|
| `--model_name` | Name of the model to load (**required**) |
| `--config` | Config to use (defaults to the saved `training_config.yaml`) |
| `--test_samples` | Number of test galaxies |
| `--mcal` | Also run NGmix metacalibration for comparison |
| `--plot` | Save residual/comparison plots |

---

## Configuration

ShearNet uses a layered YAML configuration (`shearnet/config/config_handler.py`):

1. `shearnet/config/default_config.yaml` is always loaded first.
2. A user config passed with `--config` is deep-merged on top.
3. Command-line flags override individual values.

Configs are grouped into `dataset`, `model`, `training`, `evaluation`,
`output`, `plotting`, `comparison`, and `catalog` sections. See
`configs/example.yaml` for a documented template and `configs/` for ready-made
configs (e.g. `configs/shearnet/forklike/...`, `configs/shearnet/old_cnn/...`).

---

## Data & paths

- **`SHEARNET_DATA_PATH`** — where trained models (`model_checkpoint/`) and
  plots (`plots/`) are written. Set automatically by
  `scripts/post_installation.py`, or export it yourself.
- **PSF data** — the empirical SuperBIT PSFs used by `--exp superbit` are
  bundled in `psf_data/`. Override the location with the `SHEARNET_PSF_DIR`
  environment variable if needed.
- **COSMOS catalog** — `dataset` shears/sizes/fluxes can be drawn from a COSMOS
  catalog FITS file (`catalog.cosmos_cat_fname`). If none is provided, ShearNet
  falls back to a synthetic random catalog, so training works out of the box.

---

## Python API

```python
import jax.random as random
from shearnet.core.dataset import generate_dataset
from shearnet.core.train import train_model

# Simulate 10,000 galaxies with a Gaussian PSF (FWHM = 0.25 arcsec)
images, labels = generate_dataset(10000, psf_fwhm=0.25)

# Train a CNN. Single-branch models ignore the psf_images argument, but it must
# still be supplied positionally — pass the galaxy images.
rng_key = random.PRNGKey(42)
state, train_losses, val_losses, val_losses_per_key = train_model(
    images, images, labels, rng_key, epochs=50, nn="cnn",
)
```

Key entry points are re-exported from the top-level package:

```python
from shearnet import generate_dataset, train_model
from shearnet import SimpleGalaxyNN, EnhancedGalaxyNN, GalaxyResNet
```

---

## Example results

ShearNet predicts `g1`, `g2`, `sigma`, and `flux`. Representative performance on
5,000 test galaxies (stamp size 53×53, pixel scale 0.141 arcsec):

| Method                       | MSE (g1, g2) | Time   |
|------------------------------|--------------|--------|
| ShearNet (research backed)   | ~6.75e-6     | ~6.6s  |
| ShearNet (fork-like)         | ~4e-6        | ~2.5s  |
| Moment-based (NGmix)         | ~1e-4        | ~142s  |

---

## Repository structure

```
ShearNet/
├── shearnet/            # The installable package
│   ├── core/            #   models, training loop, dataset simulation
│   ├── methods/         #   NGmix and moment-based baselines
│   ├── utils/           #   metrics, normalization, plotting, device helpers
│   ├── cli/             #   shearnet-train / shearnet-eval entry points
│   └── config/          #   layered YAML config handler + defaults
├── configs/             # Ready-made and example YAML configs
├── psf_data/            # Bundled empirical SuperBIT PSFs (for --exp superbit)
├── notebooks/           # example.ipynb — end-to-end usage walkthrough
├── scripts/             # post-installation setup
├── shear_bias/          # Research scripts for shear-bias / PSF-leakage studies
├── unit_tests/,         # Experiment configs kept as a research record
│   unit_test_variations/
├── makefile             # Installation targets
└── pyproject.toml       # Package metadata and dependencies
```

---

## Requirements

- Python 3.8+
- JAX / jaxlib (CPU or GPU), Flax, Optax, Orbax
- GalSim, NGmix
- NumPy, SciPy, Matplotlib, seaborn, tqdm, PyYAML, numba

See `pyproject.toml` for the complete, pinned list.

---

## License

MIT License — see `LICENSE`.

## Contributing

Contributions are welcome! Please open an issue or pull request.
