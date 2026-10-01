# ShearNet

A JAX-based neural network library for galaxy **shear estimation**. ShearNet
simulates galaxy images with [GalSim](https://github.com/GalSim-developers/GalSim)
(or its differentiable port, JAX-GalSim), trains neural networks to recover their
shape (`g1`, `g2`) and other parameters, and measures them side by side with
[NGmix](https://github.com/esheldon/ngmix) and its metacalibration.

Two commands, one config, one directory per run:

```bash
shearnet-train --config my.yaml --run runs/my    # train; the run directory holds everything
shearnet-eval  --run runs/my                     # measure; writes one raw FITS catalog
```

---

## Installation

```bash
git clone https://github.com/s-Sayan/ShearNet.git
cd ShearNet

make install        # CPU version       -> conda env "shearnet"
# or
make install-gpu    # GPU version (CUDA 12) -> conda env "shearnet_gpu"

conda activate shearnet      # or shearnet_gpu
```

Run `make help` for the other targets (`install-dev`, `install-all`, `clean`,
`uninstall`). With a separately managed environment:

```bash
pip install -e .                  # or  pip install -e ".[gpu]"  for GPU
pip install git+https://github.com/esheldon/ngmix.git
pip install "git+https://github.com/AdamField118/JAX-GalSim@0213d84"
```

Neither ngmix nor the JAX-GalSim fork is on PyPI. The fork (it adds the
`des.DES_PSFEx` SuperBIT PSFs) is needed for in-loop training and for
`shearnet-eval`.

### Verify the installation

The smoke config trains a tiny model for two epochs and evaluates it, the same
thing CI does:

```bash
shearnet-train --config configs/smoke.yaml --run runs/smoke
shearnet-eval  --run runs/smoke
```

---

## Training: `shearnet-train`

```bash
shearnet-train --config CONFIG [--run DIR] [--overwrite] [--dry-run] [-q | -v]
```

Everything about a run is in its config; there are no per-setting flags.
`--run` is the run directory (default: the config's `run_options.outdir`).
`--dry-run` validates the config and its input files and stops. A run directory
that already holds something is refused unless `--overwrite` is given.

The run directory is self-contained:

```
runs/my/
├── config.input.yaml        # the config exactly as given
├── config.resolved.yaml     # every setting, defaults filled in -- what was trained
├── manifest.json            # checkpoint (epoch, sha256), best val loss, provenance
├── status.json              # pending / running / completed / failed
├── model/best.msgpack       # the best-validation weights
├── normalizers/             # labels.npz (and images.npz with normalize_images)
├── training/                # history.csv, history.npz, learning_curve.png
├── logs/train.log
└── evaluations/<name>/      # one per shearnet-eval
```

## Evaluation: `shearnet-eval`

```bash
shearnet-eval --run DIR [--config OVERRIDE.yaml] [--eval-name NAME] [--overwrite] [--dry-run]
```

Measures the run's model with the run's own `evaluation` settings: it renders
every scene in `evaluation.scenes` (no shear and ±0.01 on g1 and on g2 by
default) at every ring station in `evaluation.rotations_deg`, all from the same
galaxies, PSFs and noise, and measures each stamp with every estimator in
`evaluation.estimators`. ShearNet runs on the original stamp and on all nine
metacal images ngmix fits; ngmix fits the original stamp and runs metacal.

The result is one FITS file, `DIR/evaluations/<name>/<run_name>_<name>.fits`,
with `TRUTH`, `STAMP`, `SHEARNET` and `NGMIX` tables that line up row for row.
It holds **raw measurements only** -- no response, bias, leakage, selection or
correction; those are computed from it downstream. Every column is described in
[`docs/catalog.md`](docs/catalog.md) (and in the file's own `SCHEMA` HDU).

An override config may change only `evaluation.*`,
`simulation.catalogs.eval_file` and `run_options.ncores`, and is saved under its
own name, so one model can be measured several ways:

```bash
shearnet-eval --run runs/my --config deep.yaml --eval-name deep
```

## Configuration

One YAML file with five blocks: `run_options`, `simulation`, `model`, `training`,
`evaluation`. Every key, its type, default and meaning is in
[`docs/config.md`](docs/config.md), generated from `shearnet/config/schema.py`.

* A key that is not in the schema is an **error** (with a suggestion), as is a
  setting the chosen architecture would never read.
* Anything left out takes its default; `config.resolved.yaml` records them all.
* Relative paths resolve against the config file's directory.

`configs/example.yaml` is a short commented tour, `configs/smoke.yaml` the tiny
CI run, and `configs/paper/` the paper campaign (generated from one fiducial; see
[`configs/paper/README.md`](configs/paper/README.md)). Configs in the package and unit-test
layouts are translated with a warning per changed key;
`python -m shearnet.config.legacy INPUT.yaml` prints the translation.

## On a Slurm cluster

`scripts/shearnet.sbatch` is the one batch script. From the repository root:

```bash
sbatch scripts/shearnet.sbatch CONFIG [RUN_DIR]                       # train, then evaluate
sbatch scripts/shearnet.sbatch --train-only CONFIG [RUN_DIR]
sbatch scripts/shearnet.sbatch --eval-only RUN_DIR [EVAL_CONFIG EVAL_NAME]
sbatch --array=0-27%4 scripts/shearnet.sbatch --list configs/paper/runs.txt
```

It sources `$SHEARNET_ENV` (or `./setup_env.sh`) for the environment and sets
`JAX_ENABLE_X64=1`. The header of the script documents the rest.
`research/hyperparam_search/` writes sweep configs and a run list for the same
script.

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_quickstart.ipynb` | Simulate → train → evaluate → plot, end to end, in memory. |
| `02_model_comparison.ipynb` | Compare evaluated runs: curves, tables, residuals, NGmix on the same stamps. |
| `03_catalog_builder.ipynb` | Build train/eval FITS catalogs from COSMOS / detection data. |
| `04_psf_diagnostics.ipynb` | Inspect PSFs and measure a run's PSF leakage. |

See [`notebooks/README.md`](notebooks/README.md).

## Python API

```python
import jax.random as random
from shearnet.core.dataset import generate_dataset
from shearnet.core.train import train_model

# Simulate 10,000 galaxies with a Gaussian PSF (FWHM = 0.25 arcsec)
images, labels = generate_dataset(10000, psf_fwhm=0.25)

# Train a CNN. Single-branch models take just the galaxy images; psf_images is
# only needed for the two-branch "fork-like" architectures.
rng_key = random.PRNGKey(42)
state, train_losses, val_losses, val_losses_per_key = train_model(
    images, labels, rng_key, epochs=50, nn="cnn",
)
```

A trained run as a predictor:

```python
from shearnet.artifacts import RunDir
from shearnet.evaluation.predictor import RunPredictor

predict = RunPredictor(RunDir("runs/my"))
preds = predict(galaxy_stamps, psf_stamps)    # (N, len(output_keys)), physical units
```

Every public module and function has a docstring; `help(...)` reads them.

---

## Data & paths

- **PSF data** — the SuperBIT PSFEx models used by `simulation.psf.mode: superbit`
  are bundled in `psf_data/` (`simulation.psf.psfex_file: null` uses them).
  `SHEARNET_PSF_DIR` overrides the bundled location.
- **Catalogs** — `simulation.catalogs.train_file` / `eval_file` are FITS
  catalogs with `G1`, `G2`, `HLR`, `FLUX` columns; build them with
  `03_catalog_builder.ipynb`. The two must differ (row *i* is object *i*). With
  no training catalog a synthetic population is drawn, which is only meant for
  tests and the smoke run.

## Example results

ShearNet predicts `g1` and `g2` by default (configurable via `output_keys`, e.g.
to also recover `hlr` / `flux`). Representative performance on 5,000 test galaxies
(stamp size 53×53, pixel scale 0.141 arcsec):

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
│   ├── config/          #   the config schema, loader and dialect translation
│   ├── core/            #   models, training loops, dataset simulation
│   ├── training/        #   shearnet-train: run directory, history, curves
│   ├── evaluation/      #   shearnet-eval: rendering, measurements, catalog
│   ├── artifacts/       #   run directories, checkpoints, provenance
│   ├── io/              #   the evaluation catalog's columns and FITS writer
│   ├── methods/         #   NGmix and metacalibration
│   ├── plotting/        #   scatter, PSF systematics, animations
│   ├── utils/           #   normalization, device, simulation helpers
│   └── cli/             #   shearnet-train / shearnet-eval entry points
├── configs/             # example, smoke, paper campaign, variations
├── docs/                # config and catalog reference
├── scripts/             # shearnet.sbatch, make_docs.py
├── notebooks/           # runnable walkthroughs (see notebooks/README.md)
├── tests/               # pytest suite
├── psf_data/            # bundled SuperBIT PSFEx models
├── research/            # catalog building and hyperparameter sweeps
├── makefile             # installation targets
└── pyproject.toml       # package metadata and dependencies
```

## Testing

```bash
pip install -e ".[test]"
pytest tests/
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Requirements

- Python 3.8+ (3.11 is what CI and the cluster environment use)
- JAX / jaxlib (CPU or GPU), Flax, Optax, Orbax
- GalSim, JAX-GalSim (fork), NGmix
- NumPy, SciPy, Matplotlib, Astropy, tqdm, PyYAML, numba

See `pyproject.toml` for the declared list.

---

## License

MIT License — see `LICENSE`.

## Contributing

Contributions are welcome! Please open an issue or pull request.
