# ShearNet

A JAX-based neural network library for galaxy **shear estimation**. ShearNet
simulates galaxy images with [GalSim](https://github.com/GalSim-developers/GalSim)
(or its differentiable port, JAX-GalSim), trains neural networks to recover their
shape (`g1`, `g2`) and other parameters, and measures them side by side with
[NGmix](https://github.com/esheldon/ngmix) and its metacalibration.

Train and evaluate from one configuration:

```bash
export JAX_ENABLE_X64=1
shearnet-train --config my.yaml --run runs/my
shearnet-eval --config my.yaml --run runs/my --estimator shearnet --output results/my/shearnet
shearnet-eval --config my.yaml --estimator ngmix --output results/my/ngmix
shearnet-merge --shearnet results/my/shearnet/shearnet.fits --ngmix results/my/ngmix/ngmix.fits --output results/my/merged.fits
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
└── logs/train.log
```

## Evaluation and recovery

Each estimator writes a standalone FITS catalog and keeps completed observations
in checksummed files. ngmix needs the experiment configuration but no trained model.

```bash
export JAX_ENABLE_X64=1
shearnet-eval --config my.yaml --estimator ngmix --output results/ngmix
shearnet-eval --config my.yaml --estimator shearnet --run runs/my --output results/shearnet
shearnet-merge --shearnet results/shearnet/shearnet.fits --ngmix results/ngmix/ngmix.fits --output results/merged.fits
```

Add `--resume` to continue an interrupted evaluation in the same output directory.
Completed observations are checked and reused. `--checkpoint-seconds` controls the
save interval (300 seconds by default); checkpoints also commit after 2,048 results.
A failed fit remains a flagged row. An unfinished catalog is never published as complete.

The merged file contains `TRUTH`, `STAMP`, `SHEARNET`, and `NGMIX`. These are raw
measurements; response, bias, leakage and selection are computed downstream.
Schema 2 adds object and observation UIDs and fingerprints of the common input
images. The merge checks that both jobs saw the same observations (to roundoff:
ShearNet renders on the GPU, ngmix on CPUs) and copies measurement values without
rounding or recalibration. See [docs/catalog.md](docs/catalog.md).

Training also accepts `--resume`. Its latest checkpoint stores the optimizer,
EMA, random state, batch position, history and best model. The previous checkpoint
is retained as a fallback. The inference model remains `model/best.msgpack`.

## Configuration

One YAML file with five blocks: `run_options`, `simulation`, `model`, `training`,
`evaluation`. Every key, its type, default and meaning is in
[`docs/config.md`](docs/config.md), generated from `shearnet/config/schema.py`.

* A key that is not in the schema is an **error** (with a suggestion), as is a
  setting the chosen architecture would never read.
* Anything left out takes its default; `config.resolved.yaml` records them all.
* Relative paths resolve against the config file's directory.

`configs/example.yaml` is a short commented tour, `configs/smoke.yaml` the tiny
CI run, and `configs/paper/` the paper's four unit tests (see
[`configs/paper/README.md`](configs/paper/README.md)). Only the documented configuration
schema is accepted.

## On Turing

Activate the environment, then submit from the repository root. Run the script
with `bash` on the login node so nothing waits in a queue just to submit:

```bash
for n in first second third fourth; do
    SHEARNET_TRAIN_ARGS=--overwrite bash scripts/shearnet.sbatch \
        configs/paper/unit_tests/$n.yaml --time 4-00:00:00
done
```

`sbatch --time=4-00:00:00 scripts/shearnet.sbatch CONFIG` also works; it submits
from a one-CPU job instead. Either way the dispatcher submits GPU training, GPU
ShearNet evaluation after training, independent CPU ngmix evaluation, and a CPU
merge after both evaluations.
GPU stages fail immediately if JAX cannot use the requested accelerator.
It records the job IDs and exits. Each submission creates a generation beneath the
configured run directory, with a frozen copy of the package and resolved configuration.
`--overwrite` starts a new generation; it does not delete previous results.

The log prints the generation directory. Its outputs are `training/`,
`shearnet/shearnet.fits`, `ngmix/ngmix.fits`, and `merged.fits`.

Both commands take the config you submitted and use its newest generation (a
run directory or a generation directory works too):

```bash
shearnet-pipeline status configs/paper/unit_tests/fourth.yaml
shearnet-pipeline resume configs/paper/unit_tests/fourth.yaml
```

`status` shows each stage's state and observations saved; for a stage that
stopped, it says why (out of memory, time limit, an error) and prints the resume
command to run. `resume` keeps completed stages and saved observations,
resubmits the rest and repairs their dependencies. Change a stage's resources on
the way, e.g. `--memory ngmix=96G`, `--cpus shearnet=8` or
`--time ngmix=2-00:00:00`; a stage still queued or running must be `scancel`ed
first. It refuses duplicate active writers. Results and checkpoints must
live on shared storage that survives the allocation; node-local temporary storage
is unsuitable. Turing scratch is useful for active work but can be purged.

The time limit (`--time`, or the dispatcher job's own under `sbatch`) becomes
the ceiling for each compute job; a limit that fits in `short`'s MaxTime (read from Slurm) goes there, anything longer to `long`. The merge
has a two-hour limit. Stage resource requests can be set in a JSON file passed as
`--resources FILE` after the configuration. For example:

```json
{"ngmix": {"cpus": 32, "memory": "24G", "time": "6-00:00:00", "partition": "long"}}
```

Defaults are starting points: training uses one GPU, eight CPUs and 128G. The two
evaluation jobs are timed, so they are sized like NERSC Perlmutter nodes: ShearNet
evaluation uses one A100, 16 CPUs and 64G (one GPU's share of a GPU node); ngmix
uses 128 CPUs and 480G (one CPU node; Turing's 512 GB nodes offer slightly less
than 512G, and each worker holds ~0.5 GB). To time a finished run on other
hardware without retraining, run its evaluations again:
`shearnet-pipeline resume CONFIG --rerun shearnet --rerun ngmix` with any
`--cpus`/`--memory`/`--gres` changes; the earlier results are moved to
`*.replaced-<time>`, not deleted. ShearNet
evaluation renders and predicts on the GPU, but ngmix's metacal images for it
are made on its CPUs (about 90 ms per observation), so its CPU count sets how
long it holds the GPU. The
submission validates the live partition time limit. Measure CPU/GPU utilization
and peak memory on a representative pilot before launching the full campaign.
Set `SHEARNET_ENV` to an activation script if the environment is not already active.


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
│   ├── config/          #   the config schema and loader
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
