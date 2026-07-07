# ShearNet notebooks

A small, curated set of notebooks covering the main workflows end-to-end. They
run against the current `shearnet` API and use only *simulated* data by default
(no external catalogs required), so you can work through them right after
`make install`.

| Notebook | What it does | Needs a trained model? |
|---|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Simulate → train a CNN → evaluate → plot. **Start here.** | no |
| [`02_model_comparison.ipynb`](02_model_comparison.ipynb) | Compare several trained models: learning curves, MSE/bias table, prediction & residual plots, optional NGmix baseline. | yes |
| [`03_catalog_builder.ipynb`](03_catalog_builder.ipynb) | Turn a COSMOS / detection catalog into train & eval FITS (filter, augment, split, validate). | no |
| [`04_psf_diagnostics.ipynb`](04_psf_diagnostics.ipynb) | Inspect PSF ellipticity / size and measure a model's PSF leakage. | optional |

## Conventions

- Notebooks read and write under `$SHEARNET_DATA_PATH` (defaults to the current
  directory): trained models live in `model_checkpoint/`, and per-model configs
  and loss histories in `plots/<model_name>/`.
- Each notebook opens with a small **configuration** cell — edit those constants
  rather than the code below them.
- Plots adapt to `output_keys`, so the notebooks keep working whether a model
  predicts `(g1, g2)` or additional parameters such as `hlr` / `flux`.

## Relationship to the CLIs

`01` and `02` mirror `shearnet-train` and `shearnet-eval`: the notebooks are for
interactive exploration, the CLIs for reproducible runs. `02` and `04` import the
exact functions behind `shearnet-eval`, so their numbers match the CLI.
