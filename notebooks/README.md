# ShearNet notebooks

A small, curated set of notebooks covering the main workflows end-to-end. They
run against the current `shearnet` API and use only *simulated* data by default
(no external catalogs required), so you can work through them right after
`make install`.

| Notebook | What it does | Needs a trained model? |
|---|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Simulate → train a CNN → evaluate → plot. **Start here.** | no |
| [`02_model_comparison.ipynb`](02_model_comparison.ipynb) | Compare several trained and evaluated runs: learning curves, MSE/bias table, prediction & residual plots, NGmix on the same stamps. | yes (evaluated) |
| [`03_catalog_builder.ipynb`](03_catalog_builder.ipynb) | Turn a COSMOS / detection catalog into train & eval FITS (filter, augment, split, validate). | no |
| [`04_psf_diagnostics.ipynb`](04_psf_diagnostics.ipynb) | Inspect PSF ellipticity / size and measure a model's PSF leakage. | optional |

## Conventions

- `02` and `04` read **run directories** -- what `shearnet-train --run DIR` writes
  and `shearnet-eval --run DIR` adds an evaluation catalog to. Set `RUNS` / `RUN`
  at the top. `configs/smoke.yaml` makes one in a couple of minutes:
  `shearnet-train --config configs/smoke.yaml --run runs/smoke && shearnet-eval --run runs/smoke`.
- Each notebook opens with a small **configuration** cell — edit those constants
  rather than the code below them.
- Plots adapt to `output_keys`, so the notebooks keep working whether a model
  predicts `(g1, g2)` or additional parameters such as `hlr` / `flux`.

## Relationship to the CLIs

`01` uses the same building blocks as `shearnet-train`, interactively. `02` reads
the catalogs `shearnet-eval` writes instead of re-simulating, and `04` loads a
run's model through `shearnet.evaluation.predictor.RunPredictor`, the loader
`shearnet-eval` itself uses, so its predictions match the CLI.

`paper_plots/unit_tests_figures.ipynb` reads the outputs of the evaluation
harness this repository used before `shearnet-eval`; the paper figures are made
from the new catalogs in the plotting repository.
