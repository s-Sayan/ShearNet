# Paper configs

The four unit tests the paper reports, UT1-UT4 (its Table 1): the model fixed,
the simulation made more realistic one step at a time.

| file | PSF | half-light radius | flux |
|---|---|---|---|
| `unit_tests/first.yaml` | circular Gaussian, FWHM 0.5" | 0.5" | 12258.97 counts |
| `unit_tests/second.yaml` | empirical SuperBIT PSFEx | 0.5" | 12258.97 counts |
| `unit_tests/third.yaml` | empirical SuperBIT PSFEx | catalog | 12258.97 counts |
| `unit_tests/fourth.yaml` | empirical SuperBIT PSFEx | catalog | catalog |

They differ only in `simulation.*`, plus `training.response.orbit_weight: 0`
in UT1, where rotating a circular PSF is a no-op. `tests/test_paper_configs.py`
holds them to that. Each opens with what it changes and why.

The ngmix baseline fits the PSF with `em5`, a five-Gaussian mixture, as LITB
III does. A single Gaussian cannot represent the SuperBIT PSFEx profiles: ngmix's
measured R^PSF then stops describing its actual PSF leakage and the R^PSF
correction overcorrects.

Each run directory is `runs/unit_tests/<name>` in the repository root (its
`run_options.outdir`, relative to the config file, so it follows the clone).

## Submitting

From the repository root:

```
for n in first second third fourth; do
    SHEARNET_TRAIN_ARGS=--overwrite bash scripts/shearnet.sbatch \
        configs/paper/unit_tests/$n.yaml --time 4-00:00:00
done
```

or one by hand:

```
bash scripts/shearnet.sbatch configs/paper/unit_tests/fourth.yaml --time 4-00:00:00
```

Each submission creates a generation beneath its run directory and submits
training, ShearNet evaluation, independent CPU ngmix evaluation, and merging.
The dispatcher log prints the generation path. Its `merged.fits` is the input
for the plotting repository. `shearnet-pipeline status CONFIG` shows where each
stage is; after an interruption, `shearnet-pipeline resume CONFIG` (with e.g.
`--memory ngmix=96G`) continues it, keeping completed stages and measurements.
