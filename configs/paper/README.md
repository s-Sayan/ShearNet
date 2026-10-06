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

From the repository root, all four as a job array:

```
sbatch --array=0-3 scripts/shearnet.sbatch --list configs/paper/runs.txt
```

or one by hand:

```
sbatch scripts/shearnet.sbatch configs/paper/unit_tests/fourth.yaml
```

Each job trains its unit test and then evaluates it with the config's own
`evaluation` block, writing `evaluations/default/<run_name>_default.fits` into
the run directory.
