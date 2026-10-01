# Research record

Experiment configurations and notes from the development of ShearNet. **None of
it is needed to install or use the package** -- it is kept for provenance.

| Directory | Contents |
|---|---|
| `unit_tests/` | The four simulation-ladder configs (UT1-UT4), with notes. |
| `unit_test_variations/` | Variations on those runs (PSFs, architectures, catalogs, weightings). |
| `ablations/` | The ablation arms and the script that generates them from the fiducial. |
| `hyperparam_search/` | The hyperparameter sweep driver. |
| `shear_bias/` | Catalog preparation: building the train/eval catalogs and cutting them. |

The old evaluation harness (`shear_bias/run.py`, the `m/`, `psf_leakage/` and
`timing/` scripts and every per-run `sub.sh`) is gone; it lives in the git
history before the rewrite of `shearnet-eval`.

These files reference absolute paths from the original author's cluster; treat
them as a record rather than runnable examples. For supported usage see the
top-level `README.md`.
