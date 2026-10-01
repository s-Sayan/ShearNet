# Research record

Utilities and notes from the development of ShearNet that are not part of the
package. **None of it is needed to install or use ShearNet.**

| Directory | Contents |
|---|---|
| `shear_bias/` | Catalog preparation: building the train/eval catalogs and cutting them. |
| `hyperparam_search/` | The hyperparameter sweep driver. |

Configs live in `configs/`: the paper campaign in `configs/paper/`, the
exploratory runs that led to the fiducial in `configs/variations/`. The old
evaluation harness (`shear_bias/run.py` and the `m/`, `psf_leakage/` and
`timing/` scripts, every per-run `sub.sh`) is in the git history before the
rewrite of `shearnet-eval`.
