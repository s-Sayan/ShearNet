# Research record

Utilities and notes from the development of ShearNet that are not part of the
package. **None of it is needed to install or use ShearNet.**

| Directory | Contents |
|---|---|
| `shear_bias/` | Catalog preparation: building the train/eval catalogs and cutting them. |
| `hyperparam_search/` | The hyperparameter sweep driver. |

Configs live in `configs/`: the paper campaign in `configs/paper/`, the
exploratory runs in `configs/variations/`. Evaluation uses `shearnet-eval`.
