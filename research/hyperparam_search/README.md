# Hyperparameter search (item 5)

Thin sweep runner over the existing `shearnet-train` CLI. It writes one config
per run, trains each as a subprocess, and ranks runs by their best validation
loss (read from the `*_loss.npz` training already saves). It changes nothing
about training — every knob is an ordinary config key.

## Usage

```bash
# Preview the runs without training (no compute; safe on a login node):
python research/hyperparam_search/sweep.py --sweep research/hyperparam_search/example_sweep.yaml --dry-run

# Sequential (one node runs every combo, then ranks):
python research/hyperparam_search/sweep.py --sweep research/hyperparam_search/example_sweep.yaml
```

Results are written to `sweep_out/results.csv` (ranked, best val loss first),
per-run configs to `sweep_out/configs/`, and one JSON result part per combo to
`sweep_out/parts/`. Checkpoints/plots land in the usual
`$SHEARNET_DATA_PATH/plots/<model_name>/` locations.

## SLURM job array (recommended for real runs)

Each combo is a full training (e.g. 300k samples), so run them in parallel as a
job array rather than serially. `sweep.sbatch` does exactly this; size the array
from the combo count:

```bash
# 1) how many combos?
N=$(python research/hyperparam_search/sweep.py --sweep research/hyperparam_search/example_sweep.yaml --count)

# 2) submit the array (edit sweep.sbatch's env block first; %8 caps concurrency):
sbatch --array=0-$((N-1))%8 research/hyperparam_search/sweep.sbatch

# 3) after it finishes, rank everything:
python research/hyperparam_search/sweep.py --sweep research/hyperparam_search/example_sweep.yaml --collect
```

Each array task runs `sweep.py --index $SLURM_ARRAY_TASK_ID` (one combo) and
writes its own `parts/result_<idx>.json`, so tasks never collide; `--collect`
aggregates and ranks them. Edit the `#SBATCH` resources and the environment
block in `sweep.sbatch` for your cluster.

## Writing a sweep

See `example_sweep.yaml`. Key fields:

- `base_config`: the config every run starts from (override a light one for
  quick exploration, then re-run winners at full scale).
- `method`: `grid` (full Cartesian product) or `random` (`n_samples` draws).
- `grid`: dotted config path → list of values. Anything in the config is fair
  game — `training.learning_rate`, `training.batch_size`,
  `training.weight_decay`, `training.ema_decay`, `model.type`, etc.

## Recommended flow

1. Sweep on a **small** base config (few samples/epochs) to find promising
   regions of lr / batch / weight_decay / EMA.
2. Take the top 2–3 configs and re-run at full sample count.
3. Run `research/shear_bias/m` and the `psf_leakage` benchmark on those for the
   actual m / c numbers — validation loss is the cheap proxy, not the final
   metric.
