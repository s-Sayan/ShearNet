# Hyperparameter search

`sweep.py` turns a sweep spec into one validated config per grid point plus a
submission script, and afterwards ranks the finished runs
by their best validation loss (from each run's `manifest.json`). It changes
nothing about training: every knob is an ordinary config key.

```bash
# 1) write the configs and the submission script
python research/hyperparam_search/sweep.py --sweep research/hyperparam_search/example_sweep.yaml --write

# 2) submit training jobs; ranking needs no evaluation
bash research/hyperparam_search/sweep_out/submit.sh

# 3) rank them
python research/hyperparam_search/sweep.py --sweep research/hyperparam_search/example_sweep.yaml --collect
```

`--count` prints the number of configurations. Results go to
`sweep_out/results.csv`; each run's directory is `sweep_out/runs/<name>/`.
Evaluate a winner with its complete configuration, `--estimator shearnet`,
`--run <run dir>`, and a separate `--output` directory.
