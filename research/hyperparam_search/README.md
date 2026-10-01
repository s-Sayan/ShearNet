# Hyperparameter search

`sweep.py` turns a sweep spec into one validated config per grid point plus a
run list for `scripts/shearnet.sbatch`, and afterwards ranks the finished runs
by their best validation loss (from each run's `manifest.json`). It changes
nothing about training: every knob is an ordinary config key.

```bash
# 1) write the configs and the run list (prints the sbatch line to use)
python research/hyperparam_search/sweep.py --sweep research/hyperparam_search/example_sweep.yaml --write

# 2) train them as a job array (train only: ranking needs no evaluation)
sbatch --array=0-35%8 scripts/shearnet.sbatch --list research/hyperparam_search/sweep_out/runs.txt

# 3) rank them
python research/hyperparam_search/sweep.py --sweep research/hyperparam_search/example_sweep.yaml --collect
```

`--count` prints the number of runs, for sizing `--array`. Results go to
`sweep_out/results.csv`; each run's directory is `sweep_out/runs/<name>/`.
Measure the winners with `shearnet-eval --run <run dir>`.
