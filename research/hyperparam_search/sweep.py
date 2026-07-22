#!/usr/bin/env python
"""Hyperparameter sweep runner for ShearNet (item 5: parameter search).

Drives the existing ``shearnet-train`` CLI over a grid (or random sample) of
config overrides and ranks the runs by their best validation loss. It writes one
config per run, launches training as a subprocess (so the full, unmodified
pipeline is exercised), then reads back the per-epoch ``*_loss.npz`` that
training already saves and reports the minimum validation loss per run.

This is intentionally thin: it changes nothing about training, and every knob it
sweeps is an ordinary config key. Deeper m/c benchmarking is a separate,
heavier step -- run ``research/shear_bias/m`` on the best few models the sweep
surfaces.

Three ways to run:

* Sequential (one node runs every combo, then ranks)::

      python sweep.py --sweep example_sweep.yaml

* SLURM job array (one combo per array task, then collect) -- the right mode
  when each run is a full 300k-sample training::

      N=$(python sweep.py --sweep example_sweep.yaml --count)   # combo count
      # sbatch --array=0-$((N-1)) ... running:
      python sweep.py --sweep example_sweep.yaml --index $SLURM_ARRAY_TASK_ID
      # after the array finishes:
      python sweep.py --sweep example_sweep.yaml --collect

* Preview (no training)::

      python sweep.py --sweep example_sweep.yaml --dry-run

Sweep spec (YAML)::

    base_config: configs/dry_run.yaml     # relative to this file's dir or CWD
    method: grid                          # "grid" or "random"
    n_samples: 12                         # random only: how many combos to draw
    seed: 0                               # random only: reproducible sampling
    model_name_prefix: sweep              # checkpoints/plots go under this name
    grid:                                 # dotted config path -> list of values
      training.learning_rate: [1.0e-3, 5.0e-4, 1.0e-4]
      training.batch_size: [32, 64, 128]
      training.weight_decay: [1.0e-4, 1.0e-5]
      training.ema_decay: [null, 0.999]
"""
import argparse
import csv
import glob
import itertools
import json
import os
import random as _random
import subprocess
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    """Resolve ``path`` relative to CWD, else relative to this script's dir."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    alt = os.path.join(HERE, path)
    return alt if os.path.exists(alt) else path


def _set_nested(d, dotted, value):
    """Set ``d['a']['b'] = value`` for a dotted ``'a.b'`` path (creating dicts)."""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _combos(grid, method, n_samples, seed):
    """Return the deterministic list of ``{path: value}`` dicts for the grid.

    The order is stable across invocations (full product for ``grid``; a
    seeded sample for ``random``) so ``--index N`` selects the same combo in
    every array task.
    """
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    out = []
    if method == "grid":
        for values in itertools.product(*value_lists):
            out.append(dict(zip(keys, values)))
    elif method == "random":
        rng = _random.Random(seed)
        seen = set()
        max_attempts = n_samples * 50
        while len(out) < n_samples and max_attempts > 0:
            max_attempts -= 1
            combo = tuple(rng.choice(v) for v in value_lists)
            if combo in seen:
                continue
            seen.add(combo)
            out.append(dict(zip(keys, combo)))
    else:
        raise ValueError(f"method must be 'grid' or 'random', got {method!r}")
    return out


def _slug(combo, idx):
    """Short, filesystem-safe label for a combo (prefixed by its index)."""
    parts = []
    for k, v in combo.items():
        parts.append(f"{k.split('.')[-1]}-{str(v).replace('.', 'p').replace('-', 'm')}")
    return f"{idx:03d}_" + "_".join(parts)


def _best_val_loss(plot_path, model_name):
    """Return the minimum validation loss recorded for ``model_name`` (or NaN)."""
    loss_file = os.path.join(plot_path, model_name, f"{model_name}_loss.npz")
    if not os.path.exists(loss_file):
        return float("nan")
    data = np.load(loss_file, allow_pickle=True)
    val = np.asarray(data["val_loss"], dtype=float)
    return float(np.min(val)) if val.size else float("nan")


def _run_one(idx, combo, base_config, prefix, cfg_dir, parts_dir, plot_path,
             train_cmd, dry_run):
    """Train a single combo (unless ``dry_run``) and persist its result part."""
    slug = _slug(combo, idx)
    model_name = f"{prefix}_{slug}"

    run_cfg = yaml.safe_load(yaml.safe_dump(base_config))  # deep copy
    for path, value in combo.items():
        _set_nested(run_cfg, path, value)
    _set_nested(run_cfg, "output.model_name", model_name)
    _set_nested(run_cfg, "plotting.plot", run_cfg.get("plotting", {}).get("plot", False))

    cfg_file = os.path.join(cfg_dir, f"{model_name}.yaml")
    with open(cfg_file, "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)

    cmd = [train_cmd, "--config", cfg_file]
    print(f"[sweep] combo {idx}: {model_name}\n        params: {combo}")

    if dry_run:
        print(f"        would run: {' '.join(cmd)}")
        result = {**combo, "index": idx, "model_name": model_name,
                  "val_loss": float("nan"), "status": "dry-run"}
    else:
        proc = subprocess.run(cmd)
        status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
        val_loss = _best_val_loss(plot_path, model_name) if proc.returncode == 0 else float("nan")
        print(f"        -> status={status}  best_val_loss={val_loss:.6e}")
        result = {**combo, "index": idx, "model_name": model_name,
                  "val_loss": val_loss, "status": status}

    # One JSON part per combo -> safe for concurrent array tasks.
    with open(os.path.join(parts_dir, f"result_{idx:04d}.json"), "w") as f:
        json.dump(result, f)
    return result


def _collect(outdir, parts_dir, grid):
    """Read every result part, rank by val loss, write results.csv, print table."""
    parts = sorted(glob.glob(os.path.join(parts_dir, "result_*.json")))
    results = []
    for p in parts:
        with open(p) as f:
            results.append(json.load(f))
    if not results:
        print(f"[sweep] no result parts found in {parts_dir}")
        return []

    results.sort(key=lambda r: (np.isnan(r.get("val_loss", float("nan"))),
                                r.get("val_loss", float("nan"))))
    results_file = os.path.join(outdir, "results.csv")
    fieldnames = list(grid.keys()) + ["index", "model_name", "val_loss", "status"]
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("=" * 70)
    print(f"[sweep] ranked results (best first) -> {results_file}")
    print("=" * 70)
    for r in results:
        vl = r.get("val_loss", float("nan"))
        vl_s = f"{vl:.6e}" if not np.isnan(vl) else "   n/a    "
        print(f"  {vl_s}  [{r.get('status', '?'):>10}]  {r['model_name']}")
    best = results[0]
    if not np.isnan(best.get("val_loss", float("nan"))):
        print(f"\n[sweep] best: {best['model_name']} (val_loss={best['val_loss']:.6e})")
        print("[sweep] next: run research/shear_bias/m on the top few for m/c.")
    return results


def main():
    ap = argparse.ArgumentParser(description="ShearNet hyperparameter sweep")
    ap.add_argument("--sweep", required=True, help="Path to the sweep YAML spec.")
    ap.add_argument("--outdir", default=None, help="Where to write run configs + results.")
    ap.add_argument("--train-cmd", default="shearnet-train",
                    help="Training entry point (default: shearnet-train).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write configs and print commands, but do not train.")
    ap.add_argument("--count", action="store_true",
                    help="Print the number of combos and exit (for sizing --array).")
    ap.add_argument("--index", type=int, default=None,
                    help="Run ONLY this combo index (for SLURM job arrays), then exit.")
    ap.add_argument("--collect", action="store_true",
                    help="Aggregate all result parts into a ranked results.csv, then exit.")
    args = ap.parse_args()

    with open(_resolve(args.sweep)) as f:
        spec = yaml.safe_load(f)

    base_config_path = _resolve(spec["base_config"])
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    method = spec.get("method", "grid")
    n_samples = spec.get("n_samples", 10)
    seed = spec.get("seed", 0)
    prefix = spec.get("model_name_prefix", "sweep")
    grid = spec["grid"]

    combos = _combos(grid, method, n_samples, seed)

    if args.count:
        print(len(combos))
        return 0

    outdir = args.outdir or os.path.join(HERE, "sweep_out")
    cfg_dir = os.path.join(outdir, "configs")
    parts_dir = os.path.join(outdir, "parts")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(parts_dir, exist_ok=True)

    data_path = os.getenv("SHEARNET_DATA_PATH", os.path.abspath("."))
    plot_path = os.path.join(data_path, "plots")

    if args.collect:
        _collect(outdir, parts_dir, grid)
        return 0

    print(f"[sweep] {len(combos)} combo(s); method={method}; base={base_config_path}")
    print(f"[sweep] plots dir: {plot_path}\n")

    if args.index is not None:
        # Single-combo mode (SLURM array task). Out-of-range indices no-op so a
        # slightly-too-large --array range is harmless.
        if not (0 <= args.index < len(combos)):
            print(f"[sweep] index {args.index} out of range [0,{len(combos)}); nothing to do.")
            return 0
        _run_one(args.index, combos[args.index], base_config, prefix, cfg_dir,
                 parts_dir, plot_path, args.train_cmd, args.dry_run)
        print("[sweep] single combo done; run with --collect once all array tasks finish.")
        return 0

    # Sequential: run every combo in this process, then rank.
    for idx, combo in enumerate(combos):
        print(f"[sweep] ({idx + 1}/{len(combos)})")
        _run_one(idx, combo, base_config, prefix, cfg_dir, parts_dir, plot_path,
                 args.train_cmd, args.dry_run)
        print()
    _collect(outdir, parts_dir, grid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
