#!/usr/bin/env python
"""Hyperparameter sweep: write one config per grid point, then rank the runs.

It changes nothing about training -- every knob is an ordinary config key --
and it runs nothing itself. ``--write`` turns a sweep spec into validated
configs plus a run list for the one Slurm script; ``--collect`` reads each
finished run's ``manifest.json`` and ranks them by best validation loss.

    python research/hyperparam_search/sweep.py --sweep example_sweep.yaml --write
    sbatch --array=0-$((N-1))%8 scripts/shearnet.sbatch --list sweep_out/runs.txt
    python research/hyperparam_search/sweep.py --sweep example_sweep.yaml --collect

Sweep spec (YAML)::

    base_config: ../../configs/smoke.yaml    # relative to the spec
    method: grid                             # "grid" or "random"
    n_samples: 12                            # random only
    seed: 0                                  # random only
    name_prefix: sweep
    grid:                                    # dotted config key -> values
      training.learning_rate: [1.0e-3, 5.0e-4, 1.0e-4]
      training.batch_size: [32, 64, 128]

The runs are trained only (``--train-only`` lines in the run list): ranking by
validation loss needs no evaluation. Measure the winners with ``shearnet-eval``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random as _random
import sys
from pathlib import Path

from shearnet.config import Config
from shearnet.config.loader import dump_yaml, read_yaml


def _set(tree, dotted, value):
    node = tree
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def combos(grid, method="grid", n_samples=10, seed=0):
    """The deterministic list of ``{key: value}`` points of the sweep."""
    keys = list(grid)
    values = [grid[k] for k in keys]
    if method == "grid":
        return [dict(zip(keys, point)) for point in itertools.product(*values)]
    if method != "random":
        raise ValueError(f"method must be 'grid' or 'random', got {method!r}")
    rng, seen, out = _random.Random(seed), set(), []
    for _ in range(n_samples * 50):
        if len(out) == n_samples:
            break
        point = tuple(rng.choice(v) for v in values)
        if point not in seen:
            seen.add(point)
            out.append(dict(zip(keys, point)))
    return out


def slug(point, index):
    parts = [f"{k.split('.')[-1]}-{str(v).replace('.', 'p').replace('-', 'm')}"
             for k, v in point.items()]
    return f"{index:03d}_" + "_".join(parts)


def write(spec_path: Path, outdir: Path):
    spec = read_yaml(spec_path)
    base = read_yaml((spec_path.parent / spec["base_config"]).resolve())
    points = combos(spec["grid"], spec.get("method", "grid"), spec.get("n_samples", 10),
                    spec.get("seed", 0))
    (outdir / "configs").mkdir(parents=True, exist_ok=True)
    lines = []
    for index, point in enumerate(points):
        name = f"{spec.get('name_prefix', 'sweep')}_{slug(point, index)}"
        config = json.loads(json.dumps(base))
        for key, value in point.items():
            _set(config, key, value)
        _set(config, "run_options.run_name", name)
        _set(config, "run_options.outdir", str((outdir / "runs" / name).resolve()))
        path = outdir / "configs" / f"{name}.yaml"
        Config.from_dict(config, base_dir=str(spec_path.parent))  # validate
        path.write_text(dump_yaml(config))
        lines.append(f"--train-only {path.resolve()}")
    (outdir / "runs.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(points)} configs in {outdir / 'configs'}")
    print(f"sbatch --array=0-{len(points) - 1}%8 scripts/shearnet.sbatch --list "
          f"{(outdir / 'runs.txt').resolve()}")
    return points


def collect(spec_path: Path, outdir: Path):
    spec = read_yaml(spec_path)
    rows = []
    for config_path in sorted((outdir / "configs").glob("*.yaml")):
        config = Config.from_file(config_path)
        run = Path(config.get("run_options.outdir"))
        row = {key: config.get(key) for key in spec["grid"]}
        row["run_name"] = config.get("run_options.run_name")
        row["status"], row["best_val_loss"], row["best_epoch"] = "missing", None, None
        if (run / "status.json").is_file():
            row["status"] = json.loads((run / "status.json").read_text())["state"]
        if (run / "manifest.json").is_file():
            manifest = json.loads((run / "manifest.json").read_text())
            row["best_val_loss"] = manifest.get("best_val_loss")
            row["best_epoch"] = (manifest.get("checkpoint") or {}).get("epoch")
        rows.append(row)
    rows.sort(key=lambda r: (r["best_val_loss"] is None, r["best_val_loss"] or 0.0))
    if rows:
        with open(outdir / "results.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        loss = row["best_val_loss"]
        print(f"  {'n/a' if loss is None else f'{loss:.6e}':>12}  [{row['status']:>9}]  "
              f"{row['run_name']}")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", required=True, help="the sweep spec YAML")
    parser.add_argument("--outdir", default=None, help="default: sweep_out next to the spec")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write configs and runs.txt")
    action.add_argument("--count", action="store_true", help="print the number of runs")
    action.add_argument("--collect", action="store_true", help="rank the finished runs")
    args = parser.parse_args(argv)

    spec_path = Path(args.sweep).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else spec_path.parent / "sweep_out"
    if args.count:
        spec = read_yaml(spec_path)
        print(len(combos(spec["grid"], spec.get("method", "grid"), spec.get("n_samples", 10),
                         spec.get("seed", 0))))
    elif args.write:
        write(spec_path, outdir)
    else:
        collect(spec_path, outdir)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
