"""``shearnet-train``: train one model into one run directory.

::

    shearnet-train --config configs/paper/fiducial.yaml
    shearnet-train --config configs/example.yaml --run runs/example
    shearnet-train --config configs/example.yaml --run runs/example --dry-run

Every setting is in the YAML. ``--run`` overrides ``run_options.outdir``; one
of the two must say where the run lives. ``--dry-run`` validates the config and
the input files and prints what would be written, without writing anything.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ..config import Config, ConfigError
from ..logging_utils import configure_logging, get_logger

logger = get_logger(__name__)
logging.getLogger("absl").setLevel(logging.ERROR)

_EXAMPLES = """examples:
  shearnet-train --config configs/paper/fiducial.yaml
  shearnet-train --config configs/example.yaml --run runs/example
  shearnet-train --config configs/example.yaml --run runs/example --dry-run
"""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shearnet-train",
        description="Train a ShearNet model into a self-contained run directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXAMPLES,
    )
    parser.add_argument("--config", required=True, help="the run's YAML config")
    parser.add_argument("--run", default=None,
                        help="run directory (overrides run_options.outdir)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete an existing run directory first (its evaluations too)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and report; write nothing")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    verbosity.add_argument("-v", "--verbose", action="store_true", help="debug output")
    return parser


def build_train_config(args) -> Config:
    """The validated config ``args`` describe. Reads files, writes nothing."""
    config = Config.from_file(args.config)
    if args.run:
        config = config.with_overrides({"run_options": {"outdir": os.path.abspath(args.run)}})
    if not config.get("run_options.run_name"):
        raise ConfigError(f"{args.config}: set run_options.run_name")
    if not config.get("run_options.outdir"):
        raise ConfigError(f"{args.config}: say where the run goes, with "
                          "run_options.outdir or --run")
    return config


def check_inputs(config: Config, *, evaluation: bool = True) -> None:
    """Fail now on an input file that is not there, rather than hours in.

    ``evaluation`` also checks the held-out catalog: training a model that then
    cannot be evaluated is a wasted allocation.
    """
    for key, needed in (("simulation.catalogs.train_file", True),
                        ("simulation.catalogs.eval_file", evaluation)):
        path = config.get(key)
        if needed and path is not None and not os.path.isfile(path):
            raise FileNotFoundError(f"{key} does not exist: {path}")
    if config.get("simulation.psf.mode") == "superbit":
        path = config.get("simulation.psf.psfex_file")
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"simulation.psf.psfex_file does not exist: {path}")


def _plan(config: Config) -> str:
    from ..artifacts import RunDir

    run = RunDir(config.get("run_options.outdir"))
    taken = run.root.exists() and any(run.root.iterdir())
    return "\n".join([
        f"run:        {config.get('run_options.run_name')}",
        f"directory:  {run.root}" + (f"  (already holds a run: {run.state or 'no status'})"
                                     if taken else ""),
        f"model:      {config.get('model.type')} -> {list(config.get('model.output_keys'))}",
        f"training:   {config.get('training.nobj')} objects, "
        f"{config.get('training.generation')}, {config.get('training.epochs')} epochs, "
        f"backend {config.get('simulation.backend')}",
        "would write: config.input.yaml, config.resolved.yaml, manifest.json, status.json, "
        "model/best.msgpack, normalizers/, training/history.{csv,npz}, "
        "training/learning_curve.png, logs/train.log",
    ])


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging(level=logging.WARNING if args.quiet else
                      logging.DEBUG if args.verbose else logging.INFO, force=True)
    try:
        config = build_train_config(args)
        check_inputs(config)
    except (ConfigError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2
    if args.dry_run:
        logger.info(_plan(config))
        logger.info("dry run: config and inputs are valid; nothing written")
        return 0

    from ..artifacts import RunDir
    from ..training.service import train

    train(config, RunDir(config.get("run_options.outdir")),
          input_text=Path(args.config).read_text(), overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
