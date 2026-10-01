"""``shearnet-eval``: measure a finished run and write one raw catalog.

::

    shearnet-eval --run runs/fiducial
    shearnet-eval --run runs/fiducial --config cut_eval.yaml --eval-name cut
    shearnet-eval --run runs/fiducial --dry-run

Everything comes from the run's own resolved config. ``--config`` may change
evaluation settings only (``evaluation.*``, the evaluation catalog,
``run_options.ncores``) -- never the model, the renderer or the training
population -- and its result goes under its own ``--eval-name``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..artifacts import RunDir, RunError
from ..config import Config, ConfigError
from ..logging_utils import configure_logging, get_logger

logger = get_logger(__name__)
logging.getLogger("absl").setLevel(logging.ERROR)

_EXAMPLES = """examples:
  shearnet-eval --run runs/fiducial
  shearnet-eval --run runs/fiducial --config cut_eval.yaml --eval-name cut
  shearnet-eval --run runs/fiducial --dry-run
"""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shearnet-eval",
        description="Measure a finished ShearNet run; write one raw FITS catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EXAMPLES,
    )
    parser.add_argument("--run", required=True, help="a completed training run directory")
    parser.add_argument("--config", default=None,
                        help="evaluation-only overrides (evaluation.*, the eval catalog, ncores)")
    parser.add_argument("--eval-name", default="default",
                        help="name of this evaluation; its own directory under evaluations/")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing evaluation of the same name")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and report the planned work; write nothing")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    verbosity.add_argument("-v", "--verbose", action="store_true", help="debug output")
    return parser


def resolve(args) -> Config:
    """The config the evaluation would run with. Reads files, writes nothing."""
    import os

    run = RunDir(args.run)
    run.require_completed()
    config = Config.from_file(run.config_resolved)
    if args.config:
        config = config.evaluation_override(args.config)
    if config.get("simulation.backend") != "jax-galsim":
        raise ConfigError("shearnet-eval needs a run trained with simulation.backend: "
                          "jax-galsim (the scenes and ring stations are set on its truth table)")
    catalog = config.get("simulation.catalogs.eval_file")
    if catalog is not None and not os.path.isfile(catalog):
        raise FileNotFoundError(f"simulation.catalogs.eval_file does not exist: {catalog}")
    return config


def main(argv=None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging(level=logging.WARNING if args.quiet else
                      logging.DEBUG if args.verbose else logging.INFO, force=True)
    try:
        config = resolve(args)
        edir = RunDir(args.run).evaluation(args.eval_name)
        if (not args.dry_run and not args.overwrite and edir.root.exists()
                and any(edir.root.iterdir())):
            raise RunError(f"{edir.root} already holds an evaluation ({edir.state}); give "
                           "this one another --eval-name, or pass --overwrite")
    except (ConfigError, FileNotFoundError, RunError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    if args.dry_run:
        from ..evaluation.service import plan

        run = RunDir(args.run)
        work = plan(config)
        taken = edir.root.exists() and any(edir.root.iterdir())
        logger.info("run:         %s", run.root)
        logger.info("evaluation:  %s%s", edir.root,
                    f"  (already holds an evaluation: {edir.state})" if taken else "")
        for key, value in work.items():
            logger.info("%-12s %s", key + ":", value)
        logger.info("dry run: nothing written")
        return 0

    from ..evaluation.service import evaluate

    evaluate(RunDir(args.run), override=Path(args.config) if args.config else None,
             eval_name=args.eval_name, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
