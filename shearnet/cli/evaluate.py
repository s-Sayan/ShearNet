"""Evaluate one estimator with resumable output."""
import argparse
import logging
from ..artifacts import RunDir
from ..artifacts.recovery import Interrupted
from ..config import Config
from ..logging_utils import configure_logging


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="complete experiment configuration")
    parser.add_argument("--estimator", required=True, choices=("shearnet", "ngmix"))
    parser.add_argument("--output", required=True,
                        help="directory for this estimator and its shards")
    parser.add_argument("--run", help="completed training directory; required for ShearNet")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-size", type=int,
                        help="galaxies rendered at a time (default: simulation.jax_batch_size)")
    parser.add_argument("--checkpoint-seconds", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(level=logging.WARNING if args.quiet else logging.INFO, force=True)
    try:
        config = Config.from_file(args.config)
        run = RunDir(args.run) if args.run else None
        if args.estimator == "shearnet":
            if run is None:
                parser.error("--estimator shearnet requires --run")
            run.require_completed()
            training = Config.from_file(run.config_resolved)
            for section in ("model", "training", "simulation"):
                if training.get(section) != config.get(section):
                    raise ValueError(f"{section} differs from the trained model configuration")
        from ..evaluation.service import evaluate, plan
        if args.dry_run:
            print(plan(config))
            return 0
        evaluate(config, args.output, estimator=args.estimator, run=run, resume=args.resume,
                 chunk_size=args.chunk_size, checkpoint_seconds=args.checkpoint_seconds)
    except Interrupted as exc:
        logging.warning("%s", exc)
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
