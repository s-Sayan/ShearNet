"""Check every run config against the failures that only appear at runtime.

Two jobs died this morning after less than a minute, and six more were queued
that would have trained for hours and then failed at the measurement. Both
classes are statically decidable, so they should cost a second rather than a
night:

  * an undefined name in the training entry point (a NameError raised the first
    time the in-loop path is executed, which no unit test covers),
  * a keyword the CLI passes that the trainer does not accept (a TypeError
    raised at the same moment, for the same reason),
  * a config whose backend the evaluation refuses,
  * an evaluation seed equal to the training seed, which the pipeline rejects
    by design,
  * a catalog or PSF path that does not exist on this machine,
  * a model the config describes that cannot actually be built.

    python research/ablations/preflight.py                # every generated config
    python research/ablations/preflight.py --paths        # also check paths exist
    python research/ablations/preflight.py first tier1/no_psf_response

Exit status is 0 only if every check passed, so it can gate a submission.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: The evaluation raises on anything else: R^PSF is a finite difference on the
#: PSF shear, which needs jax-galsim's explicit per-object psf_g1/psf_g2.
REQUIRED_BACKEND = "jax-galsim"


def _config_paths(names: List[str]) -> List[Path]:
    if names:
        out = []
        for name in names:
            for base in ("research/unit_test_variations", "research/ablations",
                         "research/unit_tests", ""):
                candidate = REPO / base / name / "config.yaml"
                if candidate.is_file():
                    out.append(candidate)
                    break
            else:
                raise SystemExit(f"no config.yaml for {name!r}")
        return out
    return sorted(p for p in (REPO / "research").rglob("config.yaml")
                  if "/ablations/" in str(p) or "/unit_tests/" in str(p))


def check_imports() -> List[str]:
    """Import the entry points a job runs, so a NameError surfaces here.

    An undefined name inside a function body is invisible to `import` alone,
    which is why this also compiles every module and asks pyflakes for
    undefined names when it is available.
    """
    problems: List[str] = []
    try:
        import shearnet.cli.train  # noqa: F401
        import shearnet.benchmarking  # noqa: F401
    except Exception as exc:  # pragma: no cover - the thing we are checking for
        problems.append(f"importing the training CLI failed: {type(exc).__name__}: {exc}")
        return problems

    try:
        from pyflakes.api import checkPath
        from pyflakes.reporter import Reporter
    except ImportError:
        problems.append("pyflakes not installed: cannot check for undefined names "
                        "(pip install pyflakes)")
        return problems

    import io

    for module in sorted((REPO / "shearnet").rglob("*.py")):
        out, err = io.StringIO(), io.StringIO()
        checkPath(str(module), Reporter(out, err))
        for line in out.getvalue().splitlines():
            # Match the real diagnostic, "undefined name 'x'", and not the
            # star-import advisory "unable to detect undefined names", which is
            # a note about pyflakes' own limits rather than a defect.
            if "undefined name '" in line:
                problems.append(line.replace(str(REPO) + "/", ""))
    return problems


def check_call_signatures() -> List[str]:
    """Every keyword the CLI passes must exist on the function it calls.

    Python binds keyword arguments at CALL time, so an importable module can
    still raise TypeError the first time the training path runs. This reads the
    call sites out of the AST -- the same check tests/test_call_signatures.py
    makes, repeated here so a submission is gated even when nobody ran pytest.
    """
    import ast
    import importlib
    import inspect

    watched = {
        "train_model_inloop": "shearnet.core.train_inloop",
        "train_model": "shearnet.core.train",
        "build_model": "shearnet.core.models",
    }
    problems: List[str] = []
    accepted = {}
    for function, module_name in watched.items():
        module = importlib.import_module(module_name)
        accepted[function] = set(inspect.signature(getattr(module, function)).parameters)

    for source in sorted((REPO / "shearnet").rglob("*.py")):
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.id if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if name not in watched:
                continue
            for keyword in node.keywords:
                if keyword.arg and keyword.arg not in accepted[name]:
                    problems.append(
                        f"{source.relative_to(REPO)}:{node.lineno}: passes "
                        f"{keyword.arg!r} to {name}(), which does not accept it"
                    )
    return problems


def check_config(path: Path, check_paths: bool) -> List[str]:
    """Everything about one config that can be decided without running it."""
    from shearnet.config.config_handler import Config

    problems: List[str] = []
    try:
        config = Config(str(path))
    except Exception as exc:
        return [f"will not load: {type(exc).__name__}: {exc}"]

    backend = config.get("dataset.backend")
    if backend != REQUIRED_BACKEND:
        problems.append(
            f"dataset.backend is {backend!r}; the evaluation requires "
            f"{REQUIRED_BACKEND!r} and refuses to run. This trains fine and then "
            "fails at the measurement, which is the expensive way to find out."
        )

    train_seed, eval_seed = config.get("dataset.seed"), config.get("eval.seed")
    if train_seed is not None and train_seed == eval_seed:
        problems.append(
            f"eval.seed == dataset.seed ({train_seed}); the pipeline refuses a "
            "benchmark seed equal to the training seed, because that re-renders "
            "the training galaxies with the same noise."
        )

    level = config.get("eval.evaluate.catalog_level")
    if level is not None:
        try:
            sys.path.insert(0, str(REPO / "research" / "shear_bias"))
            from catalog import resolve_level

            resolve_level(level)
        except ValueError as exc:
            problems.append(str(exc))

    if check_paths:
        for key in ("paths.train_catalog", "paths.eval_catalog",
                    "paths.psfex_model_file"):
            value = config.get(key)
            if value and not Path(value).exists():
                problems.append(f"{key} does not exist on this machine: {value}")

    try:
        problems.extend(_check_model_builds(config))
    except Exception as exc:  # pragma: no cover
        problems.append(f"model check itself failed: {type(exc).__name__}: {exc}")
    return problems


def _check_model_builds(config) -> List[str]:
    """Initialise the network the config describes."""
    import jax
    import jax.numpy as jnp

    from shearnet.core.models import build_model, is_fork_model

    keys = {
        "galaxy_type": "model.galaxy.type", "psf_type": "model.psf.type",
        "fusion": "model.fusion", "head": "model.head", "dropout": "model.dropout",
        "branch_features": "model.branch_features", "d4_features": "model.d4_features",
        "d4_depths_galaxy": "model.d4_depths_galaxy",
        "d4_depths_psf": "model.d4_depths_psf", "d4_multiscale": "model.d4_multiscale",
        "orbit_scan": "model.orbit_scan", "fusion_pos": "model.fusion_pos",
        "design": "model.design", "d_model": "model.d_model",
        "num_heads": "model.num_heads", "num_pool_heads": "model.num_pool_heads",
        "num_self_attn_layers": "model.num_self_attn_layers", "ffn_dim": "model.ffn_dim",
    }
    defaults = {"fusion": "concat", "head": "gap", "dropout": 0.0,
                "orbit_scan": True, "fusion_pos": "learned"}
    nn = config.get("model.type")
    kwargs = {name: config.get(key, defaults.get(name)) for name, key in keys.items()}
    try:
        model = build_model(nn, **kwargs)
        stamp = jnp.zeros((2, 53, 53))
        model.init(jax.random.PRNGKey(0),
                   *((stamp, stamp) if is_fork_model(nn) else (stamp,)))
    except Exception as exc:
        return [f"the model does not build: {type(exc).__name__}: {exc}"]
    return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="run names; default is every config")
    parser.add_argument("--paths", action="store_true",
                        help="also check catalog and PSF paths exist here")
    parser.add_argument("--skip-models", action="store_true",
                        help="skip building each model (faster)")
    args = parser.parse_args(argv)

    failures: List[Tuple[str, List[str]]] = []

    print("checking the training entry point ...")
    import_problems = check_imports()
    if import_problems:
        failures.append(("shearnet (imports)", import_problems))
    else:
        print("  ok  imports and undefined names")

    signature_problems = check_call_signatures()
    if signature_problems:
        failures.append(("shearnet (call signatures)", signature_problems))
    else:
        print("  ok  call signatures")

    if args.skip_models:
        globals()["_check_model_builds"] = lambda config: []

    paths = _config_paths(args.names)
    print(f"checking {len(paths)} config(s) ...")
    for path in paths:
        name = str(path.parent.relative_to(REPO))
        problems = check_config(path, args.paths)
        if problems:
            failures.append((name, problems))
        else:
            print(f"  ok  {name}")

    if failures:
        print("\n" + "=" * 70)
        print(f"{len(failures)} PROBLEM(S) -- do not submit until these are fixed")
        print("=" * 70)
        for name, problems in failures:
            print(f"\n{name}")
            for problem in problems:
                print(f"  - {problem}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
