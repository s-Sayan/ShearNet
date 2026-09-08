"""Every keyword the training CLI passes must exist on the function it calls.

Three SLURM jobs have now died on this exact class of defect: a keyword-passing
chain edited at one end and not the other. `NameError: is_fork_model`, then
`TypeError: train_model_inloop() got an unexpected keyword argument
'd4_multiscale'`. Both are decidable without running anything, and both cost a
scheduler slot to discover instead.

Python resolves keyword arguments at CALL time, so importing the module proves
nothing and neither does any test that stops short of executing the training
path. These tests read the call sites out of the AST and compare them against
the callee signatures.
"""

import ast
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Call sites that matter: a mismatch here kills a job on the cluster, and
#: nowhere else in the suite executes them.
WATCHED = {
    "train_model_inloop": "shearnet.core.train_inloop",
    "train_model": "shearnet.core.train",
    "build_model": "shearnet.core.models",
}


def _keywords_passed(source: Path, function: str):
    """Every keyword name passed to `function` anywhere in `source`."""
    tree = ast.parse(source.read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else None)
        if name != function:
            continue
        for keyword in node.keywords:
            if keyword.arg is not None:  # skip **kwargs splats
                found.append((keyword.arg, node.lineno))
    return found


def _parameters(function: str) -> set:
    import importlib

    module = importlib.import_module(WATCHED[function])
    return set(inspect.signature(getattr(module, function)).parameters)


CALLERS = sorted(
    set((path, function)
        for function in WATCHED
        for path in (REPO / "shearnet").rglob("*.py")
        if _keywords_passed(path, function))
)


@pytest.mark.parametrize("path,function", CALLERS,
                         ids=[f"{p.name}->{f}" for p, f in CALLERS])
def test_every_keyword_at_the_call_site_exists_on_the_callee(path, function):
    accepted = _parameters(function)
    bad = [(kw, line) for kw, line in _keywords_passed(path, function)
           if kw not in accepted]
    assert not bad, (
        f"{path.relative_to(REPO)} passes keywords {function}() does not accept: "
        + ", ".join(f"{kw!r} (line {line})" for kw, line in bad)
    )


def test_the_two_trainers_accept_the_same_model_knobs():
    """An arm must be able to describe the same network either way.

    `generation: upfront` routes to train_model and `inloop` to
    train_model_inloop. If only one of them accepts the D4 schedule, an
    up-front arm silently trains the default backbone -- no error, just a
    model that is not the one its config names, compared in a table against
    one that is.
    """
    knobs = {"d4_features", "d4_depths_galaxy", "d4_depths_psf", "d4_multiscale",
             "orbit_scan", "fusion_pos", "design", "d_model", "num_heads",
             "num_pool_heads", "num_self_attn_layers", "ffn_dim"}
    missing_upfront = knobs - _parameters("train_model")
    missing_inloop = knobs - _parameters("train_model_inloop")
    assert not missing_upfront, f"train_model is missing {sorted(missing_upfront)}"
    assert not missing_inloop, f"train_model_inloop is missing {sorted(missing_inloop)}"


def test_train_config_fields_are_all_accepted_by_train_model():
    """TrainConfig.run() splats its fields into train_model()."""
    from shearnet.core.specs import TrainConfig
    from shearnet.core.train import train_model

    fields = set(TrainConfig.__dataclass_fields__)
    accepted = set(inspect.signature(train_model).parameters)
    # save_path/model_name are handled by as_kwargs(), not passed through raw.
    unexpected = fields - accepted - {"save_path", "model_name"}
    assert not unexpected, f"TrainConfig carries fields train_model rejects: {sorted(unexpected)}"


def test_the_upfront_path_actually_carries_the_d4_schedule():
    """Accepting a keyword is not the same as delivering its value.

    tier2/06 is d4-fork-like with generation: upfront, so it routes through
    TrainConfig -> train_model. Before this was threaded, that path fell back to
    the default (32,48,64)/(2,2,1)/(1,1,1) backbone while the config named the
    trimmed one -- silently, which is how a wrong number reaches a table.
    """
    from shearnet.config.config_handler import Config
    from shearnet.core.specs import TrainConfig

    config = Config(str(REPO / "research/ablations/tier2/06_d4_equivariant/config.yaml"))
    kwargs = TrainConfig.from_config(config, save_path="/tmp/unused").as_kwargs()
    for key in ("d4_features", "d4_depths_galaxy", "d4_depths_psf", "design"):
        assert kwargs.get(key) == config.get(f"model.{key}"), key
