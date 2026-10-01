"""Every generated paper config loads, builds, and differs as advertised.

A config that silently fails to express its delta is worse than a missing one:
it produces a run, a number, and a table row that says something untrue. These
tests are cheap insurance against that, and they run without GalSim or a GPU
because building a Flax model needs neither.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "configs" / "paper"

_spec = importlib.util.spec_from_file_location("paper_generate", PAPER / "generate.py")
generate_configs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_configs)

ARMS = generate_configs.ALL_ARMS

#: build_model's keyword names paired with the config keys they come from.
_BUILD_KEYS = {
    "galaxy_type": "model.galaxy_branch",
    "psf_type": "model.psf_branch",
    "fusion": "model.fusion",
    "head": "model.head",
    "dropout": "model.dropout",
    "branch_features": "model.branch_features",
    "d4_features": "model.d4_features",
    "d4_depths_galaxy": "model.d4_depths_galaxy",
    "d4_depths_psf": "model.d4_depths_psf",
    "d4_multiscale": "model.d4_multiscale",
    "orbit_scan": "model.orbit_scan",
    "fusion_pos": "model.fusion_pos",
    "design": "model.design",
    "d_model": "model.d_model",
    "num_heads": "model.num_heads",
    "num_pool_heads": "model.num_pool_heads",
    "num_self_attn_layers": "model.num_self_attn_layers",
    "ffn_dim": "model.ffn_dim",
}


def _load(arm):
    from shearnet.config import Config

    return Config.from_file(PAPER / f"{arm.path}.yaml")


def _build_and_count(config):
    """Initialise the model this config describes; return its parameter count."""
    import jax
    import jax.numpy as jnp

    from shearnet.core.models import build_model, is_fork_model

    nn = config.get("model.type")
    kwargs = {name: config.get(key) for name, key in _BUILD_KEYS.items()}
    model = build_model(nn, **kwargs)
    stamp = jnp.zeros((2, 53, 53))
    inputs = (stamp, stamp) if is_fork_model(nn) else (stamp,)
    params = model.init(jax.random.PRNGKey(0), *inputs)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def _ids(arms):
    return [a.path for a in arms]


# ----------------------------------------------------------------------
# the generator and the files on disk agree
# ----------------------------------------------------------------------
def test_every_config_on_disk_matches_the_generator():
    """Hand-editing a generated config is the failure this catches.

    The header says "do not hand-edit"; this makes that enforceable rather than
    a request.
    """
    result = subprocess.run(
        [sys.executable, str(PAPER / "generate.py"), "--check"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_ladder_and_every_tier_is_present():
    paths = {arm.path for arm in ARMS}
    for rung in ("first", "second", "third", "fourth"):
        assert f"unit_tests/{rung}" in paths
    for tier in ("tier1", "tier2", "tier3", "tier4"):
        assert any(p.startswith(f"ablations/{tier}/") for p in paths), tier


def test_no_two_arms_share_a_directory_or_a_model_name():
    """A collision would have two runs write into one run directory."""
    paths = [arm.path for arm in ARMS]
    assert len(paths) == len(set(paths))
    names = [_load(arm).get("run_options.run_name") for arm in ARMS]
    assert len(names) == len(set(names)), sorted(names)
    outdirs = [_load(arm).get("run_options.outdir") for arm in ARMS]
    assert len(outdirs) == len(set(outdirs))


# ----------------------------------------------------------------------
# every config is loadable and buildable
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS, ids=_ids(ARMS))
def test_config_loads_and_declares_a_root_matching_its_directory(arm):
    config = _load(arm)
    assert config.get("run_options.outdir").endswith(arm.path), config.get("run_options.outdir")


@pytest.mark.parametrize("arm", [a for a in ARMS if a.path != "unit_tests/fourth"],
                         ids=_ids([a for a in ARMS if a.path != "unit_tests/fourth"]))
def test_every_arm_changes_what_the_code_reads(arm):
    """The failure the schema migration found: an arm whose delta nobody read.

    UT4 is excluded because it IS the fiducial simulation on the fiducial model.
    """
    from shearnet.config.schema import flatten

    fiducial = flatten(_load_fiducial().to_dict())
    resolved = flatten(_load(arm).to_dict())
    changed = {k for k in resolved
               if not k.startswith("run_options.") and resolved[k] != fiducial[k]}
    assert changed, arm.path


def _load_fiducial():
    from shearnet.config import Config

    return Config.from_file(generate_configs.FIDUCIAL)


@pytest.mark.parametrize("arm", ARMS, ids=_ids(ARMS))
def test_every_declared_delta_is_actually_in_the_file(arm):
    """The header claims a set of changes; this checks the YAML agrees.

    A delta that names a key the schema renames elsewhere would appear in the
    header and do nothing.
    """
    import yaml

    raw = yaml.safe_load((PAPER / f"{arm.path}.yaml").read_text())
    for dotted, expected in arm.delta.items():
        node = raw
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if expected is None:
            assert keys[-1] not in node, f"{dotted} should have been removed"
        else:
            assert node.get(keys[-1]) == expected, dotted


@pytest.mark.slow
@pytest.mark.parametrize("arm", ARMS, ids=_ids(ARMS))
def test_the_model_each_config_describes_actually_builds(arm):
    """Catches an architecture/branch/fusion combination that cannot exist."""
    assert _build_and_count(_load(arm)) > 0


# ----------------------------------------------------------------------
# the deltas that should change the network do change it
# ----------------------------------------------------------------------
def _fiducial_count():
    from shearnet.config import Config

    return _build_and_count(Config.from_file(generate_configs.FIDUCIAL))


@pytest.mark.slow
@pytest.mark.parametrize("path,direction", [
    ("ablations/tier4/no_multiscale_block", "fewer"),
    ("ablations/tier4/untrimmed_stem", "more"),
    ("ablations/tier4/full_resolution_psf_block", "more"),
])
def test_the_backbone_arms_move_the_parameter_count(path, direction):
    """A backbone ablation that leaves the network identical is a null result
    reported as a measurement. Each of these must actually change the model."""
    arm = next(a for a in ARMS if a.path == path)
    count = _build_and_count(_load(arm))
    fiducial = _fiducial_count()
    if direction == "fewer":
        assert count < fiducial, (count, fiducial)
    else:
        assert count > fiducial, (count, fiducial)


@pytest.mark.slow
def test_the_rope_encoding_removes_the_learned_embedding():
    """The fiducial's claim: relative RoPE has no parameters of its own.

    Rung 8 of the ladder is the fiducial with `fusion_pos: learned`, so the
    difference between them is exactly the absolute embedding table.
    """
    arm = next(a for a in ARMS
               if a.path == "ablations/tier2/08_learned_pooling_head")
    assert _build_and_count(_load(arm)) > _fiducial_count()


# ----------------------------------------------------------------------
# the arms mean what the tiers say
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", [a for a in ARMS if "/tier3/" in a.path],
                         ids=_ids([a for a in ARMS if "/tier3/" in a.path]))
def test_tier3_changes_exactly_one_response_key(arm):
    """A leave-one-out arm with two changes is not a leave-one-out."""
    assert len(arm.delta) == 1, arm.delta
    assert all(k.startswith("training.response.") for k in arm.delta), arm.delta


@pytest.mark.parametrize("arm", [a for a in ARMS if "/tier2/" in a.path],
                         ids=_ids([a for a in ARMS if "/tier2/" in a.path]))
def test_tier2_arms_warn_that_they_are_cumulative(arm):
    """The ladder is read by adjacent differences; the header must say so."""
    assert "cumulative" in arm.caveats


@pytest.mark.parametrize("arm", ARMS, ids=_ids(ARMS))
def test_every_arm_explains_itself(arm):
    """A config nobody can interpret in six months is a config nobody reruns."""
    assert len(arm.why.strip()) > 120, arm.path
    assert arm.title


#: The one non-dataset key the ladder is allowed to change, and where.
#: An ideal circular PSF makes the orbit penalty vacuous -- rotating it is the
#: identity -- and the trainer refuses the combination rather than let a term
#: that cannot act look like one that did. Switching it off at UT1 is forced BY
#: the simulation, not a free choice about the objective, and it changes no
#: gradient: the term contributes exactly zero there either way.
_LADDER_OBJECTIVE_EXCEPTIONS = {
    "unit_tests/first": {"training.response.orbit_weight"},
}


def test_the_unit_test_ladder_varies_only_the_simulation():
    """Table 1 is read down a column, which is valid only if the model is fixed.

    Every key of the four rungs' deltas must be a dataset key, except where a
    simulation choice makes an objective term structurally inapplicable -- and
    those exceptions are enumerated above rather than waved through.
    """
    ladder = [a for a in ARMS if a.path.startswith("unit_tests/")]
    assert len(ladder) == 4
    for arm in ladder:
        allowed = _LADDER_OBJECTIVE_EXCEPTIONS.get(arm.path, set())
        for key in arm.delta:
            if key in allowed:
                continue
            assert key.startswith("simulation."), (arm.path, key)


def test_the_ideal_psf_rung_switches_off_the_vacuous_orbit_term():
    """UT1 would otherwise refuse to start.

    inloop.py raises when orbit_weight is set on a circular Gaussian PSF. This
    is the failure that would have greeted UT1 immediately after the TypeError
    was fixed, so it is pinned rather than rediscovered.
    """
    arm = next(a for a in ARMS if a.path == "unit_tests/first")
    assert arm.delta.get("training.response.orbit_weight") == 0.0
    assert _load(arm).get("training.response.orbit_weight") == 0.0
    # ...and only at that rung: the other three have a real PSF to rotate.
    for rung in ("second", "third", "fourth"):
        other = next(a for a in ARMS if a.path == f"unit_tests/{rung}")
        assert "training.response.orbit_weight" not in other.delta


def test_the_ladder_rungs_agree_with_the_paper_table():
    """UT1 ideal PSF; UT2 adds the library; UT3 adds sizes; UT4 adds fluxes."""
    by_name = {a.path.rsplit("/", 1)[-1]: a.delta for a in ARMS
               if a.path.startswith("unit_tests/")}
    assert by_name["first"]["simulation.psf.mode"] == "ideal"
    for rung in ("second", "third", "fourth"):
        assert by_name[rung]["simulation.psf.mode"] == "superbit"
    assert by_name["second"]["simulation.hlr_type"] == "constant"
    assert by_name["third"]["simulation.hlr_type"] == "catalog"
    assert by_name["third"]["simulation.flux_type"] == "constant"
    assert by_name["fourth"]["simulation.flux_type"] == "catalog"


def test_a_blocked_arm_is_recorded_rather_than_quietly_skipped():
    """spatial_dropout would train an identical network on this backbone.

    Generating it anyway would produce a null result that looks like a
    measurement, so it is documented as blocked instead.
    """
    assert "tier4/spatial_dropout" in generate_configs.BLOCKED
    assert "no-op" in generate_configs.BLOCKED["tier4/spatial_dropout"]
