"""Every generated ablation config loads, builds, and differs as advertised.

A config that silently fails to express its delta is worse than a missing one:
it produces a run, a number, and a table row that says something untrue. These
tests are cheap insurance against that, and they run without GalSim or a GPU
because building a Flax model needs neither.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ablations"))

pytest.importorskip("yaml")
generate_configs = pytest.importorskip("generate_configs")

ARMS = generate_configs.ALL_ARMS

#: build_model's keyword names paired with the config keys they come from, and
#: the defaults the training CLI applies. Mirrors shearnet/cli/train.py.
_BUILD_KEYS = {
    "galaxy_type": ("model.galaxy.type", None),
    "psf_type": ("model.psf.type", None),
    "fusion": ("model.fusion", "concat"),
    "head": ("model.head", "gap"),
    "dropout": ("model.dropout", 0.0),
    "branch_features": ("model.branch_features", None),
    "d4_features": ("model.d4_features", None),
    "d4_depths_galaxy": ("model.d4_depths_galaxy", None),
    "d4_depths_psf": ("model.d4_depths_psf", None),
    "d4_multiscale": ("model.d4_multiscale", None),
    "orbit_scan": ("model.orbit_scan", True),
    "fusion_pos": ("model.fusion_pos", "learned"),
    "design": ("model.design", None),
    "d_model": ("model.d_model", None),
    "num_heads": ("model.num_heads", None),
    "num_pool_heads": ("model.num_pool_heads", None),
    "num_self_attn_layers": ("model.num_self_attn_layers", None),
    "ffn_dim": ("model.ffn_dim", None),
}


def _load(arm):
    from shearnet.config.config_handler import Config

    return Config(str(REPO / arm.path / "config.yaml"))


def _build_and_count(config):
    """Initialise the model this config describes; return its parameter count."""
    import jax
    import jax.numpy as jnp

    from shearnet.core.models import build_model, is_fork_model

    nn = config.get("model.type")
    kwargs = {name: config.get(key, default)
              for name, (key, default) in _BUILD_KEYS.items()}
    model = build_model(nn, **kwargs)
    stamp = jnp.zeros((2, 53, 53))
    inputs = (stamp, stamp) if is_fork_model(nn) else (stamp,)
    params = model.init(jax.random.PRNGKey(0), *inputs)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def _ids(arms):
    return [a.path.split("research/")[-1] for a in arms]


# ----------------------------------------------------------------------
# the generator and the files on disk agree
# ----------------------------------------------------------------------
def test_every_config_on_disk_matches_the_generator():
    """Hand-editing a generated config is the failure this catches.

    The header says "do not hand-edit"; this makes that enforceable rather than
    a request.
    """
    result = subprocess.run(
        [sys.executable, str(REPO / "research" / "ablations" / "generate_configs.py"),
         "--check"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_ladder_and_every_tier_is_present():
    paths = {arm.path for arm in ARMS}
    for rung in ("first", "second", "third", "fourth"):
        assert f"research/unit_tests/{rung}" in paths
    for tier in ("tier1", "tier2", "tier3", "tier4"):
        assert any(f"research/ablations/{tier}/" in p for p in paths), tier


def test_no_two_arms_share_a_directory_or_a_model_name():
    """A collision would have two runs overwrite each other's checkpoints."""
    paths = [arm.path for arm in ARMS]
    assert len(paths) == len(set(paths))
    names = [_load(arm).get("meta.model_name") for arm in ARMS]
    assert len(names) == len(set(names)), sorted(names)


# ----------------------------------------------------------------------
# every config is loadable and buildable
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS, ids=_ids(ARMS))
def test_config_loads_and_declares_a_root_matching_its_directory(arm):
    config = _load(arm)
    assert config.get("paths.root").endswith(arm.path), config.get("paths.root")


@pytest.mark.parametrize("arm", ARMS, ids=_ids(ARMS))
def test_the_removed_process_psf_key_is_not_carried_forward(arm):
    """It is inert now; carrying it would only warn on every run."""
    assert _load(arm).get("model.process_psf") is None


@pytest.mark.parametrize("arm", ARMS, ids=_ids(ARMS))
def test_every_declared_delta_is_actually_in_the_file(arm):
    """The header claims a set of changes; this checks the YAML agrees.

    A delta that names a key the schema renames elsewhere would appear in the
    header and do nothing.
    """
    import yaml

    raw = yaml.safe_load((REPO / arm.path / "config.yaml").read_text())
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
    from shearnet.config.config_handler import Config

    return _build_and_count(Config(str(generate_configs.FIDUCIAL)))


@pytest.mark.slow
@pytest.mark.parametrize("path,direction", [
    ("research/ablations/tier4/no_multiscale_block", "fewer"),
    ("research/ablations/tier4/untrimmed_stem", "more"),
    ("research/ablations/tier4/full_resolution_psf_block", "more"),
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
               if a.path == "research/ablations/tier2/08_learned_pooling_head")
    assert _build_and_count(_load(arm)) > _fiducial_count()


# ----------------------------------------------------------------------
# the arms mean what the tiers say
# ----------------------------------------------------------------------
@pytest.mark.parametrize("arm", [a for a in ARMS if "/tier3/" in a.path],
                         ids=_ids([a for a in ARMS if "/tier3/" in a.path]))
def test_tier3_changes_exactly_one_response_key(arm):
    """A leave-one-out arm with two changes is not a leave-one-out."""
    assert len(arm.delta) == 1, arm.delta
    assert all(k.startswith("train.response.") for k in arm.delta), arm.delta


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
    "research/unit_tests/first": {"train.response.orbit_weight"},
}


def test_the_unit_test_ladder_varies_only_the_simulation():
    """Table 1 is read down a column, which is valid only if the model is fixed.

    Every key of the four rungs' deltas must be a dataset key, except where a
    simulation choice makes an objective term structurally inapplicable -- and
    those exceptions are enumerated above rather than waved through.
    """
    ladder = [a for a in ARMS if a.path.startswith("research/unit_tests/")]
    assert len(ladder) == 4
    for arm in ladder:
        allowed = _LADDER_OBJECTIVE_EXCEPTIONS.get(arm.path, set())
        for key in arm.delta:
            if key in allowed:
                continue
            assert key.startswith(("psf.", "galaxy.", "image.")), (arm.path, key)


def test_the_ideal_psf_rung_switches_off_the_vacuous_orbit_term():
    """UT1 would otherwise refuse to start.

    inloop.py raises when orbit_weight is set on a circular Gaussian PSF. This
    is the failure that would have greeted UT1 immediately after the TypeError
    was fixed, so it is pinned rather than rediscovered.
    """
    arm = next(a for a in ARMS if a.path == "research/unit_tests/first")
    assert arm.delta.get("train.response.orbit_weight") == 0.0
    assert _load(arm).get("training.response.orbit_weight") == 0.0
    # ...and only at that rung: the other three have a real PSF to rotate.
    for rung in ("second", "third", "fourth"):
        other = next(a for a in ARMS if a.path == f"research/unit_tests/{rung}")
        assert "train.response.orbit_weight" not in other.delta


def test_the_ladder_rungs_agree_with_the_paper_table():
    """UT1 ideal PSF; UT2 adds the library; UT3 adds sizes; UT4 adds fluxes."""
    by_name = {a.path.rsplit("/", 1)[-1]: a.delta for a in ARMS
               if a.path.startswith("research/unit_tests/")}
    assert by_name["first"]["psf.mode"] == "ideal"
    for rung in ("second", "third", "fourth"):
        assert by_name[rung]["psf.mode"] == "superbit"
    assert by_name["second"]["galaxy.hlr_type"] == "constant"
    assert by_name["third"]["galaxy.hlr_type"] == "catalog"
    assert by_name["third"]["galaxy.flux_type"] == "constant"
    assert by_name["fourth"]["galaxy.flux_type"] == "catalog"


def test_a_blocked_arm_is_recorded_rather_than_quietly_skipped():
    """spatial_dropout would train an identical network on this backbone.

    Generating it anyway would produce a null result that looks like a
    measurement, so it is documented as blocked instead.
    """
    assert "tier4/spatial_dropout" in generate_configs.BLOCKED
    assert "no-op" in generate_configs.BLOCKED["tier4/spatial_dropout"]
