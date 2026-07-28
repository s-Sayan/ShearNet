"""Forward-shape contracts for every architecture and the model registry.

These need JAX/Flax but not ngmix at runtime, so they run anywhere the network
stack is installed.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp  # noqa: E402
import jax.random as random  # noqa: E402

from shearnet.core.models import (  # noqa: E402
    BRANCH_MODELS,
    SINGLE_BRANCH_MODELS,
    build_branch_model,
    build_model,
)

SINGLE_BRANCH = ["mlp", "cnn", "resnet", "research_backed", "forklens_psfnet"]


@pytest.mark.parametrize("nn", SINGLE_BRANCH)
@pytest.mark.parametrize("output_keys", [("g1", "g2"), ("g1", "g2", "hlr", "flux")])
def test_single_branch_forward_shape(nn, output_keys):
    """Each single-branch model maps (B, P, P) -> (B, len(output_keys))."""
    model = build_model(nn)
    x = jnp.ones((4, 21, 21))
    params = model.init(random.PRNGKey(0), x, output_keys=output_keys)
    preds = model.apply(params, x, output_keys=output_keys)
    assert preds.shape == (4, len(output_keys))


@pytest.mark.parametrize("fusion", ["concat", "transformer"])
def test_fork_like_forward_shape(fusion):
    """The two-branch model maps (gal, psf) -> (B, len(output_keys))."""
    model = build_model("fork-like", galaxy_type="cnn", psf_type="forklens_psf", fusion=fusion)
    gal = jnp.ones((4, 21, 21))
    psf = jnp.ones((4, 21, 21))
    output_keys = ("g1", "g2")
    params = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
    preds = model.apply(params, gal, psf, output_keys=output_keys)
    assert preds.shape == (4, len(output_keys))


@pytest.mark.parametrize("fusion", ["transformer", "concat"])
def test_d4_fork_like_forward_shape(fusion):
    """The D4-equivariant two-branch model maps (gal, psf) -> (B, n_keys)."""
    model = build_model("d4-fork-like", fusion=fusion)
    gal = jnp.ones((4, 24, 24))
    psf = jnp.ones((4, 24, 24))
    for output_keys in [("g1", "g2"), ("g1", "g2", "hlr", "flux")]:
        params = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
        preds = model.apply(params, gal, psf, output_keys=output_keys)
        assert preds.shape == (4, len(output_keys))


@pytest.mark.parametrize("fusion", ["transformer", "concat"])
def test_d4_fork_like_is_equivariant(fusion):
    """Outputs transform as a spin-2 vector under the D4 group.

    A 90-degree rotation of the (galaxy, PSF) pair must flip the sign of both
    shape components; an x-axis mirror must leave g1 unchanged and flip g2.
    This is hard-coded by the architecture (Lin et al. 2026), so it holds for
    randomly initialised weights up to float32 round-off.
    """
    model = build_model("d4-fork-like", fusion=fusion)
    gal = random.normal(random.PRNGKey(2), (3, 24, 24))
    psf = random.normal(random.PRNGKey(3), (3, 24, 24))
    output_keys = ("g1", "g2")
    params = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)

    out = model.apply(params, gal, psf, output_keys=output_keys, deterministic=True)

    # 90-degree rotation: e -> -e for both components.
    out_rot = model.apply(
        params,
        jnp.rot90(gal, 1, axes=(1, 2)),
        jnp.rot90(psf, 1, axes=(1, 2)),
        output_keys=output_keys,
        deterministic=True,
    )
    assert jnp.allclose(out_rot, -out, atol=1e-5)

    # x-axis mirror: g1 -> g1, g2 -> -g2.
    out_mir = model.apply(
        params,
        jnp.flip(gal, axis=1),
        jnp.flip(psf, axis=1),
        output_keys=output_keys,
        deterministic=True,
    )
    assert jnp.allclose(out_mir, out * jnp.array([1.0, -1.0]), atol=1e-5)


# Pluggable D4 branches: every (galaxy, psf) backbone pair must stay a valid,
# D4-equivariant estimator, because the Reynolds orbit average is exactly spin-2
# equivariant for an arbitrary square-map backbone (not just the smooth d4cnn).
D4_BRANCH_PAIRS = [
    ("d4cnn", "d4cnn"),
    ("research_backed", "forklens_psf"),
    ("research_backed", "d4cnn"),
    ("d4cnn", "forklens_psf"),
    ("forklens_psf", "research_backed"),
]


@pytest.mark.parametrize("galaxy_branch,psf_branch", D4_BRANCH_PAIRS)
def test_d4_fork_like_branches_forward_shape(galaxy_branch, psf_branch):
    """Each D4 branch pair maps (gal, psf) -> (B, n_keys)."""
    model = build_model(
        "d4-fork-like", galaxy_type=galaxy_branch, psf_type=psf_branch, fusion="transformer"
    )
    gal = jnp.ones((2, 24, 24))
    psf = jnp.ones((2, 24, 24))
    for output_keys in [("g1", "g2"), ("g1", "g2", "hlr", "flux")]:
        params = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
        preds = model.apply(params, gal, psf, output_keys=output_keys)
        assert preds.shape == (2, len(output_keys))


@pytest.mark.parametrize("galaxy_branch,psf_branch", D4_BRANCH_PAIRS)
def test_d4_fork_like_branches_are_equivariant(galaxy_branch, psf_branch):
    """Spin-2 equivariance holds for every branch pair, not just d4cnn.

    90-degree rotation flips the sign of both shape components; an x-axis mirror
    leaves g1 and flips g2. This is a property of the Reynolds average, so it
    holds for any square-map backbone at random init (up to float32 round-off).
    """
    model = build_model(
        "d4-fork-like", galaxy_type=galaxy_branch, psf_type=psf_branch, fusion="transformer"
    )
    gal = random.normal(random.PRNGKey(2), (3, 24, 24))
    psf = random.normal(random.PRNGKey(3), (3, 24, 24))
    output_keys = ("g1", "g2")
    params = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
    out = model.apply(params, gal, psf, output_keys=output_keys, deterministic=True)

    out_rot = model.apply(
        params,
        jnp.rot90(gal, 1, axes=(1, 2)),
        jnp.rot90(psf, 1, axes=(1, 2)),
        output_keys=output_keys,
        deterministic=True,
    )
    assert jnp.allclose(out_rot, -out, atol=1e-5)

    out_mir = model.apply(
        params,
        jnp.flip(gal, axis=1),
        jnp.flip(psf, axis=1),
        output_keys=output_keys,
        deterministic=True,
    )
    assert jnp.allclose(out_mir, out * jnp.array([1.0, -1.0]), atol=1e-5)


@pytest.mark.parametrize("head", ["gap", "attention"])
@pytest.mark.parametrize("galaxy_branch,psf_branch", [("d4cnn", "d4cnn"), ("research_backed", "forklens_psf")])
def test_d4_fork_like_head_is_equivariant(head, galaxy_branch, psf_branch):
    """Both pooling heads stay exactly spin-2 equivariant.

    The attention head derives its weights from the sign-free context map, so it
    rotates with psi1/psi2 and the pooled vector still transforms as w_c. The two
    extra outputs (hlr, flux) must be D4-INVARIANT (unchanged under the group).
    """
    model = build_model(
        "d4-fork-like", galaxy_type=galaxy_branch, psf_type=psf_branch,
        fusion="transformer", head=head,
    )
    gal = random.normal(random.PRNGKey(2), (3, 24, 24))
    psf = random.normal(random.PRNGKey(3), (3, 24, 24))
    ok = ("g1", "g2", "hlr", "flux")
    params = model.init(random.PRNGKey(0), gal, psf, output_keys=ok)
    out = model.apply(params, gal, psf, output_keys=ok, deterministic=True)

    out_rot = model.apply(
        params, jnp.rot90(gal, 1, axes=(1, 2)), jnp.rot90(psf, 1, axes=(1, 2)),
        output_keys=ok, deterministic=True,
    )
    # g1, g2 flip sign under 90-degree rotation; hlr, flux are invariant.
    assert jnp.allclose(out_rot[:, :2], -out[:, :2], atol=1e-5)
    assert jnp.allclose(out_rot[:, 2:], out[:, 2:], atol=1e-5)

    out_mir = model.apply(
        params, jnp.flip(gal, axis=1), jnp.flip(psf, axis=1),
        output_keys=ok, deterministic=True,
    )
    # mirror: g1 unchanged, g2 flips; hlr, flux invariant.
    assert jnp.allclose(out_mir[:, :2], out[:, :2] * jnp.array([1.0, -1.0]), atol=1e-5)
    assert jnp.allclose(out_mir[:, 2:], out[:, 2:], atol=1e-5)


def test_single_branch_accepts_unbatched_input():
    """A single 2-D stamp gets a batch axis added (shape (1, n))."""
    model = build_model("cnn")
    x2d = jnp.ones((21, 21))
    params = model.init(random.PRNGKey(0), x2d, output_keys=("g1", "g2"))
    preds = model.apply(params, x2d, output_keys=("g1", "g2"))
    assert preds.shape == (1, 2)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("does-not-exist")


def test_build_branch_model_unknown_raises():
    with pytest.raises(ValueError):
        build_branch_model("does-not-exist")


def test_registries_build_instances():
    """Every registered name instantiates the expected class."""
    for name, cls in SINGLE_BRANCH_MODELS.items():
        assert isinstance(build_model(name), cls)
    for name, cls in BRANCH_MODELS.items():
        assert isinstance(build_branch_model(name), cls)
