"""Forward-shape contracts for every architecture and the model registry.

These need JAX/Flax but not ngmix at runtime, so they run anywhere the network
stack is installed.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as random  # noqa: E402

from shearnet.core.models import (  # noqa: E402
    BRANCH_MODELS,
    SINGLE_BRANCH_MODELS,
    _ShearNetD4Backbone,
    attention_pool_diagnostics,
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
    ("shearnet-d4", "shearnet-d4"),
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


@pytest.mark.parametrize("galaxy_branch,psf_branch", D4_BRANCH_PAIRS)
def test_orbit_scan_is_the_same_model_as_the_stacked_orbit(galaxy_branch, psf_branch):
    """``orbit_scan`` is a memory layout, not a different network.

    Stacking the eight orbit members on the batch axis and scanning over them
    with shared parameters compute the same Reynolds average; the scan just
    does it one member at a time and rematerialises the internals in the
    backward pass. So the parameter tree has to be *identical* -- not merely
    the same shapes, or a checkpoint would not load across the switch -- and
    the outputs have to agree to float reassociation.
    """
    kw = dict(galaxy_type=galaxy_branch, psf_type=psf_branch, fusion="transformer")
    stacked = build_model("d4-fork-like", orbit_scan=False, **kw)
    scanned = build_model("d4-fork-like", orbit_scan=True, **kw)
    gal = random.normal(random.PRNGKey(2), (3, 24, 24))
    psf = random.normal(random.PRNGKey(3), (3, 24, 24))
    output_keys = ("g1", "g2", "hlr", "flux")

    p_stacked = stacked.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
    p_scanned = scanned.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)

    stacked_leaves = jax.tree_util.tree_leaves_with_path(p_stacked)
    scanned_leaves = jax.tree_util.tree_leaves_with_path(p_scanned)
    assert [k for k, _ in stacked_leaves] == [k for k, _ in scanned_leaves]
    for (_, a), (_, b) in zip(stacked_leaves, scanned_leaves):
        assert a.shape == b.shape
        assert jnp.array_equal(a, b)

    # and the same params run through either layout give the same answer
    a = stacked.apply(p_stacked, gal, psf, output_keys=output_keys, deterministic=True)
    b = scanned.apply(p_stacked, gal, psf, output_keys=output_keys, deterministic=True)
    assert jnp.allclose(a, b, atol=1e-5)


def test_orbit_scan_keeps_the_equivariance():
    """The scan must not break the property the orbit exists to provide."""
    model = build_model(
        "d4-fork-like",
        galaxy_type="shearnet-d4",
        psf_type="shearnet-d4",
        fusion="transformer",
        orbit_scan=True,
    )
    gal = random.normal(random.PRNGKey(2), (3, 24, 24))
    psf = random.normal(random.PRNGKey(3), (3, 24, 24))
    output_keys = ("g1", "g2", "hlr", "flux")
    params = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
    out = model.apply(params, gal, psf, output_keys=output_keys, deterministic=True)
    out_rot = model.apply(
        params,
        jnp.rot90(gal, 1, axes=(1, 2)),
        jnp.rot90(psf, 1, axes=(1, 2)),
        output_keys=output_keys,
        deterministic=True,
    )
    # shape components flip sign; hlr and flux are scalars and must not move
    assert jnp.allclose(out_rot[:, :2], -out[:, :2], atol=1e-5)
    assert jnp.allclose(out_rot[:, 2:], out[:, 2:], atol=1e-5)


@pytest.mark.parametrize(
    "d4_features,d4_depths_galaxy,d4_depths_psf",
    [
        ((32, 48, 64), (2, 2, 1), (1, 1, 1)),  # the report's schedule
        ((16, 48, 64), (1, 2, 1), (0, 1, 1)),  # the trimmed one
        ((16, 24, 32), (1, 1, 1), (0, 0, 1)),  # aggressively trimmed
    ],
)
def test_d4_schedule_is_configurable_and_stays_equivariant(
    d4_features, d4_depths_galaxy, d4_depths_psf
):
    """Trimming the stage widths and depths must not cost the symmetry.

    A depth of zero at a stage means that stage is a pooling and a 1x1
    transition with no residual block, which is the cheapest way to drop the
    full-resolution activations the orbit multiplies by eight. Equivariance
    comes from the Reynolds average over the orbit, not from the backbone, so
    it must survive any schedule -- including one with an empty first stage.
    """
    model = build_model(
        "d4-fork-like",
        galaxy_type="shearnet-d4",
        psf_type="shearnet-d4",
        fusion="transformer",
        d4_features=d4_features,
        d4_depths_galaxy=d4_depths_galaxy,
        d4_depths_psf=d4_depths_psf,
    )
    gal = random.normal(random.PRNGKey(2), (2, 24, 24))
    psf = random.normal(random.PRNGKey(3), (2, 24, 24))
    output_keys = ("g1", "g2")
    params = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
    out = model.apply(params, gal, psf, output_keys=output_keys, deterministic=True)
    assert out.shape == (2, 2)

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


def test_trimmed_d4_schedule_is_actually_smaller():
    """The knobs have to buy something, or they are just extra surface area."""
    gal = jnp.ones((2, 24, 24))
    psf = jnp.ones((2, 24, 24))

    def n_params(**kw):
        model = build_model(
            "d4-fork-like",
            galaxy_type="shearnet-d4",
            psf_type="shearnet-d4",
            fusion="transformer",
            **kw,
        )
        params = model.init(random.PRNGKey(0), gal, psf, output_keys=("g1", "g2"))
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    report = n_params()
    trimmed = n_params(
        d4_features=(16, 48, 64), d4_depths_galaxy=(1, 2, 1), d4_depths_psf=(0, 1, 1)
    )
    assert trimmed < report


@pytest.mark.parametrize("head", ["gap", "attention"])
@pytest.mark.parametrize(
    "galaxy_branch,psf_branch",
    [("d4cnn", "d4cnn"), ("shearnet-d4", "shearnet-d4"), ("research_backed", "forklens_psf")],
)
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


def test_shearnet_d4_branch_reaches_fusion_at_13x13():
    """The shearnet-d4 backbones take a 53x53 stamp to the 13x13x64 fusion map.

    This is the point of the branch: two anti-aliased downsamplings instead of
    the five avg-pools of a five-layer d4cnn, so cross-attention sees 169
    spatial tokens rather than one. The PSF variant is the lighter one -- same
    shapes, fewer residual blocks and no dilated context block -- so it must
    have strictly fewer parameters than the galaxy variant.
    """
    stamp = jnp.ones((2, 53, 53))
    galaxy = _ShearNetD4Backbone()
    psf = _ShearNetD4Backbone(depths=(1, 1, 1), multiscale=False)

    gal_params = galaxy.init(random.PRNGKey(0), stamp)
    psf_params = psf.init(random.PRNGKey(0), stamp)
    assert galaxy.apply(gal_params, stamp).shape == (2, 13, 13, 64)
    assert psf.apply(psf_params, stamp).shape == (2, 13, 13, 64)

    count = lambda p: sum(x.size for x in jax.tree_util.tree_leaves(p))  # noqa: E731
    assert count(psf_params) < count(gal_params)


def test_shearnet_d4_design_only_applies_to_its_own_branch():
    """Selecting shearnet-d4 brings its fusion FFN and deeper odd heads with it.

    The report specifies branches, fusion and heads as one architecture, so the
    branch name sets ``design``; every other branch keeps the existing layout
    and therefore the existing parameter tree.
    """
    kw = dict(fusion="transformer", head="attention")
    assert build_model("d4-fork-like", galaxy_type="shearnet-d4", **kw).design == "shearnet-d4"
    assert build_model("d4-fork-like", galaxy_type="d4cnn", **kw).design == "d4cnn"

    gal = jnp.ones((2, 24, 24))
    ok = ("g1", "g2", "hlr", "flux")
    model = build_model("d4-fork-like", galaxy_type="shearnet-d4", psf_type="shearnet-d4", **kw)
    params = model.init(random.PRNGKey(0), gal, gal, output_keys=ok)["params"]
    # Sec. 10.1: 256 -> 128 -> 128 -> 1, so three bias-free layers per component.
    assert {"odd_e1_dense0", "odd_e1_dense1", "odd_e1_dense2"} <= set(params)
    for layer in ("odd_e1_dense0", "odd_e1_dense1", "odd_e1_dense2"):
        assert "bias" not in params[layer]
    # Sec. 10.2: one final linear layer per invariant scalar.
    assert {"scalar_hlr", "scalar_flux"} <= set(params)


def test_attention_pool_maps_are_explicitly_capturable():
    """Diagnostics expose all four maps without changing normal predictions."""
    model = build_model(
        "d4-fork-like",
        galaxy_type="shearnet-d4",
        psf_type="shearnet-d4",
        fusion="transformer",
        head="attention",
    )
    gal = random.normal(random.PRNGKey(2), (2, 24, 24))
    psf = random.normal(random.PRNGKey(3), (2, 24, 24))
    output_keys = ("g1", "g2")
    variables = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
    assert "intermediates" not in variables

    ordinary = model.apply(variables, gal, psf, output_keys=output_keys)
    captured_pred, captured = model.apply(
        variables,
        gal,
        psf,
        output_keys=output_keys,
        capture_attention=True,
        mutable=["intermediates"],
    )
    maps = captured["intermediates"]["pool_attention"][0]
    assert maps.shape == (2, 6, 6, 4)
    assert jnp.allclose(jnp.sum(maps, axis=(1, 2)), 1.0, atol=1e-6)
    assert jnp.allclose(captured_pred, ordinary, atol=1e-7)


def test_attention_pool_diagnostics_identify_head_collapse():
    """Identical maps have unit similarity and one effective head."""
    maps = jnp.full((3, 2, 2, 4), 0.25)
    diagnostics = attention_pool_diagnostics(maps)
    assert jnp.allclose(diagnostics["entropy"], 1.0)
    assert jnp.allclose(diagnostics["similarity"], jnp.ones((4, 4)))
    assert float(diagnostics["max_similarity"]) == pytest.approx(1.0)
    assert float(diagnostics["effective_rank"]) == pytest.approx(1.0, abs=1e-5)


def test_attention_pool_diagnostics_distinguish_independent_heads():
    """Four disjoint maps have zero overlap and four effective heads."""
    maps = jnp.eye(4).reshape(1, 2, 2, 4)
    diagnostics = attention_pool_diagnostics(maps)
    assert jnp.allclose(diagnostics["entropy"], 0.0)
    assert jnp.allclose(diagnostics["similarity"], jnp.eye(4))
    assert float(diagnostics["max_similarity"]) == pytest.approx(0.0)
    assert float(diagnostics["effective_rank"]) == pytest.approx(4.0, abs=1e-5)


def test_attention_pool_diagnostics_run_under_jit():
    """The only caller is jitted, so the diagnostics must trace.

    ``train_inloop.attention_report`` wraps this in ``jax.jit``, so anything
    whose output shape depends on array *values* -- boolean mask indexing in
    particular -- fails at the first validation pass rather than at import. The
    three tests above all call the function eagerly, which is why that failure
    reached a GPU run. This one traces it.
    """
    jitted = jax.jit(attention_pool_diagnostics)
    for maps in (jnp.full((3, 2, 2, 4), 0.25), jnp.eye(4).reshape(1, 2, 2, 4)):
        eager = attention_pool_diagnostics(maps)
        traced = jitted(maps)
        assert set(traced) == set(eager)
        for key, value in eager.items():
            assert jnp.allclose(traced[key], value, atol=1e-6), key


def test_attention_pool_diagnostics_handle_a_single_head():
    """One head has no off-diagonal, so the similarity statistics are zero."""
    maps = jnp.full((2, 3, 3, 1), 1.0 / 9.0)
    for diagnostics in (attention_pool_diagnostics(maps),
                        jax.jit(attention_pool_diagnostics)(maps)):
        assert float(diagnostics["mean_similarity"]) == pytest.approx(0.0)
        assert float(diagnostics["max_similarity"]) == pytest.approx(0.0)
        assert float(diagnostics["effective_rank"]) == pytest.approx(1.0, abs=1e-5)


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


# ----------------------------------------------------------------------
# relative 2D RoPE in the fusion attention
# ----------------------------------------------------------------------
def _rope_scores(x_q, x_k, pos_q, pos_k, freqs):
    """Single-head attention logits under the 2D RoPE, without the softmax."""
    from shearnet.core.models import _apply_rope2d

    q = _apply_rope2d(x_q[None, :, None, :], pos_q, freqs)[0, :, 0]
    k = _apply_rope2d(x_k[None, :, None, :], pos_k, freqs)[0, :, 0]
    return q @ k.T


def test_rope2d_score_depends_only_on_the_displacement():
    """The convolution argument, made structural.

    The observation is galaxy (*) PSF, so the PSF's contribution at offset d
    depends on d alone. A positional encoding that respects that has to give an
    attention score which is unchanged when both grids are translated together.
    """
    from shearnet.core.models import _rope2d_frequencies, _rope2d_positions

    h = w = 13
    d_head = 16
    freqs = _rope2d_frequencies(d_head // 4, h - 1)
    pos = _rope2d_positions(h, w)
    q = random.normal(random.PRNGKey(1), (h * w, d_head))
    k = random.normal(random.PRNGKey(2), (h * w, d_head))

    base = _rope_scores(q, k, pos, pos, freqs)
    shift = jnp.array([3.0, -2.0])
    moved = _rope_scores(q, k, pos + shift, pos + shift, freqs)

    scale = float(jnp.std(base))
    assert scale > 1.0, "degenerate test: the scores carry no signal"
    assert float(jnp.max(jnp.abs(base - moved))) < 1e-4 * scale


def test_rope2d_is_d4_covariant_only_with_shared_axis_frequencies():
    """The one constraint D4 imposes on the encoding, and its control.

    Under the 90-degree element d = (du, dv) -> (-dv, du), so the rotated x-pair
    carries -w*dv and the rotated y-pair w*du: the original with the pairs
    swapped and one conjugated. That is realisable on the channel space by
    rho(r): (X, Y) -> (conj(Y), X) -- a signed permutation of four real
    coordinates, hence orthogonal, hence score-preserving -- but ONLY if both
    pairs of a group share the frequency. Independent per-axis schedules (the
    usual "axial RoPE") admit no such map, and this pins that difference rather
    than trusting the derivation.
    """
    from shearnet.core.models import _rope2d_frequencies, _rope2d_positions

    h = w = 13
    d_head = 16
    freqs = _rope2d_frequencies(d_head // 4, h - 1)
    pos = _rope2d_positions(h, w)
    q = random.normal(random.PRNGKey(1), (h * w, d_head))
    k = random.normal(random.PRNGKey(2), (h * w, d_head))

    def rho_r(x):
        xr = x.reshape(*x.shape[:-1], -1, 4)
        x_a, x_b, y_a, y_b = xr[..., 0], xr[..., 1], xr[..., 2], xr[..., 3]
        return jnp.stack([y_a, -y_b, x_a, x_b], axis=-1).reshape(x.shape)

    rot = jnp.rot90(jnp.arange(h * w).reshape(h, w), 1).reshape(-1)
    base = _rope_scores(q, k, pos, pos, freqs)
    scale = float(jnp.std(base))

    covariant = _rope_scores(rho_r(q)[rot], rho_r(k)[rot], pos, pos, freqs)
    assert float(jnp.max(jnp.abs(covariant - base[jnp.ix_(rot, rot)]))) < 1e-4 * scale

    # the control: tune the two axes independently and covariance is destroyed
    def split_scores(x_q, x_k):
        fx, fy = freqs, freqs * 1.7
        n = d_head // 4

        def enc(x, p):
            ang = jnp.stack(
                [p[:, 0][:, None] * fx[None, :], p[:, 1][:, None] * fy[None, :]], axis=-1
            ).reshape(p.shape[0], 2 * n)
            xr = x.reshape(*x.shape[:-1], 2 * n, 2)
            a, b = xr[..., 0], xr[..., 1]
            c, s = jnp.cos(ang), jnp.sin(ang)
            return jnp.stack([a * c - b * s, a * s + b * c], axis=-1).reshape(x.shape)

        return enc(x_q, pos) @ enc(x_k, pos).T

    split = split_scores(q, k)
    split_rot = split_scores(rho_r(q)[rot], rho_r(k)[rot])
    assert float(jnp.max(jnp.abs(split_rot - split[jnp.ix_(rot, rot)]))) > 0.5 * scale


def test_rope2d_frequencies_are_set_by_the_grid_not_a_text_default():
    """Frequencies must resolve the map, not a thousand-token sequence."""
    from shearnet.core.models import _rope2d_frequencies

    freqs = _rope2d_frequencies(4, extent=12, min_cycles=1.0, max_cycles=3.0)
    cycles = np.asarray(freqs) * 12 / (2 * np.pi)
    assert np.isclose(cycles[0], 1.0) and np.isclose(cycles[-1], 3.0)
    # every frequency completes at least one cycle over the map: none is inert
    assert (cycles >= 1.0).all()
    assert _rope2d_frequencies(1, extent=12).shape == (1,)


@pytest.mark.parametrize("fusion_pos", ["learned", "rope2d", "none"])
def test_fusion_pos_keeps_the_model_equivariant(fusion_pos):
    """Whatever the encoding, the spin-2 contract is not negotiable."""
    model = build_model(
        "d4-fork-like",
        galaxy_type="shearnet-d4",
        psf_type="shearnet-d4",
        fusion="transformer",
        fusion_pos=fusion_pos,
    )
    gal = random.normal(random.PRNGKey(2), (2, 24, 24))
    psf = random.normal(random.PRNGKey(3), (2, 24, 24))
    output_keys = ("g1", "g2", "hlr", "flux")
    params = model.init(random.PRNGKey(0), gal, psf, output_keys=output_keys)
    out = model.apply(params, gal, psf, output_keys=output_keys, deterministic=True)
    assert out.shape == (2, 4)

    out_rot = model.apply(
        params,
        jnp.rot90(gal, 1, axes=(1, 2)),
        jnp.rot90(psf, 1, axes=(1, 2)),
        output_keys=output_keys,
        deterministic=True,
    )
    assert jnp.allclose(out_rot[:, :2], -out[:, :2], atol=1e-5)
    assert jnp.allclose(out_rot[:, 2:], out[:, 2:], atol=1e-5)

    out_mir = model.apply(
        params,
        jnp.flip(gal, axis=1),
        jnp.flip(psf, axis=1),
        output_keys=output_keys,
        deterministic=True,
    )
    assert jnp.allclose(out_mir[:, :2], out[:, :2] * jnp.array([1.0, -1.0]), atol=1e-5)


def test_fusion_pos_default_is_the_historical_learned_table():
    """The default must stay 'learned', or every saved checkpoint stops loading."""
    kw = dict(
        galaxy_type="shearnet-d4", psf_type="shearnet-d4", fusion="transformer"
    )
    gal, psf = jnp.ones((2, 24, 24)), jnp.ones((2, 24, 24))

    def tree(**extra):
        model = build_model("d4-fork-like", **kw, **extra)
        p = model.init(random.PRNGKey(0), gal, psf, output_keys=("g1", "g2"))
        return jax.tree_util.tree_leaves_with_path(p)

    default, explicit = tree(), tree(fusion_pos="learned")
    assert [k for k, _ in default] == [k for k, _ in explicit]
    for (_, a), (_, b) in zip(default, explicit):
        assert jnp.array_equal(a, b)

    names = " ".join(str(k) for k, _ in default)
    assert "pos_embed" in names, "the learned table should exist under 'learned'"
    for pos in ("rope2d", "none"):
        assert "pos_embed" not in " ".join(str(k) for k, _ in tree(fusion_pos=pos))


def test_unknown_fusion_pos_is_rejected():
    model = build_model(
        "d4-fork-like", galaxy_type="shearnet-d4", psf_type="shearnet-d4",
        fusion="transformer", fusion_pos="sinusoidal",
    )
    with pytest.raises(ValueError, match="fusion_pos"):
        model.init(random.PRNGKey(0), jnp.ones((1, 24, 24)), jnp.ones((1, 24, 24)))
