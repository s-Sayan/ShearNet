"""Generate every ablation config as a one-key delta from the fiducial model.

The fiducial config -- ``research/unit_test_variations/fourth_inloop_shearnet_d4_2drope``
-- is the single source of truth. Every arm here is that file with a small,
named set of keys changed, and each arm's header says which keys and why.
Writing thirty configs by hand guarantees they drift; generating them means a
change to the fiducial propagates by re-running this script.

    python research/ablations/generate_configs.py            # write everything
    python research/ablations/generate_configs.py --check    # verify, write nothing

An arm is only a measurement if it differs from the fiducial in ONE thing. Where
that is not achievable -- the architecture ladder builds up cumulatively, so its
rungs differ in several -- the header says so explicitly rather than pretending
otherwise.

The paths in the fiducial config are absolute and belong to the machine the runs
happen on. They are rewritten per arm here; everything else is inherited.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import yaml

REPO = Path(__file__).resolve().parents[2]
FIDUCIAL = (REPO / "research" / "unit_test_variations"
            / "fourth_inloop_shearnet_d4_2drope" / "config.yaml")
#: The runs happen here, not in the container this was written in.
CLUSTER_ROOT = "/home/adfield/ShearNet"


class Arm(NamedTuple):
    """One config to write.

    Attributes:
        path: directory under the repo root, relative.
        title: one line for ``meta.description`` and the header.
        why: what the arm establishes. Prose, several lines.
        delta: dotted key -> value. ``None`` deletes the key.
        caveats: anything the reader must know before trusting the result.
    """

    path: str
    title: str
    why: str
    delta: Dict[str, Any]
    caveats: str = ""


# ----------------------------------------------------------------------
# the dataset ladder (research/unit_tests) -- model fixed, data varies
# ----------------------------------------------------------------------
#: Everything the fiducial config says about the network, so the four rungs
#: differ in the simulation and in nothing else. Table 1 of the paper reads down
#: this ladder to localize which ingredient of realism moves the calibration, and
#: that reading is only valid if the estimator is held fixed.
LADDER = [
    Arm(
        path="research/unit_tests/first",
        title="UT1: ideal Gaussian PSF, fixed size and flux",
        why="""Shear recovery in isolation. The PSF is a circular Gaussian of
0.5 arcsec FWHM and every galaxy has the same half-light radius and flux, so
the only thing varying across the population is the intrinsic shape and the
applied shear. A bias here is a bias in the estimator itself and cannot be
blamed on the PSF model or on the size-flux distribution.""",
        delta={
            "psf.mode": "ideal",
            "galaxy.hlr_type": "constant",
            "galaxy.flux_type": "constant",
            "galaxy.hlr": 0.5,
            "galaxy.flux": 12258.97,
        },
        caveats="""hlr and flux are constant here while `output_keys` still asks
for them. That is deliberate -- the network must be identical across the ladder
-- and it is safe: fit_normalizer guards a zero standard deviation (it
substitutes 1.0), so the auxiliary targets contribute a constant term rather
than a division by zero. They are simply trivial to predict at this rung.""",
    ),
    Arm(
        path="research/unit_tests/second",
        title="UT2: SuperBIT PSFEx library, fixed size and flux",
        why="""Adds the anisotropic, spatially varying PSF and nothing else.
Comparing against UT1 isolates what an empirical PSF costs the estimator,
before any spread in galaxy size or brightness is introduced. This is the rung
where PSF leakage first becomes measurable.""",
        delta={
            "psf.mode": "superbit",
            "galaxy.hlr_type": "constant",
            "galaxy.flux_type": "constant",
            "galaxy.hlr": 0.5,
            "galaxy.flux": 12258.97,
        },
    ),
    Arm(
        path="research/unit_tests/third",
        title="UT3: PSFEx library, COSMOS sizes, fixed flux",
        why="""Adds the spread in galaxy size, and with it the spread in
T/T_PSF: the population now spans marginally resolved through extended. Flux is
still fixed, so signal-to-noise is nearly constant and any change against UT2 is
resolution, not depth.""",
        delta={
            "psf.mode": "superbit",
            "galaxy.hlr_type": "catalog",
            "galaxy.flux_type": "constant",
            "galaxy.hlr": "catalog",
            "galaxy.flux": 12258.97,
        },
    ),
    Arm(
        path="research/unit_tests/fourth",
        title="UT4: the fiducial simulation",
        why="""Adds the catalog flux distribution, so size and signal-to-noise
now vary jointly as they do in the COSMOS detection sample. This is the
population every headline number in the paper is measured on, and it is the same
simulation the fiducial model trains on.""",
        delta={
            "psf.mode": "superbit",
            "galaxy.hlr_type": "catalog",
            "galaxy.flux_type": "catalog",
            "galaxy.hlr": "catalog",
            "galaxy.flux": "catalog",
        },
    ),
]


# ----------------------------------------------------------------------
# Tier 1 -- the spine
# ----------------------------------------------------------------------
TIER1 = [
    Arm(
        path="research/ablations/tier1/no_psf_response",
        title="Tier 1: the PSF response penalty removed",
        why="""The single most load-bearing ablation in the paper. lambda_PSF is
the training-time analogue of the leakage measurement, so removing it makes a
falsifiable prediction: alpha should rise. If it does, the objective is what
controls leakage and the method's central claim holds. If it does not, leakage
is controlled by the D4 architecture instead and the paper should say so.

Either outcome is publishable. Not running it leaves the central claim
untested, which is why this sits in Tier 1 rather than with the other
leave-one-out arms in Tier 3.""",
        delta={"train.response.psf_weight": 0.0},
    ),
]


# ----------------------------------------------------------------------
# Tier 2 -- the architecture ladder, built up one component at a time
# ----------------------------------------------------------------------
#: Rungs 1-6 predate the differentiable renderer, so they train on a fixed
#: up-front dataset with no response penalties: those terms need the renderer
#: inside the autodiff graph, which arrives at rung 7. Removing the whole
#: `train.response` block is therefore not an extra variable, it is what "before
#: in-loop rendering" means.
#:
#: The BACKEND stays jax-galsim. An earlier version of this also set
#: `backend: galsim`, conflating "renders up front" with "renders with a
#: different library". They are independent, and the evaluation refuses any run
#: whose backend is not jax-galsim (run.py: R^PSF is a finite difference on the
#: PSF shear, which needs jax-galsim's explicit per-object psf_g1/psf_g2). Those
#: six arms would each have trained to completion and then failed at the
#: measurement. `python research/ablations/preflight.py` now checks this
#: statically for every config.
_PRE_INLOOP = {
    "train.generation": "upfront",
    "train.response": None,
}

_LADDER_CAVEAT = """This is a cumulative ladder, not a leave-one-out: each rung
adds one component to the rung above it, so a row differs from the FIDUCIAL in
several keys while differing from its immediate predecessor in one. Read the
table by adjacent differences, not against the last row."""

TIER2 = [
    Arm(
        path="research/ablations/tier2/01_galaxy_only",
        title="Tier 2.1: galaxy image only, no PSF branch",
        why="""The baseline the ladder builds on. A single-branch network sees
the observed stamp and nothing else, so it must infer the PSF from the galaxy
image or ignore it. Everything above this row is an argument that giving the
network the PSF explicitly is worth its cost.""",
        delta={
            **_PRE_INLOOP,
            "model.architecture": "research_backed",
            "model.output_keys": ["g1", "g2"],
            "train.loss_weights": [1.0, 1.0],
            "model.head": "gap",
            "model.fusion": None,
            "model.galaxy_branch": None,
            "model.psf_branch": None,
            "model.design": None,
            "model.fusion_pos": None,
            "model.d_model": None,
            "model.num_heads": None,
            "model.num_self_attn_layers": None,
            "model.ffn_dim": None,
            "model.num_pool_heads": None,
            "model.d4_features": None,
            "model.d4_depths_galaxy": None,
            "model.d4_depths_psf": None,
            "model.orbit_scan": None,
        },
        caveats=_LADDER_CAVEAT,
    ),
    Arm(
        path="research/ablations/tier2/02_psf_branch_concat",
        title="Tier 2.2: + PSF branch, concatenation fusion",
        why="""The PSF enters on its own pathway, summarised to a global vector
and concatenated onto the galaxy features. This is the conventional two-branch
design. Because the concatenated context is one vector for the whole map, the
correction it can apply is fixed across the stamp -- which is what the next rung
changes.""",
        delta={
            **_PRE_INLOOP,
            "model.architecture": "fork-like",
            "model.fusion": "concat",
            "model.galaxy_branch": "research_backed",
            "model.psf_branch": "forklens_psf",
            "model.output_keys": ["g1", "g2"],
            "train.loss_weights": [1.0, 1.0],
            "model.head": "gap",
            "model.design": None,
            "model.fusion_pos": None,
            "model.d_model": None,
            "model.num_heads": None,
            "model.num_self_attn_layers": None,
            "model.ffn_dim": None,
            "model.num_pool_heads": None,
            "model.d4_features": None,
            "model.d4_depths_galaxy": None,
            "model.d4_depths_psf": None,
            "model.orbit_scan": None,
        },
        caveats=_LADDER_CAVEAT,
    ),
    Arm(
        path="research/ablations/tier2/03_transformer_fusion",
        title="Tier 2.3: + cross-attention fusion",
        why="""Each galaxy location queries the whole PSF map and reads what is
relevant to it, so the correction becomes position-dependent. This is the
operation that stands in for an explicit deconvolution, and the difference
against rung 2 is the measurement of whether that matters.""",
        delta={
            **_PRE_INLOOP,
            "model.architecture": "fork-like",
            "model.fusion": "transformer",
            "model.galaxy_branch": "research_backed",
            "model.psf_branch": "forklens_psf",
            "model.output_keys": ["g1", "g2"],
            "train.loss_weights": [1.0, 1.0],
            "model.head": "gap",
            "model.design": None,
            "model.fusion_pos": None,
            "model.d4_features": None,
            "model.d4_depths_galaxy": None,
            "model.d4_depths_psf": None,
            "model.orbit_scan": None,
        },
        caveats=_LADDER_CAVEAT,
    ),
    Arm(
        path="research/ablations/tier2/04_auxiliary_targets",
        title="Tier 2.4: + auxiliary size and flux targets",
        why="""The network is asked to predict half-light radius and flux
alongside the shear. Neither is reported as a science product; the question is
whether being made to represent them improves the shear, which is the usual
multi-task argument and worth testing rather than assuming.""",
        delta={
            **_PRE_INLOOP,
            "model.architecture": "fork-like",
            "model.fusion": "transformer",
            "model.galaxy_branch": "research_backed",
            "model.psf_branch": "forklens_psf",
            "model.head": "gap",
            "model.design": None,
            "model.fusion_pos": None,
            "model.d4_features": None,
            "model.d4_depths_galaxy": None,
            "model.d4_depths_psf": None,
            "model.orbit_scan": None,
        },
        caveats=_LADDER_CAVEAT,
    ),
    Arm(
        path="research/ablations/tier2/05_d4_augmentation",
        title="Tier 2.5: + D4 symmetry by augmentation",
        why="""The eight D4 transforms are applied as training augmentation
rather than built into the architecture. The network is then encouraged toward
the symmetry but not held to it, so the orientation-dependent additive bias is
suppressed on average and not identically. The contrast with rung 6 is the
paper's argument for construction over augmentation, and it is visible in the
rotation test as well as in c2.""",
        delta={
            **_PRE_INLOOP,
            "model.architecture": "fork-like",
            "model.fusion": "transformer",
            "model.galaxy_branch": "research_backed",
            "model.psf_branch": "forklens_psf",
            "model.head": "gap",
            "train.d4_augment": True,
            "model.design": None,
            "model.fusion_pos": None,
            "model.d4_features": None,
            "model.d4_depths_galaxy": None,
            "model.d4_depths_psf": None,
            "model.orbit_scan": None,
        },
        caveats=_LADDER_CAVEAT + """

`d4_augment` is an ablation control and is explicitly not for use with
d4-fork-like, where the symmetry is already exact and the augmentation would be
a no-op costing 8x the data.""",
    ),
    Arm(
        path="research/ablations/tier2/06_d4_equivariant",
        title="Tier 2.6: + D4 symmetry by construction",
        why="""The Reynolds average over the eight-element orbit, which makes
the predicted shear exactly spin-2 equivariant for any square-map backbone --
measured to 2.4e-07 at initialization. Against rung 5 this separates a symmetry
that holds on average from one that holds identically. Note what it does NOT
give you: every D4 element acts on (e1, e2) as diag(+/-1, +/-1), so it forces
the off-diagonal response to zero and leaves R11 and R22 completely unrelated.""",
        delta={
            **_PRE_INLOOP,
            "model.head": "gap",
            "model.fusion_pos": "learned",
            "model.num_pool_heads": None,
        },
        caveats=_LADDER_CAVEAT,
    ),
    Arm(
        path="research/ablations/tier2/07_inloop_rendering",
        title="Tier 2.7: + in-loop rendering and the response objectives",
        why="""The renderer moves inside the autodiff graph. Noise is redrawn
every step rather than fixed once, and the six response penalties become
computable because the derivative of the image with respect to shear, PSF
ellipticity and position now exists. This rung is where the method's central
claim -- that response control belongs in training rather than in a
post-hoc correction -- first appears.""",
        delta={"model.head": "gap", "model.fusion_pos": "learned",
               "model.num_pool_heads": None},
        caveats=_LADDER_CAVEAT + """

Two things change together here, because they cannot be separated: fresh noise
per step is a property of in-loop generation, and the response penalties are
only defined once the renderer is differentiable. Tier 3 then removes the
penalties one at a time from the fiducial model, which is where their individual
contributions are measured.""",
    ),
    Arm(
        path="research/ablations/tier2/08_learned_pooling_head",
        title="Tier 2.8: + learned pooling head (K = 4)",
        why="""Four learned spatial weightings replace the single fixed Gaussian
window, so the head sees four pooled descriptors instead of one. This relaxes a
bottleneck; whether it relaxes one that was binding is the measurement.""",
        delta={"model.fusion_pos": "learned"},
        caveats=_LADDER_CAVEAT + """

Worth reading beside the pooling diagnostic: four heads collapsing to ~1.4
effective would mean the extra capacity is not being used, and the honest
conclusion would be to report K = 1.""",
    ),
]


# ----------------------------------------------------------------------
# Tier 3 -- the response objectives, one at a time from the fiducial
# ----------------------------------------------------------------------
TIER3 = [
    Arm(
        path="research/ablations/tier3/no_gamma_response",
        title="Tier 3: the shear response penalty removed",
        why="""lambda_gamma is what ties the network's dR/dgamma to the analytic
per-object target. Removing it should be visible directly in the response
diagnostics rather than only in m, and it is the term whose absence the
metacalibration correction would have to make up for.""",
        delta={"train.response.gamma_weight": 0.0},
    ),
    Arm(
        path="research/ablations/tier3/no_shift_response",
        title="Tier 3: the translation response penalty removed",
        why="""lambda_shift constrains how much the shape estimate depends on
where the object falls in its stamp, over the +/-0.2 arcsec centroid dither.
This term is already known not to converge in the fiducial run -- it grows over
training rather than falling -- so the arm is as much a measurement of what the
weight is worth as of what its absence costs.""",
        delta={"train.response.shift_weight": 0.0},
    ),
    Arm(
        path="research/ablations/tier3/no_complement",
        title="Tier 3: the protected-subspace penalty removed",
        why="""lambda_perp damps directions orthogonal to every physical image
tangent, without touching the tangents themselves. It is the one response term
with no bias-variance trade against it, so if removing it costs nothing the
honest reporting is that it is unnecessary rather than that it is free.""",
        delta={"train.response.complement_weight": 0.0},
    ),
    Arm(
        path="research/ablations/tier3/no_psf_orbit",
        title="Tier 3: the PSF orbit penalty removed",
        why="""lambda_orbit re-renders the PSF alone at 90 degrees. It is a
simulator constraint and is distinct from the joint D4 symmetry the
architecture implements, which is exactly why it is separately ablatable: the
architecture cannot substitute for it.""",
        delta={"train.response.orbit_weight": 0.0},
    ),
    Arm(
        path="research/ablations/tier3/no_isotropy",
        title="Tier 3: the isotropy penalty removed",
        why="""lambda_iso penalises (D11 - D22)^2 + (D12 + D21)^2 on the
RESIDUAL D = R - target, and it is the one thing D4 equivariance structurally
cannot give you: every group element acts as diag(+/-1, +/-1), which zeroes the
off-diagonals and leaves R11 and R22 free. Measured on this model at
initialization, R12 and R21 sit at 1.8 and 1.0 sigma while R11 - R22 is at 35
sigma.

This arm also supplies the "objective disabled" column of the response
diagnostics table, so it is worth running even if the effect on m is small.""",
        delta={"train.response.isotropy_weight": 0.0},
    ),
    Arm(
        path="research/ablations/tier3/orbit_k2",
        title="Tier 3: PSF orbit K = 2 instead of K = 4",
        why="""K = 2 cannot reach the spin-4 term eps^2 conj(gamma); K = 4 can.
The memory argument for cutting it does not apply once the response terms run
on a 32-object sub-batch (measured saving: 0.5 of 114 MB per sample), but the
runtime cost is real -- three extra renders and forwards -- so the question is
whether the term it buys is worth them.""",
        delta={"train.response.orbit_k": 2},
    ),
    Arm(
        path="research/ablations/tier3/target_identity",
        title="Tier 3: identity response target instead of analytic",
        why="""The observed shape is a Moebius map of the applied shear, so its
exact derivative is 1 - eps^2, not 1. Averaged over an isotropic shape
distribution <eps^2> = 0 and the identity is defensible as an ENSEMBLE target --
but its per-object RMS is 0.31, so as a per-object target it fights the correct
answer on every galaxy. This arm measures what that costs.""",
        delta={"train.response.gamma_target": "identity"},
    ),
]


# ----------------------------------------------------------------------
# Tier 4 -- backbone, training procedure and loss
# ----------------------------------------------------------------------
TIER4 = [
    Arm(
        path="research/ablations/tier4/no_multiscale_block",
        title="Tier 4: the dilated context block removed",
        why="""The multi-scale block ends the galaxy branch and widens the
receptive field to the faint outer isophotes without a proportional parameter
cost. Removing it takes the model from 0.5811 M to 0.5194 M parameters, so this
is also the cheapest capacity reduction available if it turns out not to
matter.""",
        delta={"model.d4_multiscale": False},
    ),
    Arm(
        path="research/ablations/tier4/untrimmed_stem",
        title="Tier 4: untrimmed stem width (32, 48, 64)",
        why="""The fiducial halves the first stage from 32 to 16 channels. That
stage runs at the full 53x53 resolution and the orbit multiplies it by eight, so
it is essentially the whole memory bill -- but it is also where the finest
spatial detail lives. This arm pays the memory back to find out whether the trim
cost any accuracy.""",
        delta={"model.d4_features": [32, 48, 64]},
    ),
    Arm(
        path="research/ablations/tier4/full_resolution_psf_block",
        title="Tier 4: full-resolution PSF residual block restored",
        why="""The fiducial gives the PSF branch no residual block at 53x53, on
the argument that a smooth, known, low-frequency profile does not need residual
refinement at full resolution to be described. This arm restores it and tests
that argument.""",
        delta={"model.d4_depths_psf": [1, 1, 1]},
    ),
    Arm(
        path="research/ablations/tier4/no_label_normalization",
        title="Tier 4: raw labels, no z-scoring",
        why="""The outputs differ by orders of magnitude -- (g1, g2) ~ 0.1
against flux ~ 1e4 -- so without normalization the gradient is dominated by
whichever target happens to be largest. This arm measures whether that actually
harms the shear, which is the only output that matters.""",
        delta={"train.normalize_labels": False},
    ),
    Arm(
        path="research/ablations/tier4/no_ema",
        title="Tier 4: no exponential moving average of the weights",
        why="""The fiducial checkpoints a running average of the weights rather
than the final iterate. This arm removes it. Note the direction: the paper's
table lists EMA as an addition to a baseline, but the fiducial has it ON, so
this row is a removal and the table's sign needs to match.""",
        delta={"train.ema_decay": None},
    ),
    Arm(
        path="research/ablations/tier4/no_image_standardization",
        title="Tier 4: no per-channel input standardization",
        why="""Galaxy and PSF pixels are placed on a common scale before the
first convolution. As with EMA, the fiducial has this ON and the paper's table
lists it as an addition, so this row is a removal.""",
        delta={"image.normalize_images": False},
    ),
    Arm(
        path="research/ablations/tier4/loss_mae",
        title="Tier 4: mean absolute error",
        why="""MAE downweights large residuals relative to MSE, which could in
principle be more stable under the heavy-tailed errors of faint galaxies. It is
scored through m1 and alpha rather than through the training objective, so no
loss is favored by construction.""",
        delta={"train.loss": "mae"},
    ),
    Arm(
        path="research/ablations/tier4/loss_huber",
        title="Tier 4: Huber loss (delta = 1)",
        why="""Quadratic near zero and linear in the tail, so it sits between
MSE and MAE. Same scoring as the MAE arm: judged on the science metrics, not on
the value of its own objective.""",
        delta={"train.loss": "huber"},
    ),
]

ALL_ARMS: List[Arm] = LADDER + TIER1 + TIER2 + TIER3 + TIER4

#: Documented as a table row but not generatable. `dropout` reaches only a
#: `research_backed` branch (models.py: the D4 model forwards it, and the branch
#: ignores it otherwise), so on the fiducial `shearnet-d4` backbone a spatial
#: dropout arm would train an identical network and report a spurious null.
BLOCKED = {
    "tier4/spatial_dropout": (
        "model.dropout reaches only a research_backed branch. On the fiducial "
        "shearnet-d4 backbone it is silently ignored, so this arm would be a "
        "no-op reported as a measurement. Needs dropout plumbed into "
        "_ShearNetD4Backbone before it can be run."
    ),
}


# ----------------------------------------------------------------------
def _set(tree: Dict, dotted: str, value: Any) -> None:
    """Set (or, for ``None`` on a container key, delete) a dotted path."""
    keys = dotted.split(".")
    node = tree
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise TypeError(f"{dotted}: {key} is not a mapping")
    if value is None:
        node.pop(keys[-1], None)
    else:
        node[keys[-1]] = value


def _header(arm: Arm) -> str:
    """The comment block that opens a generated config."""
    lines = [
        "# " + "-" * 70,
        f"# {arm.title}",
        "# " + "-" * 70,
        "#",
        "# GENERATED by research/ablations/generate_configs.py -- do not hand-edit.",
        "# Edit the fiducial config or the arm's entry in that script and re-run.",
        "#",
        "# Fiducial:",
        f"#   {FIDUCIAL.relative_to(REPO)}",
        "#",
        "# Changed from the fiducial:",
    ]
    if arm.delta:
        width = max(len(k) for k in arm.delta)
        for key, value in arm.delta.items():
            shown = "<removed>" if value is None else repr(value)
            lines.append(f"#   {key:<{width}}  ->  {shown}")
    else:
        lines.append("#   nothing (this IS the fiducial configuration)")
    lines += ["#", "# Why:"]
    lines += [f"# {line}".rstrip() for line in arm.why.strip().splitlines()]
    if arm.caveats:
        lines += ["#", "# Read this before trusting the result:"]
        lines += [f"# {line}".rstrip() for line in arm.caveats.strip().splitlines()]
    lines += ["# " + "-" * 70, ""]
    return "\n".join(lines)


def build(arm: Arm, fiducial: Dict) -> str:
    """Render one arm's config.yaml as text."""
    config = copy.deepcopy(fiducial)
    name = Path(arm.path).name
    if arm.path.startswith("research/unit_tests/"):
        name = f"d4_unit_{name}"
    _set(config, "meta.model_name", name)
    _set(config, "meta.description", arm.title)
    _set(config, "paths.root", f"{CLUSTER_ROOT}/{arm.path}")
    # Inert since the toggle was removed; carrying it forward would only invite
    # the warning on every run.
    _set(config, "model.process_psf", None)
    for key, value in arm.delta.items():
        _set(config, key, value)
    body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False, width=88)
    return _header(arm) + body


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the files on disk match; write nothing")
    args = parser.parse_args(argv)

    fiducial = yaml.safe_load(FIDUCIAL.read_text())
    stale: List[str] = []
    for arm in ALL_ARMS:
        text = build(arm, fiducial)
        path = REPO / arm.path / "config.yaml"
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(arm.path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {arm.path}/config.yaml")

    if args.check:
        if stale:
            print("stale or missing:", *stale, sep="\n  ")
            return 1
        print(f"all {len(ALL_ARMS)} configs match")
        return 0

    print(f"\n{len(ALL_ARMS)} configs written.")
    if BLOCKED:
        print("\nDocumented but NOT generated, because they would be no-ops:")
        for name, reason in BLOCKED.items():
            print(f"  {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
