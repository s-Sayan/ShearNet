"""The one configuration schema ShearNet reads.

Every setting either program understands is listed in :data:`FIELDS`, with its
type, default and meaning. A YAML key that is not listed is an error, not a
silently ignored line. This prevents misspelled settings from silently
selecting defaults.

The layout follows the SuperBIT configs where there is an obvious equivalent
(``run_options`` for identity and execution, scientific ``snake_case`` names
with units in the docs), and is otherwise ShearNet's own:

``run_options``  who the run is and where it lives
``simulation``   how a stamp is rendered -- shared by training and evaluation
``model``        the architecture
``training``     the training population, objective and schedule
``evaluation``   what ``shearnet-eval`` measures: scenes, ring, metacal

Nothing here renders, trains, or creates a directory. :func:`resolve` takes a
plain mapping and returns a fully populated, validated nested dict.
"""

from __future__ import annotations

import copy
import difflib
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """A configuration that cannot be run as written."""


@dataclass(frozen=True)
class Field:
    """One setting.

    ``kind`` is one of ``int``, ``float``, ``bool``, ``str``, ``path``,
    ``list[int]``, ``list[float]``, ``list[str]`` or ``scenes``. ``path`` is a
    string resolved against the directory of the YAML that set it.
    """

    kind: str
    default: Any
    doc: str
    nullable: bool = False
    choices: Optional[Tuple[Any, ...]] = None
    check: Optional[Callable[[Any], Optional[str]]] = None
    unit: str = ""


def _positive(value):
    return None if value > 0 else "must be > 0"


def _non_negative(value):
    return None if value >= 0 else "must be >= 0"


def _fraction(value):
    return None if 0.0 < value < 1.0 else "must be strictly between 0 and 1"


def _shear_range(value):
    return None if 0.0 <= value < 1.0 else "must satisfy 0 <= value < 1"


def _all_positive(values):
    return None if all(v > 0 for v in values) else "every entry must be > 0"


def _decay(value):
    return None if 0.0 < value < 1.0 else "must be strictly between 0 and 1"


#: The five scenes an evaluation measures unless told otherwise: no applied
#: shear (the PSF-leakage population), and a +/- pair on each component.
DEFAULT_SCENES = (
    {"name": "zero", "g1": 0.0, "g2": 0.0},
    {"name": "g1_plus", "g1": 0.01, "g2": 0.0},
    {"name": "g1_minus", "g1": -0.01, "g2": 0.0},
    {"name": "g2_plus", "g1": 0.0, "g2": 0.01},
    {"name": "g2_minus", "g1": 0.0, "g2": -0.01},
)

ESTIMATORS = ("shearnet", "ngmix")

FIELDS: Dict[str, Field] = {
    # -- identity and execution --------------------------------------------
    "run_options.run_name": Field(
        "str", None, "Name of the run. Prefixes the evaluation catalog filename."),
    "run_options.outdir": Field(
        "path", None, "Run directory: model, normalizers, history, logs and every "
        "evaluation of this model live under it. `--run` on the command line "
        "overrides it.", nullable=True),
    "run_options.description": Field(
        "str", None, "Free text, carried into the run manifest.", nullable=True),
    "run_options.ncores": Field(
        "int", None, "Worker processes for GalSim rendering and the ngmix fits. "
        "null uses SLURM_CPUS_PER_TASK on a cluster and 1 elsewhere.",
        nullable=True, check=_positive),

    # -- rendering, shared by training and evaluation ----------------------
    "simulation.backend": Field(
        "str", "galsim", "Renderer. jax-galsim is differentiable and batched; "
        "in-loop training and the evaluation need it.",
        choices=("galsim", "jax-galsim")),
    "simulation.pixel_scale": Field(
        "float", 0.141, "Pixel scale.", check=_positive, unit="arcsec / pixel"),
    "simulation.stamp_size": Field(
        "int", 53, "Side of the square postage stamp.", check=_positive, unit="pixel"),
    "simulation.noise_sigma": Field(
        "float", 1.0e-5, "Standard deviation of the Gaussian pixel noise, in image "
        "counts. Not shape noise and not S/N.", check=_non_negative, unit="count"),
    "simulation.gal_model": Field(
        "str", "exp", "Galaxy light profile.", choices=("exp", "gauss")),
    "simulation.hlr_type": Field(
        "str", "constant", "constant: every galaxy has hlr 0.5 arcsec. catalog: the "
        "HLR column of the catalog.", choices=("constant", "catalog")),
    "simulation.flux_type": Field(
        "str", "constant", "constant: every galaxy has flux 12258.97. catalog: the "
        "FLUX column of the catalog.", choices=("constant", "catalog")),
    "simulation.apply_psf_shear": Field(
        "bool", False, "Draw a random per-object shear onto the PSF (an artificial "
        "transform of the PSF model, not a measured PSF ellipticity)."),
    "simulation.psf_shear_range": Field(
        "float", 0.05, "Half-width of the uniform PSF-shear draw when "
        "apply_psf_shear is on.", check=_shear_range),
    "simulation.jax_fft_size": Field(
        "int", 256, "Pinned FFT grid of the jax-galsim renderer.", check=_positive,
        unit="pixel"),
    "simulation.jax_batch_size": Field(
        "int", 256, "Objects per batched jax-galsim render. Memory scales as "
        "jax_batch_size * jax_fft_size**2.", check=_positive),
    "simulation.psf.mode": Field(
        "str", "ideal", "ideal: a round Gaussian of gaussian_fwhm. superbit: the "
        "PSFEx models under psfex_file.", choices=("ideal", "superbit")),
    "simulation.psf.gaussian_fwhm": Field(
        "float", 0.5, "FWHM of the ideal Gaussian PSF.", check=_positive,
        unit="arcsec"),
    "simulation.psf.psfex_file": Field(
        "path", None, "A PSFEx .psf file, or a directory of them, for mode superbit. "
        "null uses the library bundled with the repository.", nullable=True),
    "simulation.catalogs.train_file": Field(
        "path", None, "Catalog FITS with G1, G2, HLR and FLUX columns that the "
        "training galaxies are drawn from. null draws a synthetic population, "
        "which is only ever useful for tests.", nullable=True),
    "simulation.catalogs.eval_file": Field(
        "path", None, "Held-out catalog the evaluation galaxies are drawn from. "
        "Required whenever train_file is set: object i is row i whatever the seed, "
        "so a shared catalog would re-measure the training galaxies.",
        nullable=True),

    # -- architecture ------------------------------------------------------
    "model.type": Field(
        "str", "cnn", "Architecture: mlp, cnn, resnet, research_backed, "
        "forklens_psfnet (single stamp), fork-like or d4-fork-like (galaxy + PSF).",
        choices=("mlp", "cnn", "resnet", "research_backed", "forklens_psfnet",
                 "fork-like", "d4-fork-like")),
    "model.galaxy_branch": Field(
        "str", "research_backed", "Galaxy branch of a two-branch model."),
    "model.psf_branch": Field(
        "str", "forklens_psf", "PSF branch of a two-branch model."),
    "model.output_keys": Field(
        "list[str]", ["g1", "g2"], "Network outputs, in order. g1/g2 are the "
        "pre-PSF ellipticity of the rendered galaxy (intrinsic shape composed with "
        "the applied shear); hlr and flux are the profile inputs."),
    "model.gap": Field("bool", False, "Global average pooling in the branches."),
    "model.fusion": Field(
        "str", "concat", "How the two branches are combined.",
        choices=("concat", "transformer")),
    "model.fusion_pos": Field(
        "str", "learned", "Positional encoding of the transformer fusion.",
        choices=("learned", "rope2d", "none")),
    "model.head": Field(
        "str", "gap", "Pooling head of d4-fork-like.", choices=("gap", "attention")),
    "model.dropout": Field(
        "float", 0.0, "Spatial dropout of a research_backed branch.",
        check=_non_negative),
    "model.branch_features": Field(
        "list[int]", None, "Channel widths of a d4cnn branch.", nullable=True,
        check=_all_positive),
    "model.d4_features": Field(
        "list[int]", None, "Stage widths of a shearnet-d4 branch.", nullable=True,
        check=_all_positive),
    "model.d4_depths_galaxy": Field(
        "list[int]", None, "Residual blocks per stage, galaxy shearnet-d4 branch.",
        nullable=True),
    "model.d4_depths_psf": Field(
        "list[int]", None, "Residual blocks per stage, PSF shearnet-d4 branch.",
        nullable=True),
    "model.d4_multiscale": Field(
        "bool", None, "Dilated context block at the end of the galaxy shearnet-d4 "
        "branch. null keeps the architecture default (on).", nullable=True),
    "model.orbit_scan": Field(
        "bool", True, "Run the D4 orbit as a rematerialised scan (less memory, "
        "same parameters)."),
    "model.design": Field(
        "str", None, "d4-fork-like fusion/head layout: d4cnn or shearnet-d4. null "
        "infers it from the galaxy branch.", nullable=True,
        choices=("d4cnn", "shearnet-d4")),
    "model.d_model": Field("int", None, "Fusion width.", nullable=True, check=_positive),
    "model.num_heads": Field(
        "int", None, "Fusion attention heads.", nullable=True, check=_positive),
    "model.num_pool_heads": Field(
        "int", None, "Attention pooling maps (head: attention).", nullable=True,
        check=_positive),
    "model.num_self_attn_layers": Field(
        "int", None, "Self-attention layers in the fusion block.", nullable=True,
        check=_non_negative),
    "model.ffn_dim": Field(
        "int", None, "Feed-forward width in the fusion block.", nullable=True,
        check=_non_negative),

    # -- training ----------------------------------------------------------
    "training.seed": Field(
        "int", 42, "Seed of the training population and of the optimiser."),
    "training.nobj": Field(
        "int", 10000, "Training objects (validation included).", check=_positive),
    "training.generation": Field(
        "str", "upfront", "upfront: render the dataset before training. inloop: "
        "render inside the jitted step (needs jax-galsim).",
        choices=("upfront", "inloop")),
    "training.base_shear_range": Field(
        "float", 0.0, "Half-width of a uniform per-object applied shear in the "
        "training population (jax-galsim only).", check=_shear_range),
    "training.epochs": Field("int", 10, "Maximum epochs.", check=_positive),
    "training.batch_size": Field("int", 32, "Batch size.", check=_positive),
    "training.learning_rate": Field(
        "float", 1.0e-3, "Peak learning rate.", check=_positive),
    "training.weight_decay": Field(
        "float", 1.0e-4, "AdamW weight decay.", check=_non_negative),
    "training.patience": Field(
        "int", 10, "Early stopping: validations without improvement.",
        check=_positive),
    "training.val_split": Field(
        "float", 0.2, "Fraction of training.nobj held out for validation.",
        check=_fraction),
    "training.eval_interval": Field(
        "int", 1, "Validate every N epochs.", check=_positive),
    "training.loss": Field(
        "str", "mse", "Training loss, a registry name (mse, mae, huber, ...)."),
    "training.loss_weights": Field(
        "list[float]", None, "One weight per output key, in order. null weights "
        "them equally.", nullable=True),
    "training.ema_decay": Field(
        "float", None, "Exponential moving average of the weights; null is off. "
        "When on, validation and the saved model use the averaged weights.",
        nullable=True, check=_decay),
    "training.resample_noise": Field(
        "bool", False, "Upfront only: render noise-free and draw fresh noise every "
        "epoch."),
    "training.normalize_labels": Field(
        "bool", True, "Z-score the labels, fit on the training portion."),
    "training.normalize_images": Field(
        "bool", False, "Standardize the input stamps, fit on the training portion."),
    "training.d4_augment": Field(
        "bool", False, "8x D4 augmentation of the training portion. An ablation "
        "control for non-equivariant models; upfront only."),
    "training.response.gamma_weight": Field(
        "float", 0.0, "Drive R^gamma to its target.", check=_non_negative),
    "training.response.psf_weight": Field(
        "float", 0.0, "Drive the PSF-shear response to zero.", check=_non_negative),
    "training.response.shift_weight": Field(
        "float", 0.0, "Drive the translation response to zero.",
        check=_non_negative),
    "training.response.complement_weight": Field(
        "float", 0.0, "Penalise sensitivity outside the physical tangents.",
        check=_non_negative),
    "training.response.orbit_weight": Field(
        "float", 0.0, "PSF-orbit consistency under shared noise.",
        check=_non_negative),
    "training.response.isotropy_weight": Field(
        "float", 0.0, "Penalise (R11 - R22)^2 + (R12 + R21)^2 of the residual.",
        check=_non_negative),
    "training.response.every_n_steps": Field(
        "int", 1, "Evaluate the response terms every N steps.", check=_positive),
    "training.response.orbit_k": Field(
        "int", 2, "PSF orbit size.", choices=(2, 4)),
    "training.response.gamma_target": Field(
        "str", "analytic", "analytic: the exact per-object derivative of the label. "
        "identity: I, an ensemble target.", choices=("analytic", "identity")),
    "training.response.batch": Field(
        "int", 0, "Evaluate the response terms on the first N objects of each batch "
        "(0 = all).", check=_non_negative),
    "training.response.report": Field(
        "bool", None, "Log the measured response matrices at every validation. null "
        "is on whenever a response weight is nonzero.", nullable=True),
    "training.noise.min_sd": Field(
        "float", None, "In-loop depth augmentation: lower noise bound.",
        nullable=True, check=_non_negative),
    "training.noise.max_sd": Field(
        "float", None, "In-loop depth augmentation: upper noise bound.",
        nullable=True, check=_non_negative),
    "training.noise.condition": Field(
        "bool", False, "Express the inputs in units of the sampled noise."),

    # -- evaluation --------------------------------------------------------
    "evaluation.seed": Field(
        "int", 58, "Seed of the evaluation population. Must differ from "
        "training.seed."),
    "evaluation.nobj": Field(
        "int", 1000, "Objects per scene and ring station.", check=_positive),
    "evaluation.estimators": Field(
        "list[str]", ["shearnet", "ngmix"], "What is measured on every stamp."),
    "evaluation.batch_size": Field(
        "int", 4096, "Stamps per ShearNet forward pass.", check=_positive),
    "evaluation.scenes": Field(
        "scenes", [dict(s) for s in DEFAULT_SCENES], "Applied reduced shears to "
        "render, each {name, g1, g2}. Every scene is rendered from the same "
        "galaxies, offsets, PSFs and noise, so any two are pair-matched."),
    "evaluation.rotations_deg": Field(
        "list[float]", [0.0], "Ring stations: every scene is re-rendered with the "
        "intrinsic galaxy shape rotated by each of these angles. The PSF is not "
        "rotated. {0, 90} cancels the intrinsic shape; {0, 45, 90, 135} also the "
        "O(eps^2 g) term.", unit="deg"),
    "evaluation.metacal.psf": Field(
        "str", "dilate", "Metacal reconvolution PSF. Only dilate has the *_psf "
        "products.", choices=("dilate",)),
    "evaluation.metacal.step": Field(
        "float", 0.01, "One-sided metacal shear step (ngmix 'step', SuperBIT "
        "'mcal_shear'). The +/- products are 2*step apart.", check=_positive),
    "evaluation.metacal.shearnet": Field(
        "bool", True, "Also run ShearNet on the nine metacal images ngmix fits."),
    "evaluation.ngmix.gal_model": Field("str", "gauss", "ngmix galaxy model."),
    "evaluation.ngmix.psf_model": Field("str", "gauss", "ngmix PSF model."),
}

#: Settings an evaluation may change relative to the run it measures. Anything
#: else -- the architecture, the renderer, the training population -- is the
#: run's own and comes from its saved config.
EVALUATION_OVERRIDABLE = frozenset(
    {k for k in FIELDS if k.startswith("evaluation.")}
    | {"simulation.catalogs.eval_file", "run_options.ncores"}
)

#: The top-level blocks. Anything else at the top level is an error.
SECTIONS = ("run_options", "simulation", "model", "training", "evaluation")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def flatten(tree: Mapping, prefix: str = "") -> Dict[str, Any]:
    """``{"a": {"b": 1}}`` -> ``{"a.b": 1}``. Lists and scalars are leaves."""
    out = {}
    for key, value in tree.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and dotted not in FIELDS:
            # an empty block (`noise: {}`) sets nothing
            out.update(flatten(value, dotted + "."))
        else:
            out[dotted] = value
    return out


def unflatten(flat: Mapping[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`flatten`."""
    tree: Dict[str, Any] = {}
    for dotted, value in flat.items():
        node = tree
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def defaults() -> Dict[str, Any]:
    """The fully populated default config, without a run name."""
    flat = {k: copy.deepcopy(f.default) for k, f in FIELDS.items()}
    tree = unflatten(flat)
    return {"schema_version": SCHEMA_VERSION, **tree}


def _suggest(key: str) -> str:
    close = difflib.get_close_matches(key, FIELDS, n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce(key: str, field: Field, value):
    """Check ``value`` against ``field`` and return it in canonical form."""
    if value is None:
        if field.nullable:
            return None
        raise ConfigError(f"{key} may not be null")
    kind = field.kind
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            if _is_number(value) and float(value).is_integer():
                value = int(value)
            else:
                raise ConfigError(f"{key} must be an integer, got {value!r}")
    elif kind == "float":
        if not _is_number(value):
            raise ConfigError(f"{key} must be a number, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise ConfigError(f"{key} must be finite, got {value!r}")
    elif kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{key} must be true or false, got {value!r}")
    elif kind in ("str", "path"):
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string, got {value!r}")
        value = os.path.expanduser(os.path.expandvars(value)) if kind == "path" else value
    elif kind.startswith("list["):
        inner = kind[5:-1]
        if not isinstance(value, (list, tuple)) or not value:
            raise ConfigError(f"{key} must be a non-empty list, got {value!r}")
        item_field = Field(inner, None, "")
        value = [_coerce(f"{key}[{i}]", item_field, v) for i, v in enumerate(value)]
    elif kind == "scenes":
        value = _scenes(key, value)
    else:  # pragma: no cover - a schema bug
        raise AssertionError(f"unknown field kind {kind!r}")

    if field.choices is not None:
        items = value if isinstance(value, list) else [value]
        bad = [v for v in items if v not in field.choices]
        if bad:
            raise ConfigError(
                f"{key} must be one of {list(field.choices)}, got {bad[0]!r}")
    if field.check is not None:
        problem = field.check(value)
        if problem:
            raise ConfigError(f"{key} {problem}, got {value!r}")
    return value


def _scenes(key, value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{key} must be a non-empty list of {{name, g1, g2}}")
    out, names = [], set()
    for i, scene in enumerate(value):
        if not isinstance(scene, Mapping) or set(scene) != {"name", "g1", "g2"}:
            raise ConfigError(
                f"{key}[{i}] must have exactly the keys name, g1 and g2, got {scene!r}")
        name = scene["name"]
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{key}[{i}].name must be a non-empty string")
        if name in names:
            raise ConfigError(f"{key}: scene name {name!r} appears twice")
        names.add(name)
        g1 = _coerce(f"{key}[{i}].g1", Field("float", 0.0, ""), scene["g1"])
        g2 = _coerce(f"{key}[{i}].g2", Field("float", 0.0, ""), scene["g2"])
        if math.hypot(g1, g2) >= 1.0:
            raise ConfigError(f"{key}[{i}]: |g| = {math.hypot(g1, g2):.3f} must be < 1")
        out.append({"name": name, "g1": g1, "g2": g2})
    return out


# ----------------------------------------------------------------------
# resolution
# ----------------------------------------------------------------------
def check_keys(flat: Mapping[str, Any], allowed: Iterable[str] = None) -> None:
    """Reject anything not in the schema (or not in ``allowed``)."""
    allowed = FIELDS if allowed is None else set(allowed)
    unknown = sorted(k for k in flat if k not in allowed)
    if unknown:
        lines = [f"  {k}{_suggest(k)}" for k in unknown]
        raise ConfigError("unknown config keys:\n" + "\n".join(lines))


def resolve(user: Mapping[str, Any], base_dir: Optional[str] = None,
            base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Validate ``user`` and fill every unset field.

    ``user`` is a nested mapping in this schema (alternate config dialects are translated
    before they get here). ``base_dir`` resolves relative ``path`` fields: it is
    the directory of the file the values came from. ``base`` is an already
    resolved config to layer ``user`` on top of instead of the defaults.
    """
    user = dict(user)
    version = user.pop("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version {version!r} is not supported; this ShearNet reads "
            f"version {SCHEMA_VERSION}")
    stray = sorted(k for k in user if k not in SECTIONS)
    if stray:
        raise ConfigError(
            f"unknown top-level blocks {stray}; expected a subset of {list(SECTIONS)}")
    flat = flatten(user)
    check_keys(flat)

    resolved = flatten(base) if base is not None else flatten(defaults())
    resolved.pop("schema_version", None)
    for key, value in flat.items():
        value = _coerce(key, FIELDS[key], value)
        if FIELDS[key].kind == "path" and value is not None and base_dir is not None:
            value = os.path.normpath(os.path.join(base_dir, value))
        resolved[key] = value
    validate(resolved)
    return {"schema_version": SCHEMA_VERSION, **unflatten(resolved)}


# ----------------------------------------------------------------------
# cross-field rules
# ----------------------------------------------------------------------
#: Architectures that take a (galaxy, PSF) pair.
FORK_MODELS = ("fork-like", "d4-fork-like")


def _used_model_fields(flat) -> set:
    """The model.* settings the configured architecture actually reads."""
    nn = flat["model.type"]
    used = {"model.type", "model.output_keys", "model.gap"}
    if nn == "fork-like":
        used |= {"model.galaxy_branch", "model.psf_branch", "model.fusion"}
    elif nn == "d4-fork-like":
        branches = {flat["model.galaxy_branch"], flat["model.psf_branch"]}
        used |= {
            "model.galaxy_branch", "model.psf_branch", "model.fusion", "model.head",
            "model.orbit_scan", "model.fusion_pos", "model.design", "model.d_model",
            "model.num_heads", "model.num_self_attn_layers", "model.ffn_dim",
        }
        if flat["model.head"] == "attention":
            used.add("model.num_pool_heads")
        if "research_backed" in branches:
            used.add("model.dropout")
        # build_model maps the generic 'cnn' (and a missing branch) onto d4cnn
        if branches & {"d4cnn", "cnn"}:
            used.add("model.branch_features")
        if "shearnet-d4" in branches:
            used |= {"model.d4_features", "model.d4_depths_galaxy",
                     "model.d4_depths_psf", "model.d4_multiscale"}
    elif nn == "research_backed":
        used.add("model.dropout")
    return used


def validate(flat: Mapping[str, Any]) -> None:
    """Rules that span more than one field. Raises :class:`ConfigError`."""
    problems = []

    # an ignored setting is an ablation that silently ran the control
    used = _used_model_fields(flat)
    for key in sorted(k for k in FIELDS if k.startswith("model.")):
        if key not in used and flat[key] != FIELDS[key].default:
            problems.append(
                f"{key} = {flat[key]!r} has no effect on model.type "
                f"{flat['model.type']!r} (with these branches); remove it")
    if flat["model.type"] == "d4-fork-like" and flat["model.gap"]:
        problems.append("d4-fork-like refuses model.gap true; choose the pooling "
                        "with model.head")

    keys = flat["model.output_keys"]
    valid_keys = {"g1", "g2", "hlr", "flux", "psf_e1", "psf_e2", "psf_T"}
    if len(set(keys)) != len(keys):
        problems.append(f"model.output_keys repeats a key: {keys}")
    if set(keys) - valid_keys:
        problems.append(f"model.output_keys {sorted(set(keys) - valid_keys)} are not "
                        f"in {sorted(valid_keys)}")
    weights = flat["training.loss_weights"]
    if weights is not None and len(weights) != len(keys):
        problems.append(f"training.loss_weights has {len(weights)} entries for "
                        f"{len(keys)} output keys")

    inloop = flat["training.generation"] == "inloop"
    jax = flat["simulation.backend"] == "jax-galsim"
    if inloop and not jax:
        problems.append("training.generation inloop renders inside the jitted step "
                        "and needs simulation.backend jax-galsim")
    if inloop and flat["training.d4_augment"]:
        problems.append("training.d4_augment duplicates a materialised array and has "
                        "no meaning with generation inloop")
    if inloop and flat["training.resample_noise"]:
        problems.append("training.resample_noise is the upfront path's fresh noise; "
                        "inloop already draws fresh noise every step")
    if inloop and set(keys) & {"psf_e1", "psf_e2", "psf_T"}:
        problems.append("psf_* output keys need an ngmix fit per stamp and cannot be "
                        "rendered in-loop")
    if flat["training.base_shear_range"] and not jax:
        problems.append("training.base_shear_range is jax-galsim only")

    response = [k for k in FIELDS if k.startswith("training.response.") and
                k.endswith("_weight") and flat[k]]
    if response and not inloop:
        problems.append(f"{response} need the renderer in the autodiff graph: set "
                        "training.generation inloop")
    if response and not {"g1", "g2"} <= set(keys):
        problems.append("the response terms need g1 and g2 among model.output_keys")
    if (flat["training.response.orbit_weight"] and flat["simulation.psf.mode"] == "ideal"
            and not flat["simulation.apply_psf_shear"]):
        problems.append("training.response.orbit_weight rotates the PSF, which is the "
                        "identity on a round ideal PSF; set it to 0 for psf.mode ideal")

    low, high = flat["training.noise.min_sd"], flat["training.noise.max_sd"]
    if (low is None) != (high is None):
        problems.append("training.noise needs both min_sd and max_sd, or neither")
    elif low is not None and low > high:
        problems.append("training.noise.min_sd exceeds max_sd")
    if flat["training.noise.condition"] and low is None:
        problems.append("training.noise.condition needs training.noise.min_sd/max_sd")
    if flat["training.noise.condition"] and flat["training.normalize_images"]:
        problems.append("training.noise.condition and training.normalize_images are "
                        "two incompatible ways to set the input scale")
    if low is not None and not inloop:
        problems.append("training.noise ranges are drawn per in-loop batch; set "
                        "training.generation inloop")

    if flat["simulation.catalogs.train_file"] and not flat["simulation.catalogs.eval_file"]:
        problems.append("simulation.catalogs.train_file is set but eval_file is not: "
                        "row i is the same galaxy whatever the seed, so the evaluation "
                        "would re-measure the training galaxies")
    train_cat, eval_cat = (flat["simulation.catalogs.train_file"],
                           flat["simulation.catalogs.eval_file"])
    if train_cat and eval_cat and os.path.abspath(train_cat) == os.path.abspath(eval_cat):
        problems.append("simulation.catalogs.eval_file is the training catalog")
    if flat["evaluation.seed"] == flat["training.seed"]:
        problems.append("evaluation.seed equals training.seed; the evaluation must not "
                        "re-render the training noise and offsets")

    estimators = flat["evaluation.estimators"]
    if set(estimators) - set(ESTIMATORS):
        problems.append(f"evaluation.estimators {sorted(set(estimators) - set(ESTIMATORS))}"
                        f" are not in {list(ESTIMATORS)}")
    if len(set(estimators)) != len(estimators):
        problems.append(f"evaluation.estimators repeats an entry: {estimators}")
    if flat["evaluation.metacal.shearnet"] and "shearnet" not in estimators:
        problems.append("evaluation.metacal.shearnet needs shearnet in "
                        "evaluation.estimators")
    rotations = flat["evaluation.rotations_deg"]
    if len(set(rotations)) != len(rotations):
        problems.append(f"evaluation.rotations_deg repeats an angle: {rotations}")

    if problems:
        raise ConfigError("invalid configuration:\n" +
                          "\n".join(f"  - {p}" for p in problems))


def markdown() -> str:
    """The settings table of ``docs/config.md``, one section per block."""
    lines = []
    for section in SECTIONS:
        lines += [f"### `{section}`", "", "| key | type | default | meaning |",
                  "|---|---|---|---|"]
        for key, field in FIELDS.items():
            if not key.startswith(section + "."):
                continue
            kind = field.kind + (" or null" if field.nullable else "")
            if field.choices:
                kind += ": " + " / ".join(str(c) for c in field.choices)
            default = field.default
            if key == "evaluation.scenes":
                default = "zero, +/-0.01 on g1, +/-0.01 on g2"
            shown = "null" if default is None else f"`{default}`"
            doc = field.doc + (f" [{field.unit}]" if field.unit else "")
            lines.append(f"| `{key[len(section) + 1:]}` | {kind} | {shown} | {doc} |")
        lines.append("")
    return "\n".join(lines)
