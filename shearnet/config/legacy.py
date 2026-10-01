"""Translate the two pre-schema config dialects into the current schema.

Before :mod:`shearnet.config.schema` there were two YAML layouts:

* the package layout -- ``dataset`` / ``model`` / ``training`` / ``output`` /
  ``plotting`` / ``comparison`` / ``catalog`` -- deep-merged over
  ``default_config.yaml``;
* the unit-test layout -- ``meta`` / ``paths`` / ``image`` / ``psf`` /
  ``galaxy`` / ``model`` / ``train`` / ``eval`` -- copied onto the package keys
  through a fixed map, after the same merge.

:func:`migrate` reproduces that resolution exactly (the old defaults and the old
map are frozen below), then renames every value the old code actually read onto
its current key. Keys the old code never read are reported, not carried.

Two keys are deliberately NOT reproduced, because reproducing them would
reproduce a bug: the unit-test layout's ``train.loss`` and ``train.d4_augment``
were never in the old map, so every config that set them trained with MSE and
without augmentation. They now mean what they say, and :func:`migrate` notes it.

Use it on a file::

    python -m shearnet.config.legacy old.yaml > new.yaml
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Dict, List, Mapping, Tuple

from .schema import DEFAULT_SCENES, FIELDS, SCHEMA_VERSION, ConfigError, flatten, unflatten

#: ``shearnet/config/default_config.yaml`` as it was when the schema replaced it.
LEGACY_DEFAULTS: Dict[str, Any] = {
    "dataset": {
        "samples": 10000, "psf_fwhm": 0.5, "exp": "ideal", "nse_sd": 1.0e-5,
        "base_shear_range": 0.0, "seed": 42, "hlr_type": "constant",
        "flux_type": "constant", "stamp_size": 53, "pixel_size": 0.141,
        "apply_psf_shear": False, "psf_shear_range": 0.05, "normalize_labels": True,
        "normalize_images": False, "d4_augment": False, "nproc": None,
        "compute_metacal": False, "backend": "galsim", "jax_fft_size": 256,
        "jax_batch_size": 256, "generation": "upfront",
    },
    "model": {
        "type": "cnn", "galaxy": {"type": "research_backed"},
        "psf": {"type": "forklens_psf"}, "output_keys": ["g1", "g2"], "gap": False,
        "fusion": "concat", "fusion_pos": "learned",
    },
    "training": {
        "epochs": 10, "batch_size": 32, "learning_rate": 1.0e-3, "weight_decay": 1.0e-4,
        "patience": 10, "val_split": 0.2, "eval_interval": 1, "loss": "mse",
        "ema_decay": None,
        "response": {
            "gamma_weight": 0.0, "psf_weight": 0.0, "shift_weight": 0.0,
            "complement_weight": 0.0, "orbit_weight": 0.0, "every_n_steps": 1,
            "orbit_k": 2, "gamma_target": "analytic", "batch": 0, "report": None,
        },
        "noise": {"min_sd": None, "max_sd": None, "condition": False},
    },
    "evaluation": {"test_samples": 1000, "seed": 58},
    "output": {"save_path": None, "plot_path": None, "model_name": "my_model"},
    "plotting": {"plot": True},
    "comparison": {"mcal": True, "ngmix": True, "psf_model": "gauss", "gal_model": "gauss"},
    "catalog": {"cosmos_cat_fname": None},
}

#: The unit-test layout's map onto the package keys, as it was.
LEGACY_UNIT_TEST_MAP = {
    "train.samples": "dataset.samples",
    "train.seed": "dataset.seed",
    "image.noise_sd": "dataset.nse_sd",
    "image.stamp_size": "dataset.stamp_size",
    "image.pixel_scale": "dataset.pixel_size",
    "psf.gaussian_fwhm": "dataset.psf_fwhm",
    "psf.mode": "dataset.exp",
    "galaxy.hlr_type": "dataset.hlr_type",
    "galaxy.flux_type": "dataset.flux_type",
    "paths.psfex_model_file": "dataset.psfex_model_file",
    "model.architecture": "model.type",
    "model.galaxy_branch": "model.galaxy.type",
    "model.psf_branch": "model.psf.type",
    "train.epochs": "training.epochs",
    "train.batch_size": "training.batch_size",
    "train.learning_rate": "training.learning_rate",
    "train.weight_decay": "training.weight_decay",
    "train.patience": "training.patience",
    "train.val_split": "training.val_split",
    "train.eval_interval": "training.eval_interval",
    "train.loss_weights": "training.loss_weights",
    "train.ema_decay": "training.ema_decay",
    "train.resample_noise": "training.resample_noise",
    "meta.model_name": "output.model_name",
    "train.plot": "plotting.plot",
    "paths.train_catalog": "catalog.cosmos_cat_fname",
    "image.normalize_images": "dataset.normalize_images",
    "train.normalize_labels": "dataset.normalize_labels",
    "train.nproc": "dataset.nproc",
    "train.compute_metacal": "dataset.compute_metacal",
    "train.backend": "dataset.backend",
    "train.generation": "dataset.generation",
    "image.backend": "dataset.backend",
    "train.jax_fft_size": "dataset.jax_fft_size",
    "train.jax_batch_size": "dataset.jax_batch_size",
    "train.base_shear_range": "dataset.base_shear_range",
    "train.apply_psf_shear": "dataset.apply_psf_shear",
    "train.psf_shear_range": "dataset.psf_shear_range",
    "train.response": "training.response",
    "train.noise": "training.noise",
}

#: Resolved package key -> current key, for everything the old code read.
_RENAMES = {
    "dataset.samples": "training.nobj",
    "dataset.psf_fwhm": "simulation.psf.gaussian_fwhm",
    "dataset.exp": "simulation.psf.mode",
    "dataset.nse_sd": "simulation.noise_sigma",
    "dataset.base_shear_range": "training.base_shear_range",
    "dataset.seed": "training.seed",
    "dataset.hlr_type": "simulation.hlr_type",
    "dataset.flux_type": "simulation.flux_type",
    "dataset.stamp_size": "simulation.stamp_size",
    "dataset.pixel_size": "simulation.pixel_scale",
    "dataset.apply_psf_shear": "simulation.apply_psf_shear",
    "dataset.psf_shear_range": "simulation.psf_shear_range",
    "dataset.normalize_labels": "training.normalize_labels",
    "dataset.normalize_images": "training.normalize_images",
    "dataset.d4_augment": "training.d4_augment",
    "dataset.nproc": "run_options.ncores",
    "dataset.backend": "simulation.backend",
    "dataset.jax_fft_size": "simulation.jax_fft_size",
    "dataset.jax_batch_size": "simulation.jax_batch_size",
    "dataset.generation": "training.generation",
    "dataset.psfex_model_file": "simulation.psf.psfex_file",
    "dataset.type": "simulation.gal_model",
    "model.galaxy.type": "model.galaxy_branch",
    "model.psf.type": "model.psf_branch",
    "evaluation.test_samples": "evaluation.nobj",
    "evaluation.seed": "evaluation.seed",
    "output.model_name": "run_options.run_name",
    "comparison.psf_model": "evaluation.ngmix.psf_model",
    "comparison.gal_model": "evaluation.ngmix.gal_model",
    "catalog.cosmos_cat_fname": "simulation.catalogs.train_file",
}
for _key in FIELDS:
    if _key.startswith(("model.", "training.")) and _key not in _RENAMES.values():
        _RENAMES.setdefault(_key, _key)

#: Read by nothing, or by code that no longer exists. Reported and dropped.
_DROPPED = {
    "output.save_path", "output.plot_path", "comparison.mcal", "comparison.ngmix",
    "model.process_psf", "dataset.normalized", "paths.checkpoint_dir",
    "paths.plot_dir", "psf.noise", "psf.stamp_size", "galaxy.hlr", "galaxy.flux",
    "eval.include_shearnet",
}
_DROPPED_PREFIXES = ("plotting.", "provenance.", "eval.leakage.", "eval.timing.")
#: Analysis settings of the old evaluator. Cuts, responses, jackknives and the
#: catalog level now belong to whatever reads the evaluation FITS.
_EVAL_DROPPED = {
    "n_jackknife", "c_convention", "resample", "psf_response_apply", "catalog_level",
    "output", "anacal_epochs", "anacal_sigma_arcsec", "anacal_stamp_size",
    "estimators", "fpfs_cross_check", "fpfs_sigma_shapelets", "metacal",
    "psf_response", "psf_response_direct_ngmix", "psf_response_direct_shearnet",
    "psf_response_direct_step", "psf_response_shearnet", "psf_response_step",
    "reconv_psf",
}
#: The unit-test blocks the old map consumed (anything else in them is checked).
_UNIT_TEST_BLOCKS = ("meta", "paths", "image", "psf", "galaxy", "train", "eval")
#: The ring an old ``shape_noise_cancel`` asked for.
_RINGS = {1: [0.0], 2: [0.0, 90.0], 4: [0.0, 45.0, 90.0, 135.0]}


def is_legacy(raw: Mapping[str, Any]) -> bool:
    """True for a mapping in either old dialect.

    ``model``, ``training`` and ``evaluation`` exist in both the old and the
    current layout, so only a block that exists nowhere else marks a file as
    old; anything else is read as the current schema (and an old key in a
    shared block then fails as an unknown key, with a suggestion).
    """
    if "schema_version" in raw:
        return False
    only_old = {"dataset", "output", "plotting", "comparison", "catalog", "provenance",
                *_UNIT_TEST_BLOCKS}
    return bool(set(raw) & only_old)


def _get(tree, dotted, default=None):
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def _set(tree, dotted, value):
    node = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _merge(base, update):
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _old_resolution(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """What the old ``Config`` held after loading ``raw``."""
    user = copy.deepcopy(dict(raw))
    dataset = user.get("dataset")
    if isinstance(dataset, dict) and "psf_sigma" in dataset:
        dataset.setdefault("psf_fwhm", dataset.pop("psf_sigma"))
    old = copy.deepcopy(LEGACY_DEFAULTS)
    _merge(old, user)
    if old.get("meta") is not None or old.get("train") is not None:
        for src, dst in LEGACY_UNIT_TEST_MAP.items():
            value = _get(old, src)
            if value is not None:
                _set(old, dst, copy.deepcopy(value))
    return old


def _scenes(shear: float, component) -> List[dict]:
    """The old harness's populations: the unsheared LEAKAGE one plus each pair."""
    if isinstance(component, str):
        text = component.strip().lower()
        comps = [0, 1] if text in ("both", "all", "01", "0,1") else [int(c) for c in text.split(",")]
    elif isinstance(component, (list, tuple)):
        comps = [int(c) for c in component]
    else:
        comps = [int(component)]
    scenes = [{"name": "zero", "g1": 0.0, "g2": 0.0}]
    for k in dict.fromkeys(comps):
        axis = ("g1", "g2")[k]
        scenes.append({"name": f"{axis}_plus", "g1": 0.0, "g2": 0.0, axis: float(shear)})
        scenes.append({"name": f"{axis}_minus", "g1": 0.0, "g2": 0.0, axis: -float(shear)})
    return [dict(s, g1=float(s["g1"]), g2=float(s["g2"])) for s in scenes]


def _rotations(value) -> List[float]:
    if isinstance(value, bool) or value is None:
        return list(_RINGS[2] if value else _RINGS[1])
    if int(value) not in _RINGS:
        raise ConfigError(f"eval shape_noise_cancel {value!r} is not one of {sorted(_RINGS)}")
    return list(_RINGS[int(value)])


def migrate(raw: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """``(config, notes)``: ``raw`` in the current schema, and what changed.

    Only settings that differ from the current defaults are written, so the
    result reads as the experiment rather than as the whole schema.
    """
    if not is_legacy(raw):
        raise ConfigError("not a legacy config (it has schema_version, or no old blocks)")
    notes: List[str] = []
    old = _old_resolution(raw)
    unit_test = old.get("meta") is not None or old.get("train") is not None
    out: Dict[str, Any] = {}

    missing = object()
    for src, dst in _RENAMES.items():
        value = _get(old, src, missing)
        if value is not missing:
            out[dst] = copy.deepcopy(value)
    if _get(old, "dataset.compute_metacal"):
        raise ConfigError("dataset.compute_metacal is no longer supported; nothing ever "
                          "read the images it stored")

    # the unit-test layout's own keys the map never reached
    if unit_test:
        for src, dst in (("meta.description", "run_options.description"),
                         ("paths.root", "run_options.outdir"),
                         ("paths.eval_catalog", "simulation.catalogs.eval_file")):
            if _get(raw, src) is not None:
                out[dst] = _get(raw, src)
        for src, dst in (("train.loss", "training.loss"),
                         ("train.d4_augment", "training.d4_augment")):
            if _get(raw, src) is not None:
                out[dst] = _get(raw, src)
                notes.append(f"{src} = {_get(raw, src)!r} is now honoured (it was never "
                             "read: the run it configured trained the default)")

    # the evaluation block of the old research harness. A config without one
    # was never measured by it, so it gets the current defaults instead.
    eval_block = raw.get("eval") if isinstance(raw.get("eval"), Mapping) else {}
    if eval_block:
        section = eval_block.get("evaluate") or eval_block.get("bias") or {}
        if eval_block.get("seed") is not None:
            out["evaluation.seed"] = eval_block["seed"]
        if eval_block.get("n_obs") is not None:
            out["evaluation.nobj"] = eval_block["n_obs"]
        if eval_block.get("gal_model") is not None:
            out["evaluation.ngmix.gal_model"] = eval_block["gal_model"]
        baseline = section.get("baseline", "ngmix")
        if baseline in ("anacal", "both"):
            notes.append(f"eval baseline {baseline!r}: the AnaCal estimator is gone; "
                         "measuring shearnet and ngmix")
        out["evaluation.estimators"] = ["shearnet", "ngmix"]
        step = section.get("response_step", section.get("metacal_step"))
        if step is not None:
            out["evaluation.metacal.step"] = step
        out["evaluation.scenes"] = _scenes(section.get("shear_true", 0.01),
                                           section.get("component", 0))
        out["evaluation.rotations_deg"] = _rotations(
            section.get("shape_noise_cancel", False))
        if section.get("psf_model") is not None:
            out["evaluation.ngmix.psf_model"] = section["psf_model"]
        out["evaluation.metacal.shearnet"] = bool(section.get("shearnet_metacal", False))
        if section.get("shearnet_batch_size") is not None:
            out["evaluation.batch_size"] = section["shearnet_batch_size"]
        if section.get("ngmix_nproc") is not None and out.get("run_options.ncores") is None:
            out["run_options.ncores"] = section["ngmix_nproc"]

    # account for every key the input had
    consumed = set(LEGACY_UNIT_TEST_MAP) | set(_RENAMES) | {
        "dataset.psf_sigma", "meta.description", "paths.root", "paths.eval_catalog",
        "train.loss", "train.d4_augment", "eval.seed", "eval.n_obs", "eval.gal_model",
        "evaluation.test_samples", "evaluation.seed", "dataset.compute_metacal",
    }
    for key in flatten_any(raw):
        if key in consumed or any(key.startswith(c + ".") for c in consumed):
            continue
        if key.startswith(("eval.evaluate.", "eval.bias.")):
            name = key.split(".", 2)[2]
            if name in _EVAL_DROPPED:
                notes.append(f"{key} dropped: analysis settings belong to whatever reads "
                             "the evaluation FITS")
                continue
            if name in ("baseline", "response_step", "metacal_step", "shear_true",
                        "component", "shape_noise_cancel", "psf_model",
                        "shearnet_metacal", "shearnet_batch_size", "ngmix_nproc"):
                continue
        if key in _DROPPED or key.startswith(_DROPPED_PREFIXES):
            notes.append(f"{key} dropped: nothing read it")
            continue
        raise ConfigError(f"legacy key {key!r} has no translation")

    # write only what differs from the current defaults
    from .schema import defaults as current_defaults

    base = flatten(current_defaults())
    minimal = {k: v for k, v in out.items() if k not in base or base[k] != v}
    if minimal.get("evaluation.scenes") == [dict(s) for s in DEFAULT_SCENES]:
        minimal.pop("evaluation.scenes")
    tree = unflatten(minimal)
    ordered = {"schema_version": SCHEMA_VERSION}
    for block in ("run_options", "simulation", "model", "training", "evaluation"):
        if block in tree:
            ordered[block] = tree[block]
    return ordered, notes


def flatten_any(tree: Mapping, prefix: str = "") -> List[str]:
    """Dotted leaf keys of an arbitrary mapping."""
    keys = []
    for key, value in tree.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            keys.extend(flatten_any(value, dotted + "."))
        else:
            keys.append(dotted)
    return keys


def main(argv=None) -> int:
    """``python -m shearnet.config.legacy OLD.yaml`` prints the translation."""
    from .loader import dump_yaml, read_yaml

    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: python -m shearnet.config.legacy OLD.yaml", file=sys.stderr)
        return 2
    config, notes = migrate(read_yaml(argv[0]))
    for note in notes:
        print(f"# {note}", file=sys.stderr)
    sys.stdout.write(dump_yaml(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
