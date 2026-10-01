"""Every column of the evaluation catalog: name, type, shape, unit, meaning.

One row of every per-record table is one catalog object, in one scene, at one
ring station. All of them share the key columns and the row order
(scene-major, then station, then catalog row), so ``TRUTH``, ``STAMP``,
``SHEARNET`` and ``NGMIX`` line up row for row and also join on ``record_id``.

Ellipticity convention: every ``g``/``e`` here is the reduced-shear-style
ellipticity epsilon = (1 - q) / (1 + q) exp(2 i phi) -- ngmix's ``g``, GalSim's
``Shear.g1/g2`` -- in image axes. None of them is a per-object shear; ``g`` is
what an estimator reports, ``e_*`` are the simulation's inputs.

Variant suffixes follow SuperBIT's metacal tables (``g_noshear``, ``g_1p``,
``T_1p``, ``Tpsf_1p``, ``s2n_1p``, ...): ``_original`` is the stamp as rendered,
the nine metacal suffixes are ngmix's deconvolve/shear/reconvolve products with
``psf: dilate``. The ``*_psf`` products shear the dilated reconvolution PSF,
not the original PSF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

SCHEMA_NAME = "shearnet-eval"
SCHEMA_VERSION = 1

#: Metacal products, as in :mod:`shearnet.evaluation.measurements`.
METACAL_TYPES = (
    "noshear",
    "1p", "1m", "2p", "2m",
    "1p_psf", "1m_psf", "2p_psf", "2m_psf",
)


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    shape: Tuple[int, ...] = ()
    unit: str = ""
    description: str = ""


KEY_COLUMNS = [
    Column("record_id", "i8", description="unique row id: (scene_id * n_rotations + "
           "rotation_id) * n_objects + catalog_row"),
    Column("catalog_row", "i8", description="row of the evaluation catalog this object is "
           "drawn from (0-based); the same galaxy in every scene and station"),
    Column("scene_id", "i2", description="index into the SCENES table"),
    Column("rotation_id", "i2", description="index into the ROTATIONS table"),
]

TRUTH_COLUMNS = KEY_COLUMNS + [
    Column("rotation_deg", "f8", unit="deg",
           description="active rotation of the source shape and offset at this station"),
    Column("e_source", "f8", (2,), description="source-model ellipticity after the "
           "station's rotation, before the applied shear (the catalog G1/G2 rotated)"),
    Column("g_applied", "f8", (2,), description="applied reduced shear of the scene "
           "(GalSim .shear(); area preserving, no magnification)"),
    Column("e_prepsf", "f8", (2,), description="ellipticity of the pre-PSF galaxy: "
           "e_source composed with g_applied. The g1/g2 the network is trained on"),
    Column("hlr", "f8", unit="arcsec", description="half_light_radius of the circular "
           "profile before shaping (circularized; not a measured, PSF-convolved size)"),
    Column("flux_model", "f8", unit="count", description="total flux of the profile"),
    Column("offset", "f8", (2,), unit="arcsec", description="sub-pixel offset (dx, dy) of "
           "the galaxy from the stamp centre"),
    Column("psf_shear", "f8", (2,), description="artificial reduced shear applied to the "
           "PSF model (simulation.apply_psf_shear); zero otherwise"),
    Column("psf_pos", "f8", (2,), unit="pixel", description="PSFEx focal-plane position "
           "(x, y) the PSF was evaluated at; zero for an ideal PSF"),
    Column("psf_file_id", "i4", description="index into the PSF_FILES table; -1 for an "
           "ideal Gaussian PSF"),
    Column("q_source", "f8", description="catalog axis ratio b/a (Q column), unrotated; "
           "NaN if the catalog has none"),
    Column("phi_source", "f8", unit="rad", description="catalog position angle (PHI "
           "column), unrotated; NaN if the catalog has none"),
]

STAMP_COLUMNS = KEY_COLUMNS + [
    Column("psf_g", "f8", (2,), description="ngmix adaptive-moment ellipticity of the PSF "
           "stamp, epsilon convention"),
    Column("psf_T_hsm", "f8", unit="arcsec2", description="GalSim HSM 2 sigma^2 of the PSF "
           "stamp, sigma = det(M)^(1/4): a determinant size (the historical Tpsf)"),
    Column("psf_T_admom", "f8", unit="arcsec2", description="ngmix adaptive-moment trace "
           "Irr + Icc of the PSF stamp"),
    Column("psf_flags", "i4", description="non-zero where the PSF moments failed"),
    Column("flux_stamp", "f8", unit="count", description="sum of the noisy galaxy stamp"),
    Column("s2n_stamp", "f8", description="sqrt(sum I^2) / noise_sigma on the noisy stamp; "
           "not ngmix s2n"),
]

#: ngmix's own result names, kept as they are, per variant.
_NGMIX_FIELDS = [
    ("g", "f8", (2,), "", "fitted ellipticity, epsilon convention"),
    ("g_cov", "f8", (2, 2), "", "covariance of g from the fit"),
    ("T", "f8", (), "arcsec2", "fitted pre-PSF size Irr + Icc of the Gaussian model"),
    ("Tpsf", "f8", (), "arcsec2", "T of ngmix's fit to the PSF this variant was fitted "
     "with (the dilated reconvolution PSF for metacal products)"),
    ("flux", "f8", (), "count", "fitted flux"),
    ("s2n", "f8", (), "", "ngmix s2n of the fit"),
    ("flags", "i4", (), "", "ngmix flags; 0 is a good fit, 2**30 means no result"),
]


def _variant_columns(variant: str) -> List[Column]:
    where = ("the original stamp" if variant == "original"
             else f"the metacal {variant} product")
    return [Column(f"{name}_{variant}", dtype, shape, unit, f"{text}, on {where}")
            for name, dtype, shape, unit, text in _NGMIX_FIELDS]


def ngmix_columns(metacal: bool = True) -> List[Column]:
    columns = list(KEY_COLUMNS) + _variant_columns("original")
    if metacal:
        for variant in METACAL_TYPES:
            columns += _variant_columns(variant)
    return columns


def shearnet_columns(output_keys: Sequence[str], metacal: bool = True) -> List[Column]:
    """ShearNet's raw predictions, in physical label units, per variant."""
    variants = ["original"] + (list(METACAL_TYPES) if metacal else [])
    columns = list(KEY_COLUMNS)
    others = [k for k in output_keys if k not in ("g1", "g2")]
    units = {"hlr": "arcsec", "flux": "count", "psf_T": "arcsec2"}
    for variant in variants:
        where = ("the original stamp" if variant == "original"
                 else f"the metacal {variant} image/PSF pair ngmix fitted")
        if "g1" in output_keys:
            columns.append(Column(f"g_{variant}", "f8", (2,), description=(
                f"predicted (g1, g2) on {where}: the network's estimate of e_prepsf")))
        for key in others:
            columns.append(Column(f"{key}_{variant}", "f8", (), units.get(key, ""),
                                  f"predicted {key} on {where}"))
        columns.append(Column(f"flags_{variant}", "i4", description=(
            f"non-zero where the prediction on {where} is not finite"
            + ("" if variant == "original" else " or ngmix produced no such image"))))
    return columns


def schema_rows(tables: Dict[str, List[Column]]):
    """``(extension, column, dtype, shape, unit, description)`` for the SCHEMA HDU."""
    rows = []
    for extension, columns in tables.items():
        for column in columns:
            rows.append((extension, column.name, column.dtype,
                         "x".join(str(s) for s in column.shape) or "1", column.unit,
                         column.description))
    return rows


def markdown(output_keys: Sequence[str] = ("g1", "g2", "hlr", "flux")) -> str:
    """The column tables of ``docs/catalog.md`` (for a model with ``output_keys``)."""
    blocks = []
    tables = {
        "TRUTH": TRUTH_COLUMNS + [Column(f"label_{k}", "f8", (), "", f"training target {k}")
                                  for k in output_keys],
        "STAMP": STAMP_COLUMNS,
        "SHEARNET": [c for c in shearnet_columns(output_keys)
                     if c.name.endswith("_original") or c in KEY_COLUMNS],
        "NGMIX": [c for c in ngmix_columns() if c.name.endswith("_original")
                  or c in KEY_COLUMNS],
    }
    for name, columns in tables.items():
        lines = [f"### {name}", "", "| column | type | unit | meaning |", "|---|---|---|---|"]
        for column in columns:
            kind = column.dtype + (f" {'x'.join(map(str, column.shape))}" if column.shape else "")
            lines.append(f"| `{column.name}` | {kind} | {column.unit} | {column.description} |")
        if name in ("SHEARNET", "NGMIX"):
            lines += ["", "Every `*_original` column repeats for each metacal product with "
                      "the suffix `_noshear`, `_1p`, `_1m`, `_2p`, `_2m`, `_1p_psf`, "
                      "`_1m_psf`, `_2p_psf`, `_2m_psf`."]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"
