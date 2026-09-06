"""Per-object catalog schema policy for the evaluation FITS.

The evaluation harness measures every estimator on every ring station of every
sheared population, and the natural thing to do with that is to write all of it
out. At the production size -- ``n_obs`` 200000, ``component: both``, a
four-station ring -- that is 800000 rows carrying roughly 950 bytes each, and
the file lands near 1.2 GB. Most of those bytes are not recoverable science:
they are float64 where the quantity is known to three digits, responses for
corrections no reported number divides by, and columns holding one value
repeated 200000 times.

This module is the policy that decides what actually gets written. It is
deliberately independent of the harness: it takes the assembled ``{name: array}``
column dictionaries and returns filtered, downcast ones, matching on column-name
*prefixes and correction tokens* rather than on any knowledge of how ring
stations are suffixed. That keeps it working when the station scheme changes.

Three levels, chosen by ``eval.evaluate.catalog_level``:

``summary``
    No per-object tables at all. ``SUMMARY``, ``BINNED`` and ``LEAKSUM``
    survive, which is every number the ablation tables report (m1, c2, alpha,
    shape noise). Kilobytes. This is the right level for an ablation arm: the
    paper quotes four scalars from it and nothing else.

``paper`` (default)
    One (shape, response) pair per estimator -- the pair the reported number
    actually divides by -- plus the truth, PSF moments and S/N needed to
    re-derive m, c, alpha and beta and to bin any of them. float32. This is
    what to run the fiducial model at.

``full``
    Every column the harness measured, still downcast to float32. For
    debugging a response that is not behaving; not for a campaign.

Nothing here changes a measured value. The derived tables are computed from the
full-precision arrays before slimming, so ``SUMMARY`` is bit-for-bit unaffected
by the level -- a property :func:`shrink_fits.main` checks explicitly.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "CATALOG_LEVELS",
    "DEFAULT_LEVEL",
    "CORRECTION_TOKENS",
    "PAPER_KEEP_PREFIXES",
    "PAPER_DROP_TOKENS",
    "SlimReport",
    "slim_columns",
    "resolve_level",
    "estimate_row_bytes",
]

#: Ordered from smallest file to largest. ``resolve_level`` validates against it.
CATALOG_LEVELS = ("summary", "paper", "full")

#: What a run writes when the config does not say. ``paper`` keeps everything
#: needed to reconstruct every number in the paper and nothing else.
DEFAULT_LEVEL = "paper"

#: The correction vocabulary the harness uses in column names. A column name is
#: ``<quantity>_<estimator>[_<correction>][_<station suffix>]``, and these are
#: the only tokens that can appear in the correction slot. Matched as ``_<token>``
#: anywhere in the name so an unknown trailing station suffix cannot hide one.
CORRECTION_TOKENS = ("sim", "metacal", "anacal")

#: Columns kept at ``paper`` level regardless of estimator: the applied and
#: observed truth, the PSF moments the leakage fit regresses against, the S/N
#: the binned tables stratify by, and the catalog truth the size trend needs.
#: Matched with :func:`fnmatch.fnmatchcase`, so a station suffix is covered by
#: the trailing star.
PAPER_KEEP_PREFIXES = (
    "g_th*",       # observed truth shear, per station
    "gpsf*",       # PSF ellipticity -- the leakage x-axis
    "Tpsf*",       # PSF size -- the beta coefficient regresses on it
    "s2n",         # pair-mean galaxy S/N (the bare column, not s2n_ngmix)
    "hlr_th*",     # catalog half-light radius -- the size trend
    "flux_th*",    # catalog flux
    "flag_*",      # per-estimator failure flags
    "e_*",         # every shape, subject to PAPER_DROP_TOKENS below
    "R_*",         # every shear response, subject to the same
    "Rpsf_*",      # PSF response (LEAKAGE table)
)

#: At ``paper`` level, drop the correction each estimator's reported number does
#: NOT divide by. ngmix is scored through metacalibration, so its ``sim``
#: (scene-shear) response is a diagnostic; ShearNet is scored through the direct
#: ``sim`` route, so its ``metacal`` columns are the diagnostic. Both survive at
#: ``full``. Keyed by estimator, matched only when the estimator name is also in
#: the column, so an unrelated column is never caught by a bare token.
PAPER_DROP_TOKENS: Dict[str, Tuple[str, ...]] = {
    "ngmix": ("sim",),
    "shearnet": ("metacal",),
}

#: Dropped at ``paper`` level outright. ``Rbar_psf_*`` is an ensemble scalar the
#: harness broadcasts to one value per row; it is recorded in SLIMMETA instead.
#: The auxiliary size/flux predictions backed a per-galaxy accuracy table that
#: the current draft no longer carries.
PAPER_DROP_EXACT = (
    "Rbar_psf_*",
    "hlr_shearnet*",
    "flux_shearnet*",
    "s2n_ngmix*",
    "T_ngmix*",
    "flux_ngmix*",
)


class SlimReport:
    """What :func:`slim_columns` did, for the SLIMMETA table and the log.

    Attributes:
        kept: column names retained, in input order.
        dropped: ``{name: reason}`` for every column removed.
        constants: ``{name: value}`` for columns that held a single repeated
            value. The value is preserved here, so nothing is lost by dropping
            the column -- it is recoverable exactly.
        aliases: ``{dropped: kept}`` for columns bitwise identical to another
            retained column. ``e_<est>_raw`` equals ``e_<est>`` whenever no PSF
            response correction is applied, which is the default.
        before_bytes / after_bytes: nbytes summed over the column arrays.
    """

    def __init__(self) -> None:
        self.kept: List[str] = []
        self.dropped: Dict[str, str] = {}
        self.constants: Dict[str, float] = {}
        self.aliases: Dict[str, str] = {}
        self.before_bytes = 0
        self.after_bytes = 0

    @property
    def ratio(self) -> float:
        """Size before over size after, or 1.0 when there was nothing to do."""
        return self.before_bytes / max(self.after_bytes, 1)

    def log(self, table_name: str) -> None:
        """Emit a one-line summary at INFO and the drop reasons at DEBUG."""
        logger.info(
            "%s: %d -> %d columns, %.1f -> %.1f MB (%.1fx)",
            table_name, len(self.kept) + len(self.dropped), len(self.kept),
            self.before_bytes / 1e6, self.after_bytes / 1e6, self.ratio,
        )
        for name, reason in sorted(self.dropped.items()):
            logger.debug("%s: dropped %s (%s)", table_name, name, reason)

    def meta_rows(self) -> List[Tuple[str, str, str]]:
        """``(kind, name, value)`` triples for the SLIMMETA table.

        Written beside the slimmed tables so a reader can tell a column that was
        never measured from one that was dropped because it was constant, and
        can recover the constant.
        """
        rows: List[Tuple[str, str, str]] = []
        for name, value in sorted(self.constants.items()):
            rows.append(("constant", name, repr(value)))
        for dropped, kept in sorted(self.aliases.items()):
            rows.append(("alias", dropped, kept))
        for name, reason in sorted(self.dropped.items()):
            if name not in self.constants and name not in self.aliases:
                rows.append(("dropped", name, reason))
        return rows


def resolve_level(value: Optional[str]) -> str:
    """Validate a configured catalog level, defaulting when unset.

    Raises rather than silently falling back: a typo in the config would
    otherwise write a 1.2 GB file for fifty ablation arms.
    """
    if value is None:
        return DEFAULT_LEVEL
    level = str(value).strip().lower()
    if level not in CATALOG_LEVELS:
        raise ValueError(
            f"catalog_level {value!r} is not one of {CATALOG_LEVELS}. "
            "'summary' writes no per-object tables, 'paper' writes the columns "
            "every reported number is derived from, 'full' writes everything."
        )
    return level


def _estimators_in(name: str) -> List[str]:
    """Which known estimator names appear in a column name."""
    return [est for est in PAPER_DROP_TOKENS if est in name]


def _has_token(name: str, token: str) -> bool:
    """True when ``_<token>`` appears as a whole component of ``name``.

    Word-boundary matched so ``_sim`` does not fire on a hypothetical
    ``_similar``, and so a trailing ring-station suffix after the token is
    irrelevant -- which is the property that keeps this module independent of
    how stations are named.
    """
    return re.search(rf"_{re.escape(token)}(?:_|$)", name) is not None


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _paper_keep(name: str) -> Tuple[bool, str]:
    """Decide one column at ``paper`` level. Returns ``(keep, reason)``."""
    if _matches_any(name, PAPER_DROP_EXACT):
        return False, "not used by any reported number"
    if not _matches_any(name, PAPER_KEEP_PREFIXES):
        return False, "outside the paper column set"
    for estimator, tokens in PAPER_DROP_TOKENS.items():
        if estimator not in name:
            continue
        for token in tokens:
            if _has_token(name, token):
                return False, f"{estimator} is not scored through '{token}'"
    return True, ""


def _downcast(array: np.ndarray, float_dtype) -> np.ndarray:
    """float64 -> ``float_dtype``; integers narrowed to the smallest safe width.

    Only storage precision changes. Every reported quantity is a mean over at
    least 10^4 objects accumulated in float64 at read time, and the largest
    per-object value here is a flux of order 10^4, so float32's 7 significant
    digits sit five orders below the shape noise on any of them.
    """
    # Compared by kind and width, not against np.float64 directly: a column read
    # back from a FITS file carries big-endian '>f8', and `dtype == np.float64`
    # is False for it, so an identity check here silently skips every column
    # that came from disk -- which is exactly the path shrink_fits takes.
    if np.issubdtype(array.dtype, np.floating) and array.dtype.itemsize > np.dtype(float_dtype).itemsize:
        return array.astype(float_dtype, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        lo, hi = array.min(initial=0), array.max(initial=0)
        for candidate in (np.int8, np.int16, np.int32):
            info = np.iinfo(candidate)
            if lo >= info.min and hi <= info.max:
                return array.astype(candidate, copy=False)
    return array


def _constant_value(array: np.ndarray) -> Optional[float]:
    """The single value a 1-D column holds, or None if it varies.

    NaN-tolerant: an all-NaN column is constant too, and is worth dropping --
    it means the measurement never succeeded.
    """
    if array.ndim != 1 or array.size == 0:
        return None
    if not np.issubdtype(array.dtype, np.number):
        return None
    first = array[0]
    if np.isnan(first):
        return float("nan") if np.all(np.isnan(array)) else None
    if np.all(array == first):
        return float(first)
    return None


def slim_columns(
    columns: Mapping[str, np.ndarray],
    level: str = DEFAULT_LEVEL,
    *,
    float_dtype=np.float32,
    drop_constants: bool = True,
    drop_aliases: bool = True,
) -> Tuple[Dict[str, np.ndarray], SlimReport]:
    """Apply the schema policy to one table's columns.

    Args:
        columns: ``{name: array}`` as the harness assembles it. Not mutated.
        level: one of :data:`CATALOG_LEVELS`. ``summary`` returns no columns at
            all, which is the caller's signal to skip the HDU entirely.
        float_dtype: storage dtype for float64 columns. Pass ``np.float64`` to
            keep full precision while still dropping unused columns.
        drop_constants: replace a column holding one repeated value with a
            SLIMMETA entry recording that value.
        drop_aliases: drop a column bitwise identical to one already kept,
            recording which it duplicates.

    Returns:
        ``(kept_columns, report)``.
    """
    level = resolve_level(level)
    report = SlimReport()
    for array in columns.values():
        if array is not None:
            report.before_bytes += int(np.asarray(array).nbytes)

    if level == "summary":
        for name in columns:
            report.dropped[name] = "catalog_level=summary writes no rows"
        return {}, report

    kept: Dict[str, np.ndarray] = {}
    seen: Dict[bytes, str] = {}
    for name, raw in columns.items():
        if raw is None:
            report.dropped[name] = "empty"
            continue
        array = np.asarray(raw)

        if level == "paper":
            keep, reason = _paper_keep(name)
            if not keep:
                report.dropped[name] = reason
                continue

        if drop_constants:
            value = _constant_value(array)
            if value is not None:
                report.constants[name] = value
                report.dropped[name] = "constant across all rows"
                continue

        array = _downcast(array, float_dtype)

        if drop_aliases:
            # tobytes() is exact; two columns alias only if they are the same
            # array to the last bit, which is what `e_<est>_raw` is when no PSF
            # response correction was applied.
            digest = array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
            if digest in seen:
                report.aliases[name] = seen[digest]
                report.dropped[name] = f"identical to {seen[digest]}"
                continue
            seen[digest] = name

        kept[name] = array
        report.kept.append(name)
        report.after_bytes += int(array.nbytes)

    return kept, report


def estimate_row_bytes(columns: Mapping[str, np.ndarray]) -> float:
    """Bytes per row across a column dict, for the preflight budget line."""
    total = 0
    rows = 0
    for array in columns.values():
        if array is None:
            continue
        array = np.asarray(array)
        if array.size == 0:
            continue
        rows = max(rows, array.shape[0])
        total += int(array.nbytes)
    return total / max(rows, 1)


def meta_table(reports: Sequence[Tuple[str, SlimReport]]):
    """An ``astropy.table.Table`` of every drop, for the SLIMMETA HDU.

    Carries the table each row came from so a reader can tell a column dropped
    from ``LEAKAGE`` from the same name dropped from ``TAB_P``.
    """
    from astropy.table import Table

    table_names: List[str] = []
    kinds: List[str] = []
    names: List[str] = []
    values: List[str] = []
    for table_name, report in reports:
        for kind, name, value in report.meta_rows():
            table_names.append(table_name)
            kinds.append(kind)
            names.append(name)
            values.append(value)
    if not names:
        # An empty Table with no columns cannot round-trip through BinTableHDU,
        # so give it the schema with zero rows.
        return Table(names=("table", "kind", "column", "value"),
                     dtype=("U16", "U16", "U64", "U64"))
    return Table({"table": table_names, "kind": kinds,
                  "column": names, "value": values})


def write_evaluation_fits(
    path,
    *,
    primary_header,
    tables: Mapping[int, Mapping[str, Mapping[str, np.ndarray]]],
    leakage_columns: Mapping[str, np.ndarray],
    derived: Sequence[Tuple[str, object]],
    level: str = DEFAULT_LEVEL,
    float_dtype=np.float32,
):
    """Write the evaluation FITS with the per-object tables slimmed to ``level``.

    A drop-in for the tail of the harness's own writer, so wiring it in is an
    import and one delegating call rather than a rewrite. The derived tables in
    ``derived`` are written verbatim: they were computed inside the harness from
    the full-precision arrays, so they are unaffected by the level and remain the
    authoritative numbers whatever the per-object tables were trimmed to.

    Args:
        path: destination, created with parents.
        primary_header: the header the harness already populated with the run's
            provenance keys.
        tables: ``tables[component][population]`` of ``{name: array}``, exactly
            as :func:`_measure_shear_pair` returns it.
        leakage_columns: the unsheared population's columns.
        derived: ``[(hdu_name, astropy_table), ...]`` in the order to write.
        level: one of :data:`CATALOG_LEVELS`.

    Returns:
        The path written.
    """
    from pathlib import Path

    from astropy.io import fits
    from astropy.table import Table

    level = resolve_level(level)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    primary = fits.PrimaryHDU(header=primary_header)
    primary.header["CATLEVEL"] = (level, "per-object catalog schema level")
    hdus = [primary]
    reports: List[Tuple[str, SlimReport]] = []

    # The first measured component keeps the historical TAB_P / TAB_M names, so
    # a config measuring only g1 writes the file it always did.
    for order, component in enumerate(sorted(tables)):
        tag = "" if order == 0 else str(order + 1)
        for label, name in (("plus", f"TAB_P{tag}"), ("minus", f"TAB_M{tag}")):
            kept, report = slim_columns(
                tables[component][label], level, float_dtype=float_dtype
            )
            report.log(name)
            reports.append((name, report))
            if not kept:
                continue
            hdu = fits.BinTableHDU(Table(kept), name=name)
            hdu.header["COMPONEN"] = (component, "sheared component: 0 = g1, 1 = g2")
            hdus.append(hdu)

    kept, report = slim_columns(leakage_columns, level, float_dtype=float_dtype)
    report.log("LEAKAGE")
    reports.append(("LEAKAGE", report))
    if kept:
        hdus.append(fits.BinTableHDU(Table(kept), name="LEAKAGE"))

    for name, table in derived:
        hdus.append(fits.BinTableHDU(table, name=name))
    hdus.append(fits.BinTableHDU(meta_table(reports), name="SLIMMETA"))

    fits.HDUList(hdus).writeto(path, overwrite=True)
    total = sum(report.after_bytes for _, report in reports)
    logger.info("Saved evaluation to %s (level=%s, %.1f MB of rows)",
                path, level, total / 1e6)
    return path
