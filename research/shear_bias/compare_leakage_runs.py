#!/usr/bin/env python
"""
Compare two PSF-leakage runs object-by-object to explain why two runs can have
(almost) the same ensemble R11_psf but very different PSF-leakage plots.

Typical use: run A = metacal (deconvolution + dilation, psf_response: true),
             run B = direct (skip deconvolution, psf_response_direct_*: true).

Because both runs use the same eval.seed and n_obs, object i in file A is the
SAME galaxy / PSF / field position as object i in file B, so we can compare them
row-by-row.

Key columns in a psf_leakage FITS (written by helpers.leakage_response*_to_table):
  gpsf       (N,2)  PSF ellipticity = moments of obs.psf  <-- x-axis of leakage plot
  g_raw      (N,2)  ngmix noshear shape, UNcorrected
  g          (N,2)  ngmix noshear shape (corrected in metacal run, == g_raw in direct)
  g_sn_raw   (N,2)  ShearNet noshear shape, UNcorrected
  g_sn       (N,2)  ShearNet noshear shape (corrected iff psf_response_shearnet)
  r11_psf, r11_psf_sn (N,)  per-object PSF response
  rbar_psf, rbar_psf_sn (N,) constant ensemble PSF response
  g_th, gal_flux_th, gal_hlr_th, s2n, T, Tpsf

The two things this settles:
  1. Are e_psf (gpsf) and the RAW galaxy shapes actually the same between runs?
     -> the "do I measure PSF moments on the raw or the dilated PSF?" question.
  2. Within EACH run, does the population leakage slope alpha (raw g_sn vs e_psf)
     equal the ensemble R11_psf? If not, R11_psf is NOT a valid leakage measure
     and correcting with it will damage the real (small) leakage.

Usage:
    python compare_leakage_runs.py METACAL.fits DIRECT.fits \
        --labels metacal direct --out leakage_compare.png
"""
import argparse
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _col(tab, name):
    return np.asarray(tab[name], dtype=float) if name in tab.colnames else None


def slope(x, y):
    """Least-squares slope of y vs x over finite points (population leakage alpha)."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    b1, b0 = np.polyfit(x[m], y[m], 1)
    # crude error on slope
    resid = y[m] - (b1 * x[m] + b0)
    sxx = np.sum((x[m] - x[m].mean()) ** 2)
    serr = np.sqrt(np.sum(resid**2) / (m.sum() - 2) / sxx) if sxx > 0 else np.nan
    return b1, serr


def summarize(tag, tab):
    print(f"\n===== {tag} =====")
    print(f"rows: {len(tab)}")
    print(f"columns: {tab.colnames}")
    gpsf = _col(tab, "gpsf")
    out = {"gpsf": gpsf}
    for comp, lab in ((0, "1"), (1, "2")):
        ep = gpsf[:, comp]
        print(f"\n  component {lab}:  <e{lab}_psf> = {np.nanmean(ep):+.4f}  "
              f"std = {np.nanstd(ep):.4f}")
        for col in ("g_raw", "g", "g_sn_raw", "g_sn"):
            arr = _col(tab, col)
            if arr is None:
                continue
            a, ae = slope(ep, arr[:, comp])
            print(f"    leakage slope alpha{lab}({col:9s} vs e{lab}_psf) = "
                  f"{a:+.4f} +/- {ae:.4f}")
        for col in ("r11_psf", "r11_psf_sn"):
            arr = _col(tab, col)
            if arr is not None:
                print(f"    <{col}> (ensemble PSF response)      = {np.nanmean(arr):+.4f}")
        for col in ("rbar_psf", "rbar_psf_sn"):
            arr = _col(tab, col)
            if arr is not None:
                print(f"    {col} (constant)                    = {np.nanmean(arr):+.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fits_a")
    ap.add_argument("fits_b")
    ap.add_argument("--labels", nargs=2, default=["run_A", "run_B"])
    ap.add_argument("--out", default="leakage_compare.png")
    args = ap.parse_args()

    la, lb = args.labels
    ta = Table.read(args.fits_a)
    tb = Table.read(args.fits_b)

    sa = summarize(la, ta)
    sb = summarize(lb, tb)

    # ---- object matching check (same seed/n_obs -> row i is the same object) ----
    n = min(len(ta), len(tb))
    if len(ta) != len(tb):
        print(f"\n[WARN] row counts differ ({len(ta)} vs {len(tb)}); "
              f"comparing the first {n} rows by index.")
    fa, fb = _col(ta, "gal_flux_th"), _col(tb, "gal_flux_th")
    if fa is not None and fb is not None:
        d = np.nanmax(np.abs(fa[:n] - fb[:n]))
        print(f"\n[match check] max |gal_flux_th_A - gal_flux_th_B| over {n} rows "
              f"= {d:.3e}  ({'MATCHED by row' if d < 1e-3 else 'NOT matched - do not trust per-object scatter'})")

    # ---- per-object comparison scatter + leakage overlay ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def scat(ax, xa, xb, title, unit_line=True):
        m = np.isfinite(xa) & np.isfinite(xb)
        ax.plot(xa[m], xb[m], ".", ms=2, alpha=0.25)
        if unit_line and m.sum():
            lo = np.nanpercentile(np.concatenate([xa[m], xb[m]]), 1)
            hi = np.nanpercentile(np.concatenate([xa[m], xb[m]]), 99)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_title(title)

    # row 0: are e_psf and the RAW shapes the same object-by-object?
    scat(axes[0, 0], sa["gpsf"][:n, 0], sb["gpsf"][:n, 0], r"$e_1^{PSF}$  A vs B")
    gsr_a, gsr_b = _col(ta, "g_sn_raw"), _col(tb, "g_sn_raw")
    if gsr_a is not None and gsr_b is not None:
        scat(axes[0, 1], gsr_a[:n, 0], gsr_b[:n, 0], r"ShearNet $e_1$ (RAW)  A vs B")
    gr_a, gr_b = _col(ta, "g_raw"), _col(tb, "g_raw")
    if gr_a is not None and gr_b is not None:
        scat(axes[0, 2], gr_a[:n, 0], gr_b[:n, 0], r"ngmix $e_1$ (RAW)  A vs B")

    # row 1: leakage within each run (RAW ShearNet shape vs e_psf) + slope
    def leak_panel(ax, tab, tag, color):
        gpsf = _col(tab, "gpsf")[:, 0]
        graw = _col(tab, "g_sn_raw")
        xs = np.linspace(np.nanpercentile(gpsf, 1), np.nanpercentile(gpsf, 99), 50)
        if graw is not None:
            a, _ = slope(gpsf, graw[:, 0])
            ax.plot(gpsf, graw[:, 0], ".", ms=2, alpha=0.15, color=color)
            ax.plot(xs, a * xs + np.nanmean(graw[:, 0]) - a * np.nanmean(gpsf),
                    color=color, lw=2, label=f"raw α₁={a:+.3f}")
            ax.set_ylim(np.nanpercentile(graw[:, 0], [1, 99]) * 3)
        rr = _col(tab, "r11_psf_sn")
        if rr is not None:
            R = np.nanmean(rr)
            ax.plot(xs, R * xs, "r--", lw=2, label=f"slope=R11_psf_sn={R:+.3f}")
        ax.axhline(0, color="grey", ls=":", lw=0.8)
        ax.set_xlabel(r"$e_1^{PSF}$"); ax.set_ylabel(r"ShearNet $e_1$ (raw)")
        ax.set_title(f"{tag}: leakage vs R11_psf_sn")
        ax.legend(fontsize=8)

    leak_panel(axes[1, 0], ta, la, "C0")
    leak_panel(axes[1, 1], tb, lb, "C2")

    # corrected vs raw ShearNet shape, A vs B, to expose the correction artifact
    gc_a, gc_b = _col(ta, "g_sn"), _col(tb, "g_sn")
    if gc_a is not None and gc_b is not None:
        scat(axes[1, 2], gc_a[:n, 0], gc_b[:n, 0], r"ShearNet $e_1$ (PLOTTED col g_sn) A vs B")

    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved figure -> {args.out}")
    print("\nWhat to look for:")
    print("  * top-left ~ y=x  -> e_psf is the SAME between runs (PSF moments not the cause)")
    print("  * top-middle ~ y=x -> RAW ShearNet shapes agree (image differences not the cause)")
    print("  * bottom panels: if the fitted raw leakage slope alpha differs from")
    print("    R11_psf_sn, then R11_psf_sn is NOT the population leakage and")
    print("    correcting g_sn with it (metacal run) DAMAGES the true (small) leakage.")


if __name__ == "__main__":
    main()
