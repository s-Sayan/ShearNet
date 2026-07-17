#!/usr/bin/env python
"""
Evidence figure: the two INDEPENDENT ways to measure PSF leakage, side by side
for ngmix (a shape estimator we trust) and ShearNet.

  (1) population leakage slope  alpha = d<e_gal>/d(e_psf)   [binned points + fit]
  (2) metacal PSF response      R_psf = (e_1p_psf - e_1m_psf)/(2 step)  [a line]

For a well-behaved (PSF-deconvolving) estimator these MUST coincide -- and for
ngmix they do. If for ShearNet the R_psf line badly misses the binned data while
the population-slope line goes through it, then R_psf is NOT ShearNet's leakage.
The reader can see this directly; no knowledge of the estimator's internals is
needed. ngmix is the control that proves the measurement itself is sound.

Reads one psf_leakage FITS. Uses the RAW (uncorrected) shapes:
    ngmix    : g_raw   (falls back to g)
    ShearNet : g_sn_raw (falls back to g_sn)
    e_psf    : gpsf
    R_psf    : r11_psf / r11_psf_sn (skipped if absent)

Usage:
    python leakage_evidence.py LEAKAGE.fits --out leakage_evidence.png \
        [--label "ShearNet fork-like, no correction"]
"""
import argparse
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _col(tab, *names):
    for n in names:
        if n in tab.colnames:
            return np.asarray(tab[n], dtype=float)
    return None


def binned(x, y, nbin=12, min_count=30):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    edges = np.percentile(x, np.linspace(0, 100, nbin + 1))
    idx = np.digitize(x, edges)
    xs, ys, es = [], [], []
    for i in range(1, len(edges)):
        sel = idx == i
        if sel.sum() >= min_count:
            xs.append(np.mean(x[sel]))
            ys.append(np.mean(y[sel]))
            es.append(np.std(y[sel]) / np.sqrt(sel.sum()))
    return np.array(xs), np.array(ys), np.array(es)


def pop_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    b1, b0 = np.polyfit(x[m], y[m], 1)
    resid = y[m] - (b1 * x[m] + b0)
    sxx = np.sum((x[m] - x[m].mean()) ** 2)
    serr = np.sqrt(np.sum(resid**2) / (m.sum() - 2) / sxx)
    return b1, b0, serr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fits")
    ap.add_argument("--out", default="leakage_evidence.png")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    tab = Table.read(args.fits)
    gpsf = _col(tab, "gpsf")

    estimators = [
        ("ngmix", _col(tab, "g_raw", "g"), _col(tab, "r11_psf"), "#3B4CC0"),
        ("ShearNet", _col(tab, "g_sn_raw", "g_sn"), _col(tab, "r11_psf_sn"), "#2ca02c"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, comp in enumerate((0, 1)):
        ep = gpsf[:, comp]
        for col, (name, eg, rr, color) in enumerate(estimators):
            ax = axes[row, col]
            if eg is None:
                ax.set_visible(False)
                continue
            xb, yb, eb = binned(ep, eg[:, comp])
            a, b0, ae = pop_slope(ep, eg[:, comp])
            xs = np.linspace(np.nanpercentile(ep, 1), np.nanpercentile(ep, 99), 50)

            ax.errorbar(xb, yb, yerr=eb, fmt="o", ms=6, color=color, zorder=3,
                        label=r"data: $\langle e\rangle$ vs $e^{PSF}$")
            ax.plot(xs, a * xs + b0, "-", color=color, lw=2.5, zorder=2,
                    label=fr"population leakage $\alpha={a:+.3f}\pm{ae:.3f}$")
            if rr is not None:
                R = np.nanmean(rr)
                # R_psf line drawn through the data mean so slopes compare fairly
                ax.plot(xs, R * (xs - np.nanmean(ep)) + np.nanmean(yb),
                        "r--", lw=2.5, zorder=1,
                        label=fr"metacal response $R_{{psf}}={R:+.3f}$")
            ax.axhline(0, color="grey", ls=":", lw=0.8)
            ax.set_xlabel(fr"$e_{comp+1}^{{PSF}}$")
            ax.set_ylabel(fr"$\langle e_{comp+1}\rangle$ (raw, uncorrected)")
            ax.set_title(f"{name}  —  $e_{comp+1}$")
            ax.legend(fontsize=9, loc="best")

    sup = "Two independent leakage measures: population slope (data) vs metacal R_psf"
    if args.label:
        sup += f"\n{args.label}"
    fig.suptitle(sup, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved -> {args.out}")

    # printed numbers for the caption / table
    print("\n  estimator   comp   population_alpha      R_psf     ratio R/alpha")
    for name, eg, rr, _ in estimators:
        if eg is None:
            continue
        for comp in (0, 1):
            a, _, ae = pop_slope(gpsf[:, comp], eg[:, comp])
            R = np.nanmean(rr) if rr is not None else np.nan
            print(f"  {name:9s}   e{comp+1}   {a:+8.4f} +/- {ae:.4f}   "
                  f"{R:+8.4f}   {R/a if a else np.nan:8.1f}")
    print("\nRead: where R_psf ~ alpha (ngmix), the two agree -> R_psf is the leakage.")
    print("      where R_psf >> alpha (ShearNet), the dashed line misses the data")
    print("      -> R_psf is NOT the leakage; the data define alpha.")


if __name__ == "__main__":
    main()
