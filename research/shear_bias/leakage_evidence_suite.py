#!/usr/bin/env python
"""
PSF-leakage evidence SUITE across three runs of the SAME model:

    --metacal : PSF response measured & corrected via metacal (deconv + dilation)
    --direct  : PSF response measured via skip-deconvolution (shear the real PSF)
    --none    : no PSF-response machinery at all (plain galaxy (x) PSF)

Each run supplies two FITS: the PSF-leakage benchmark and the m/bias benchmark.
Pass a run directory (paths derived by convention) or the FITS directly.

It produces four figures that let the reader reach two conclusions on their own,
using only shear-measurement quantities (alpha, R_psf, m, c) and ngmix as a
trusted control -- no neural-network reasoning required:

  A_physical_leakage.png : the RAW (uncorrected) leakage slope alpha is the same
      in all three runs -> the implementations agree on the physical quantity,
      and for ShearNet it is small.
  B_three_measures.png   : for ngmix the population alpha and BOTH responses
      (metacal, direct) coincide -> the response methods are implemented
      correctly; for ShearNet both responses are ~50x the physical alpha ->
      neither response gives ShearNet a physical leakage.
  C_correction_identity.png : the metacal-corrected shape equals
      raw - R_psf * e_psf to machine precision -> the correction code is correct;
      the dramatic -0.6 slope is that exact arithmetic with a coefficient
      (R_psf) that is not the leakage.
  D_bias_mc.png          : the SAME correction reduces ngmix's |c| but inflates
      ShearNet's |c|, with m unchanged in both.

Usage:
    python leakage_evidence_suite.py \
        --metacal research/unit_test_variations/fourth_psf_response_corrected \
        --direct  research/unit_test_variations/fourth_psf_no_deconv \
        --none    research/unit_tests/fourth \
        --outdir  evidence_suite
"""
import argparse
import os
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_NG, C_SN = "#3B4CC0", "#2ca02c"


def leak_path(p):
    if p is None or p.endswith(".fits"):
        return p
    return os.path.join(p, "benchmarking", "psf_leakage", "psf_leakage_results.fits")


def m_path(p):
    if p is None or p.endswith(".fits"):
        return p
    return os.path.join(p, "benchmarking", "m", "metacal_results.fits")


def col(tab, *names):
    for n in names:
        if tab is not None and n in tab.colnames:
            return np.asarray(tab[n], float)
    return None


def binned(x, y, nbin=11, min_count=30):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    edges = np.percentile(x, np.linspace(0, 100, nbin + 1))
    idx = np.digitize(x, edges)
    xs, ys, es = [], [], []
    for i in range(1, len(edges)):
        s = idx == i
        if s.sum() >= min_count:
            xs.append(x[s].mean()); ys.append(y[s].mean())
            es.append(y[s].std() / np.sqrt(s.sum()))
    return np.array(xs), np.array(ys), np.array(es)


def pop_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    b1, b0 = np.polyfit(x[m], y[m], 1)
    resid = y[m] - (b1 * x[m] + b0)
    sxx = np.sum((x[m] - x[m].mean()) ** 2)
    se = np.sqrt(np.sum(resid ** 2) / (m.sum() - 2) / sxx)
    return b1, b0, se


def sn_raw(tab):
    return col(tab, "g_sn_raw", "g_sn")


def ng_raw(tab):
    return col(tab, "g_raw", "g")


# ---------------------------------------------------------------- Figure A
def fig_physical(runs, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    markers = {"metacal": "o", "direct": "s", "none": "^"}
    for r_i, comp in enumerate((0, 1)):
        for c_i, (name, getter, color) in enumerate(
                [("ngmix", ng_raw, C_NG), ("ShearNet", sn_raw, C_SN)]):
            ax = axes[r_i, c_i]
            txt = []
            for label, tab in runs.items():
                eg = getter(tab)
                gp = col(tab, "gpsf")
                if eg is None or gp is None:
                    continue
                x, y, e = binned(gp[:, comp], eg[:, comp])
                a, _, ae = pop_slope(gp[:, comp], eg[:, comp])
                ax.errorbar(x, y, yerr=e, fmt=markers.get(label, "o"), ms=5,
                            color=color, alpha=0.8, mfc="none",
                            label=f"{label}: α={a:+.3f}±{ae:.3f}")
                txt.append(a)
            ax.axhline(0, color="grey", ls=":", lw=0.8)
            ax.set_xlabel(fr"$e_{comp+1}^{{PSF}}$")
            ax.set_ylabel(fr"$\langle e_{comp+1}\rangle$ (RAW, uncorrected)")
            ax.set_title(f"{name} — physical leakage $e_{comp+1}$  (all runs overlaid)")
            ax.legend(fontsize=8)
    fig.suptitle("A. The RAW leakage slope is the same in all three runs "
                 "(implementations agree; ShearNet's is small)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure B
def fig_three_measures(runs, outpath):
    """For e1: population alpha vs metacal R_psf vs direct R_psf, per estimator."""
    def alpha_e1(getter):
        # physical leakage from the plainest available run
        for pref in ("none", "direct", "metacal"):
            tab = runs.get(pref)
            eg, gp = getter(tab), col(tab, "gpsf")
            if eg is not None and gp is not None:
                a, _, ae = pop_slope(gp[:, 0], eg[:, 0])
                return a, ae
        return np.nan, np.nan

    def rpsf(run_key, rcol):
        v = col(runs.get(run_key), rcol)
        return np.nanmean(v) if v is not None else np.nan

    data = {
        "ngmix": dict(alpha=alpha_e1(ng_raw),
                      Rm=rpsf("metacal", "r11_psf"),
                      Rd=rpsf("direct", "r11_psf"), color=C_NG),
        "ShearNet": dict(alpha=alpha_e1(sn_raw),
                         Rm=rpsf("metacal", "r11_psf_sn"),
                         Rd=rpsf("direct", "r11_psf_sn"), color=C_SN),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, d) in zip(axes, data.items()):
        vals = [d["alpha"][0], d["Rm"], d["Rd"]]
        errs = [d["alpha"][1], 0, 0]
        labels = ["population α\n(the physical leakage)",
                  "metacal $R_{psf}$\n(response)",
                  "direct $R_{psf}$\n(response)"]
        colors = [d["color"], "#B40426", "#ff7f0e"]
        bars = ax.bar(range(3), vals, yerr=errs, color=colors, alpha=0.85, capsize=4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=11)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(r"$e_1$ response / leakage")
        ax.set_title(f"{name}")
    fig.suptitle("B. Two response methods vs the physical leakage (e₁).  "
                 "ngmix: all three agree → methods are valid.  "
                 "ShearNet: both responses ≫ physical → neither is its leakage.",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure C
def fig_correction_identity(runs, outpath):
    tab = runs.get("metacal")
    gsn = col(tab, "g_sn")
    gsn_raw = col(tab, "g_sn_raw")
    gpsf = col(tab, "gpsf")
    rbar = col(tab, "rbar_psf_sn")
    if gsn is None or gsn_raw is None or gpsf is None or rbar is None:
        return False
    R = np.nanmean(rbar)
    predicted = gsn_raw - R * gpsf          # what the correction claims to do
    resid = gsn - predicted

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot(predicted[:, 0], gsn[:, 0], ".", ms=2, alpha=0.2)
    lo, hi = np.nanpercentile(gsn[:, 0], [1, 99])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(r"$e_1^{raw} - R_{psf}\, e_1^{PSF}$  (claimed correction)")
    ax.set_ylabel(r"stored $g\_sn$ (corrected column)")
    ax.set_title(f"metacal correction identity  (max|resid| = {np.nanmax(np.abs(resid)):.1e})")

    ax = axes[1]
    a_raw, _, _ = pop_slope(gpsf[:, 0], gsn_raw[:, 0])
    a_cor, _, _ = pop_slope(gpsf[:, 0], gsn[:, 0])
    ax.bar([0, 1, 2], [a_raw, -R, a_cor],
           color=[C_SN, "#B40426", "grey"], alpha=0.85)
    for i, v in enumerate([a_raw, -R, a_cor]):
        ax.text(i, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=11)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["raw slope\nα(g_sn_raw)", "−R_psf", "corrected slope\nα(g_sn)"],
                       fontsize=9)
    ax.set_ylabel(r"$e_1$ leakage slope")
    ax.set_title(f"corrected = raw − R_psf:  {a_cor:+.3f} = {a_raw:+.3f} + ({-R:+.3f})")
    fig.suptitle("C. The correction does EXACTLY raw − R_psf·e_psf (code is correct); "
                 "the −0.6 slope is that arithmetic with a non-leakage coefficient.",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- Figure D
def jackknife_mc(tp, tm, shear_true, njac, g_col, r11_col):
    gp, gm = np.asarray(tp[g_col], float), np.asarray(tm[g_col], float)
    Rp, Rm = np.asarray(tp[r11_col], float), np.asarray(tm[r11_col], float)
    N = len(gp)
    gamma1 = (gp[:, 0] - gm[:, 0]) / 2.0
    c_per = (gp[:, 1] + gm[:, 1]) / 2.0
    R1 = 0.5 * (Rp + Rm)
    chunks = np.array_split(np.arange(N), njac)
    mj, cj = [], []
    for ch in chunks:
        mask = np.ones(N, bool); mask[ch] = False
        mj.append(np.nanmean(gamma1[mask]) / np.nanmean(R1[mask]) / shear_true - 1)
        cj.append(np.nanmean(c_per[mask]))
    mj, cj = np.array(mj), np.array(cj)
    me = np.sqrt((njac - 1) * np.mean((mj - mj.mean()) ** 2))
    ce = np.sqrt((njac - 1) * np.mean((cj - cj.mean()) ** 2))
    return mj.mean(), me, cj.mean(), ce


def fig_bias(m_metacal, shear_true, njac, outpath):
    if m_metacal is None:
        return False
    tp = Table.read(m_metacal, hdu="TAB_P")
    tm = Table.read(m_metacal, hdu="TAB_M")

    def have(c):
        return c in tp.colnames and c in tm.colnames

    spec = [
        ("ngmix", "uncorrected", "g_noshear_raw", "r11", C_NG),
        ("ngmix", "R_psf-corr", "g_noshear", "r11", C_NG),
        ("ShearNet", "uncorrected", "g_sn_noshear_raw", "r11_sn", C_SN),
        ("ShearNet", "R_psf-corr", "g_sn_noshear", "r11_sn", C_SN),
    ]
    res = []
    for est, setting, gc, rc, color in spec:
        if not (have(gc) and have(rc)):
            res.append((est, setting, np.nan, np.nan, np.nan, np.nan, color))
            continue
        m, me, c, ce = jackknife_mc(tp, tm, shear_true, njac, gc, rc)
        res.append((est, setting, m, me, c, ce, color))

    fig, (axm, axc) = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = [f"{e}\n{s}" for e, s, *_ in res]
    xs = np.arange(len(res))
    hatch = ["", "//", "", "//"]
    axm.bar(xs, [r[2] for r in res], yerr=[r[3] for r in res],
            color=[r[6] for r in res], hatch=hatch, alpha=0.85, capsize=4)
    axm.axhline(0, color="grey", lw=0.8)
    axm.set_xticks(xs); axm.set_xticklabels(labels, fontsize=8)
    axm.set_title("multiplicative bias m  (should be unchanged by a leakage correction)")
    axc.bar(xs, [abs(r[4]) for r in res], yerr=[r[5] for r in res],
            color=[r[6] for r in res], hatch=hatch, alpha=0.85, capsize=4)
    axc.set_xticks(xs); axc.set_xticklabels(labels, fontsize=8)
    axc.set_title("additive bias |c|  (a real correction REDUCES it)")
    for i, r in enumerate(res):
        if np.isfinite(r[4]):
            axc.text(i, abs(r[4]), f"{abs(r[4]):.1e}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("D. Same R_psf correction: ngmix |c| DOWN (real leakage removed), "
                 "ShearNet |c| UP (bias manufactured); m unchanged. (hatch = corrected)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metacal", required=True)
    ap.add_argument("--direct", required=True)
    ap.add_argument("--none", dest="none_", required=True)
    ap.add_argument("--outdir", default="evidence_suite")
    ap.add_argument("--shear-true", type=float, default=0.01)
    ap.add_argument("--njac", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs = {}
    for key, p in (("metacal", args.metacal), ("direct", args.direct), ("none", args.none_)):
        lp = leak_path(p)
        runs[key] = Table.read(lp) if lp and os.path.exists(lp) else None
        print(f"{key:8s} leakage: {lp}  ->  "
              f"{'loaded ' + str(len(runs[key])) + ' rows' if runs[key] is not None else 'MISSING'}")

    fig_physical(runs, os.path.join(args.outdir, "A_physical_leakage.png"))
    fig_three_measures(runs, os.path.join(args.outdir, "B_three_measures.png"))
    okC = fig_correction_identity(runs, os.path.join(args.outdir, "C_correction_identity.png"))
    okD = fig_bias(m_path(args.metacal), args.shear_true, args.njac,
                   os.path.join(args.outdir, "D_bias_mc.png"))

    print("\nwrote:")
    for f in ("A_physical_leakage.png", "B_three_measures.png",
              "C_correction_identity.png", "D_bias_mc.png"):
        p = os.path.join(args.outdir, f)
        print(f"  {p}  {'(ok)' if os.path.exists(p) else '(skipped - missing columns)'}")


if __name__ == "__main__":
    main()
