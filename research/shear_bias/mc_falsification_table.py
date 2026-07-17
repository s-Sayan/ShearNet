#!/usr/bin/env python
"""
m/c falsification table: does the metacal PSF-response correction help or hurt?

Reads an m/bias FITS (TAB_P / TAB_M, written by research/shear_bias/m/main.py
with eval.bias.psf_response[_shearnet]: true so both raw and corrected shape
columns are present) and prints the multiplicative (m) and additive (c) bias for:

    ngmix     uncorrected   (g_noshear_raw)
    ngmix     R_psf-corrected (g_noshear)
    ShearNet  uncorrected   (g_sn_noshear_raw)
    ShearNet  R_psf-corrected (g_sn_noshear)

Interpretation (all in shear-bias language, no neural-net knowledge needed):
  A genuine PSF-leakage correction should REDUCE |c| and leave m unchanged.
  * For ngmix (a PSF-deconvolving estimator) the metacal R_psf correction does
    exactly that -> R_psf is ngmix's real leakage.
  * If for ShearNet the same correction INCREASES |c|, then the quantity being
    subtracted (R_psf ~ 0.6) is not ShearNet's leakage; applying it manufactures
    an additive bias rather than removing one. m being ~unchanged confirms the
    effect is purely additive (a c effect), as PSF leakage must be.

Usage:
    python mc_falsification_table.py metacal_results.fits [--shear-true 0.01] [--njac 20]
"""
import argparse
import numpy as np
from astropy.table import Table


def jackknife_mc(tab_p, tab_m, shear_true, njac, g_col, r11_col):
    """Compact copy of m/helpers.jackknife_mc_v2 -> (m, m_err, c, c_err)."""
    gp = np.asarray(tab_p[g_col], float)
    gm = np.asarray(tab_m[g_col], float)
    Rp = np.asarray(tab_p[r11_col], float)
    Rm = np.asarray(tab_m[r11_col], float)
    N = len(gp)

    gamma1 = (gp[:, 0] - gm[:, 0]) / 2.0
    c_per = (gp[:, 1] + gm[:, 1]) / 2.0
    R1 = 0.5 * (Rp + Rm)

    chunks = np.array_split(np.arange(N), njac)
    m_jk, c_jk = [], []
    for chunk in chunks:
        mask = np.ones(N, bool)
        mask[chunk] = False
        shear = np.nanmean(gamma1[mask]) / np.nanmean(R1[mask])
        m_jk.append(shear / shear_true - 1.0)
        c_jk.append(np.nanmean(c_per[mask]))
    m_jk, c_jk = np.array(m_jk), np.array(c_jk)
    m_mean, c_mean = m_jk.mean(), c_jk.mean()
    m_err = np.sqrt((njac - 1) * np.mean((m_jk - m_mean) ** 2))
    c_err = np.sqrt((njac - 1) * np.mean((c_jk - c_mean) ** 2))
    return m_mean, m_err, c_mean, c_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fits")
    ap.add_argument("--shear-true", type=float, default=0.01)
    ap.add_argument("--njac", type=int, default=20)
    args = ap.parse_args()

    tp = Table.read(args.fits, hdu="TAB_P")
    tm = Table.read(args.fits, hdu="TAB_M")

    def have(c):
        return c in tp.colnames and c in tm.colnames

    rows = [
        ("ngmix",    "uncorrected",     "g_noshear_raw",    "g_noshear",    "r11"),
        ("ngmix",    "R_psf-corrected", "g_noshear",        None,           "r11"),
        ("ShearNet", "uncorrected",     "g_sn_noshear_raw", "g_sn_noshear", "r11_sn"),
        ("ShearNet", "R_psf-corrected", "g_sn_noshear",     None,           "r11_sn"),
    ]

    print(f"\nFile: {args.fits}   (shear_true={args.shear_true}, njac={args.njac})")
    print(f"Columns: {tp.colnames}\n")
    print(f"{'estimator':9s}  {'setting':16s}  {'m':>22s}  {'c':>22s}")
    print("-" * 76)
    results = {}
    for est, setting, col, fallback, rcol in rows:
        use = col if have(col) else (fallback if fallback and have(fallback) else None)
        if use is None or not have(rcol):
            print(f"{est:9s}  {setting:16s}  {'(column missing)':>22s}")
            continue
        m, me, c, ce = jackknife_mc(tp, tm, args.shear_true, args.njac, use, rcol)
        results[(est, setting)] = (m, c)
        print(f"{est:9s}  {setting:16s}  "
              f"{m:+.3e} ± {me:.1e}   {c:+.3e} ± {ce:.1e}   [{use}]")

    print("\nReading the table:")
    for est in ("ngmix", "ShearNet"):
        u = results.get((est, "uncorrected"))
        k = results.get((est, "R_psf-corrected"))
        if u and k:
            dm = k[0] - u[0]
            dc_abs = abs(k[1]) - abs(u[1])
            verdict = ("|c| DECREASED -> correction removed a real leakage"
                       if dc_abs < 0 else
                       "|c| INCREASED -> correction ADDED bias (R_psf is not the leakage)")
            print(f"  {est:9s}: Δm = {dm:+.2e} (should be ~0);  "
                  f"|c|: {abs(u[1]):.2e} -> {abs(k[1]):.2e}  =>  {verdict}")


if __name__ == "__main__":
    main()
