"""Is the measured PSF leakage really ellipticity leakage, or PSF-size response?

``alpha`` is the regression slope of the mean measured shape against the PSF
ellipticity. That is a clean leakage measurement only if ``e^PSF`` is
uncorrelated with everything else about the PSF. It is nearly true when the
training and evaluation PSFs come from ONE PSFEx model, where the PSF varies
with field position and little else. It is much less true across a DIRECTORY of
models, which differ in size and higher moments too -- and any dependence of the
estimator on those shows up inside ``alpha`` wearing ellipticity's clothes.

Measured on the 50-file SuperBIT set, HSM moments on the rendered PSF stamps::

                                one model      50 models
    fractional spread in T         0.101          0.241
    corr(e1, T)                   -0.117         -0.273
    dT/de1  [arcsec^2]            -0.0277        -0.1320     <- 4.8x steeper
    corr(e2, T)                   +0.061         +0.141
    dT/de2  [arcsec^2]            +0.0193        +0.0672     <- 3.5x steeper

So a fixed amount of PSF-size sensitivity buys ~5x more apparent ``alpha`` on
the directory than on the single model. Comparing ``alpha`` across that change
is not comparing like with like.

This script separates the two on an existing ``evaluation.fits`` by fitting

    <e_a> = c + alpha * e^PSF_a + beta * (T^PSF - <T^PSF>)

so ``alpha`` is the leakage at fixed PSF size and ``beta`` is the size response.
If ``alpha`` falls substantially when ``beta`` is free, the single-parameter
number was absorbing size dependence.

Usage::

    python leakage_vs_size.py /path/to/evaluation.fits [--njac 30]
"""

from __future__ import annotations

import argparse

import numpy as np


def _jackknife(fit, n, njac):
    """Delete-one-block jackknife errors on whatever ``fit(index)`` returns."""
    chunks = np.array_split(np.arange(n), njac)
    samples = np.asarray(
        [fit(np.setdiff1d(np.arange(n), chunk)) for chunk in chunks]
    )
    return np.sqrt((njac - 1) * np.mean((samples - samples.mean(axis=0)) ** 2, axis=0))


def _fit(design, values, index=None):
    index = slice(None) if index is None else index
    return np.linalg.lstsq(design[index], values[index], rcond=None)[0]


def analyse(path, njac=30):
    from astropy.io import fits

    with fits.open(path) as hdul:
        leak = hdul["LEAKAGE"].data
        names = set(leak.names)
        gpsf = np.asarray(leak["gpsf"], float)
        tpsf = np.asarray(leak["Tpsf"], float)
        estimators = sorted(
            n[2:-4] for n in names if n.startswith("e_") and n.endswith("_raw")
        ) or sorted(
            n[2:] for n in names
            if n.startswith("e_") and not n.endswith("_raw") and f"R{n[1:]}" not in names
        )

        print(f"PSF population in this file: n = {len(tpsf)}")
        print(f"  T^PSF   mean {tpsf.mean():.4f}  fractional spread "
              f"{tpsf.std() / tpsf.mean():.3f}")
        for a in (0, 1):
            print(f"  corr(e{a + 1}^PSF, T^PSF) = "
                  f"{np.corrcoef(gpsf[:, a], tpsf)[0, 1]:+.3f}")

        centred_t = tpsf - tpsf.mean()
        for estimator in estimators:
            column = f"e_{estimator}_raw" if f"e_{estimator}_raw" in names else f"e_{estimator}"
            e = np.asarray(leak[column], float)
            print(f"\n--- {estimator}   (column {column}) ---")
            print(f"{'':<4}{'alpha alone':>26}{'alpha | size held':>26}{'beta (size)':>22}")
            for a in (0, 1):
                good = np.isfinite(e[:, a]) & np.isfinite(gpsf[:, a]) & np.isfinite(tpsf)
                y, x, t = e[good, a], gpsf[good, a], centred_t[good]
                n = y.size
                if n < njac + 2:
                    print(f"  e{a + 1}: too few rows")
                    continue

                simple = np.column_stack([np.ones(n), x])
                joint = np.column_stack([np.ones(n), x, t])
                a_only = _fit(simple, y)[1]
                a_err = _jackknife(lambda i: _fit(simple, y, i)[1:2], n, njac)[0]
                pars = _fit(joint, y)
                joint_err = _jackknife(lambda i: _fit(joint, y, i)[1:], n, njac)

                print(f"  e{a + 1}: {a_only:>+12.3e} +/- {a_err:<9.1e}"
                      f"{pars[1]:>+13.3e} +/- {joint_err[0]:<9.1e}"
                      f"{pars[2]:>+11.3e} +/- {joint_err[1]:<8.1e}")
                if abs(a_only) > 3 * a_err and abs(pars[1]) < abs(a_only) / 2:
                    print(f"        -> alpha halves once PSF size is controlled for; "
                          f"the single-parameter number was absorbing size response")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PSF leakage vs PSF size")
    args.add_argument("fits")
    args.add_argument("--njac", type=int, default=30)
    parsed = args.parse_args()
    analyse(parsed.fits, njac=parsed.njac)
