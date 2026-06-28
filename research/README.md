# Research record

This directory holds experiment configurations, scripts, and notes from the
development of ShearNet. **None of it is needed to install or use the package** —
it is kept for reproducibility and provenance.

| Directory | Contents |
|---|---|
| `unit_tests/` | Numbered experiment runs (`first`, `second`, …) with their configs and submit scripts. |
| `unit_test_variations/` | Variations on those runs (different PSFs, architectures, catalogs, weightings). |
| `test/` | Early SuperBIT-PSF and fusion-transformer trials. |
| `shear_bias/` | Scripts and configs for shear-bias (`m`) and PSF-leakage studies. |

These files may reference absolute paths and cluster-specific submit scripts from
the original author's environment; treat them as a historical record rather than
runnable examples. For current, supported usage see the top-level `README.md`.
