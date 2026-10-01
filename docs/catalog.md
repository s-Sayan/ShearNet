# The evaluation catalog

`shearnet-eval --run RUN` writes one FITS file,
`RUN/evaluations/<name>/<run_name>_<name>.fits`. It holds **raw measurements
only**: what ShearNet predicted and what ngmix fitted, on every stamp, plus the
truth the stamps were drawn from and everything needed to interpret them. It
holds no response, no bias, no leakage slope, no selection, no weight and no
correction -- those are computed from this file, downstream.

## Rows

Every per-record table (`TRUTH`, `STAMP`, `SHEARNET`, `NGMIX`) has one row per
**catalog object x scene x ring station**, in the same order:

    record_id = (scene_id * n_rotations + rotation_id) * n_objects + catalog_row

so the four tables line up row for row (and join on `record_id`).

* A **scene** (`SCENES`: `scene_id`, `name`, `g1`, `g2`) is one applied reduced
  shear. The default five are `zero`, `g1_plus`, `g1_minus`, `g2_plus`,
  `g2_minus` at |g| = 0.01.
* A **ring station** (`ROTATIONS`: `rotation_id`, `rotation_deg`,
  `noise_quarter_turns`) rotates every galaxy's source shape and offset; the PSF
  is not rotated.
* **Pairing.** Object `catalog_row` is the same galaxy, at the same offset,
  behind the same PSF, with the same noise in every scene; across stations the
  noise turns with the galaxy by whole quarter turns (45 and 135 reuse the noise
  of 0 and 90). Any two scenes can be differenced object by object.

A fit that failed is a row with its flag set and NaN values -- never a missing
row and never a zero.

## Extensions

| HDU | what |
|---|---|
| `PRIMARY` | header: `SCHEMA`/`SCHEMAV`, `RUNNAME`, `EVALNAME`, `CKPTSHA` (sha256 of the model), `CKPTEPO`, `NOBJ`, `NSCENE`, `NROT`, `NRECORD`, `BACKEND`, `PSFMODE`, `X64`, `MCALSTEP`, `MCALPSF` |
| `TRUTH` | what each stamp was drawn from |
| `STAMP` | PSF moments and stamp-level observables |
| `SHEARNET` | ShearNet's predictions (if `shearnet` was measured) |
| `NGMIX` | ngmix's fits (if `ngmix` was measured) |
| `SCHEMA` | every column: dtype, shape, unit, meaning |
| `SCENES`, `ROTATIONS` | the scenes and ring stations |
| `BLOCKS` | per scene x station: first record, ngmix seeds, seconds |
| `PSF_FILES` | the PSFEx files `TRUTH.psf_file_id` points into |
| `PROTOCOL` | the measurement protocol in words and numbers |
| `CONFIG` | the run's training config and this evaluation's config, as YAML |
| `PROVENANCE` | code, packages, machine, and the training manifest |

Read a table with `astropy.table.Table.read(path, hdu="NGMIX")` (or
`shearnet.io.fits_catalog.read_table`). Do not assume the first table is any
particular one.

## Conventions

* Every `g`/`e` is the reduced-shear-style ellipticity
  epsilon = (1 - q)/(1 + q) exp(2 i phi) (ngmix's `g`, GalSim's `Shear.g1/g2`),
  in image axes. `g_*` columns are what an estimator reported; `e_*` columns are
  inputs to the simulation. None of them is a per-object shear.
* `TRUTH.e_prepsf` -- the source shape composed with the applied shear -- is what
  ShearNet is trained to predict as `(g1, g2)`.
* Variants follow SuperBIT's metacal tables: `_original` is the stamp as rendered;
  `_noshear`, `_1p`, `_1m`, `_2p`, `_2m` are ngmix's deconvolve / shear /
  reconvolve products and `_1p_psf` ... `_2m_psf` shear the *dilated
  reconvolution* PSF, all with `psf: dilate` and the one-sided `step` in the
  header (`MCALSTEP`; the `p`/`m` products are `2 * step` apart). ShearNet is run
  on the very image/PSF pairs ngmix fitted.
* Two PSF sizes, both arcsec^2: `STAMP.psf_T_hsm` is GalSim HSM's `2 sigma^2`
  with `sigma = det(M)^(1/4)` (what the old catalogs called `Tpsf`);
  `STAMP.psf_T_admom` is the adaptive-moment trace `Irr + Icc`. `NGMIX.Tpsf_<t>`
  is the T of ngmix's own Gaussian fit to the PSF of that variant -- for metacal
  products the dilated reconvolution PSF -- which is the `Tpsf_noshear`
  SuperBIT's `T/Tpsf` cut divides by.

## From the old catalogs

The old `benchmarking/evaluation.fits` had one table per population (`TAB_P`,
`TAB_M` for +/-g1, `TAB_P2`, `TAB_M2` for +/-g2, `LEAKAGE` for no shear), ring
stations as `_r45`/`_r90`/`_r135` column suffixes, and derived columns. Select
the scene and station instead, and derive:

| old | new |
|---|---|
| `TAB_P`, `TAB_M`, `TAB_P2`, `TAB_M2`, `LEAKAGE` | scenes `g1_plus`, `g1_minus`, `g2_plus`, `g2_minus`, `zero` |
| `<col>_r45` etc. | rows with `rotation_deg == 45` |
| `g_th` | `TRUTH.e_prepsf` (also `label_g1`, `label_g2`) |
| `hlr_th`, `flux_th` | `TRUTH.hlr`, `TRUTH.flux_model` |
| `gpsf`, `Tpsf` | `STAMP.psf_g`, `STAMP.psf_T_hsm` |
| `s2n` | `STAMP.s2n_stamp` |
| `e_shearnet`, `e_shearnet_uncorrected`, `e_shearnet_original` | `SHEARNET.g_original` |
| `hlr_shearnet`, `flux_shearnet` | `SHEARNET.hlr_original`, `SHEARNET.flux_original` |
| `e_ngmix`, `e_ngmix_uncorrected`, `e_ngmix_original` | `NGMIX.g_original` |
| `T_ngmix`, `s2n_ngmix`, `flux_ngmix` | `NGMIX.T_noshear`, `NGMIX.s2n_noshear`, `NGMIX.flux_noshear` |
| `T_ngmix_1p`, `s2n_ngmix_1p`, ... | `NGMIX.T_1p`, `NGMIX.s2n_1p`, ... |
| `flag_ngmix` | any of the nine `NGMIX.flags_<t>` non-zero |
| `e_ngmix_metacal_raw` | `NGMIX.g_noshear` where all nine flags are 0 (NaN otherwise) |
| `Rgamma_ngmix_metacal` (`R_ngmix_metacal`) | `R[:, a, 0] = (g_1p - g_1m)[:, a] / (2 step)`, `R[:, a, 1] = (g_2p - g_2m)[:, a] / (2 step)`, NaN where any flag is set |
| `Rpsf_ngmix_metacal` | the same with `_1p_psf`, `_1m_psf`, `_2p_psf`, `_2m_psf` |
| the `shearnet` versions of the four above | the same on `SHEARNET`; ShearNet's flag is a non-finite prediction |
| `Rbarpsf_*`, `e_*_metacal`, `e_*_metacal_corrected` | derived, see below |
| `*_ring` | the mean over the stations of each `catalog_row` |
| `SUMMARY`, `BINNED`, `LEAKSUM` | not written: compute them from the above |

**The old `metacal` m and c**, exactly: for each +/- pair and each station
separately, keep the objects good in both signs (all flags 0, finite `Rpsf` and
`psf_g`); `Rbar_psf` = mean over them of `(Rpsf_plus + Rpsf_minus) / 2`; the
corrected shape is `g_noshear - Rbar_psf @ psf_g` for both signs. Average the
corrected shapes and `Rgamma` over the stations (an object failed at one
station is failed), then form the paired estimator `<(e+ - e-)/2> / <R>`. This
reproduces the old SUMMARY from the new catalog; checked on a test run, the
only differences come from the old harness rendering the rotated stations in
float64 (now float32, like the 0-degree station and the training stamps), and
are ~1e-3 of the jackknife error on m.

## Columns

<!-- generated:start -->

### TRUTH

| column | type | unit | meaning |
|---|---|---|---|
| `record_id` | i8 |  | unique row id: (scene_id * n_rotations + rotation_id) * n_objects + catalog_row |
| `catalog_row` | i8 |  | row of the evaluation catalog this object is drawn from (0-based); the same galaxy in every scene and station |
| `scene_id` | i2 |  | index into the SCENES table |
| `rotation_id` | i2 |  | index into the ROTATIONS table |
| `rotation_deg` | f8 | deg | active rotation of the source shape and offset at this station |
| `e_source` | f8 2 |  | source-model ellipticity after the station's rotation, before the applied shear (the catalog G1/G2 rotated) |
| `g_applied` | f8 2 |  | applied reduced shear of the scene (GalSim .shear(); area preserving, no magnification) |
| `e_prepsf` | f8 2 |  | ellipticity of the pre-PSF galaxy: e_source composed with g_applied. The g1/g2 the network is trained on |
| `hlr` | f8 | arcsec | half_light_radius of the circular profile before shaping (circularized; not a measured, PSF-convolved size) |
| `flux_model` | f8 | count | total flux of the profile |
| `offset` | f8 2 | arcsec | sub-pixel offset (dx, dy) of the galaxy from the stamp centre |
| `psf_shear` | f8 2 |  | artificial reduced shear applied to the PSF model (simulation.apply_psf_shear); zero otherwise |
| `psf_pos` | f8 2 | pixel | PSFEx focal-plane position (x, y) the PSF was evaluated at; zero for an ideal PSF |
| `psf_file_id` | i4 |  | index into the PSF_FILES table; -1 for an ideal Gaussian PSF |
| `q_source` | f8 |  | catalog axis ratio b/a (Q column), unrotated; NaN if the catalog has none |
| `phi_source` | f8 | rad | catalog position angle (PHI column), unrotated; NaN if the catalog has none |
| `label_g1` | f8 |  | training target g1 |
| `label_g2` | f8 |  | training target g2 |
| `label_hlr` | f8 |  | training target hlr |
| `label_flux` | f8 |  | training target flux |

### STAMP

| column | type | unit | meaning |
|---|---|---|---|
| `record_id` | i8 |  | unique row id: (scene_id * n_rotations + rotation_id) * n_objects + catalog_row |
| `catalog_row` | i8 |  | row of the evaluation catalog this object is drawn from (0-based); the same galaxy in every scene and station |
| `scene_id` | i2 |  | index into the SCENES table |
| `rotation_id` | i2 |  | index into the ROTATIONS table |
| `psf_g` | f8 2 |  | ngmix adaptive-moment ellipticity of the PSF stamp, epsilon convention |
| `psf_T_hsm` | f8 | arcsec2 | GalSim HSM 2 sigma^2 of the PSF stamp, sigma = det(M)^(1/4): a determinant size (the historical Tpsf) |
| `psf_T_admom` | f8 | arcsec2 | ngmix adaptive-moment trace Irr + Icc of the PSF stamp |
| `psf_flags` | i4 |  | non-zero where the PSF moments failed |
| `flux_stamp` | f8 | count | sum of the noisy galaxy stamp |
| `s2n_stamp` | f8 |  | sqrt(sum I^2) / noise_sigma on the noisy stamp; not ngmix s2n |

### SHEARNET

| column | type | unit | meaning |
|---|---|---|---|
| `record_id` | i8 |  | unique row id: (scene_id * n_rotations + rotation_id) * n_objects + catalog_row |
| `catalog_row` | i8 |  | row of the evaluation catalog this object is drawn from (0-based); the same galaxy in every scene and station |
| `scene_id` | i2 |  | index into the SCENES table |
| `rotation_id` | i2 |  | index into the ROTATIONS table |
| `g_original` | f8 2 |  | predicted (g1, g2) on the original stamp: the network's estimate of e_prepsf |
| `hlr_original` | f8 | arcsec | predicted hlr on the original stamp |
| `flux_original` | f8 | count | predicted flux on the original stamp |
| `flags_original` | i4 |  | non-zero where the prediction on the original stamp is not finite |

Every `*_original` column repeats for each metacal product with the suffix `_noshear`, `_1p`, `_1m`, `_2p`, `_2m`, `_1p_psf`, `_1m_psf`, `_2p_psf`, `_2m_psf`.

### NGMIX

| column | type | unit | meaning |
|---|---|---|---|
| `record_id` | i8 |  | unique row id: (scene_id * n_rotations + rotation_id) * n_objects + catalog_row |
| `catalog_row` | i8 |  | row of the evaluation catalog this object is drawn from (0-based); the same galaxy in every scene and station |
| `scene_id` | i2 |  | index into the SCENES table |
| `rotation_id` | i2 |  | index into the ROTATIONS table |
| `g_original` | f8 2 |  | fitted ellipticity, epsilon convention, on the original stamp |
| `g_cov_original` | f8 2x2 |  | covariance of g from the fit, on the original stamp |
| `T_original` | f8 | arcsec2 | fitted pre-PSF size Irr + Icc of the Gaussian model, on the original stamp |
| `Tpsf_original` | f8 | arcsec2 | T of ngmix's fit to the PSF this variant was fitted with (the dilated reconvolution PSF for metacal products), on the original stamp |
| `flux_original` | f8 | count | fitted flux, on the original stamp |
| `s2n_original` | f8 |  | ngmix s2n of the fit, on the original stamp |
| `flags_original` | i4 |  | ngmix flags; 0 is a good fit, 2**30 means no result, on the original stamp |

Every `*_original` column repeats for each metacal product with the suffix `_noshear`, `_1p`, `_1m`, `_2p`, `_2m`, `_1p_psf`, `_1m_psf`, `_2p_psf`, `_2m_psf`.

<!-- generated:end -->
