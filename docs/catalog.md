# The evaluation catalog

Each evaluator writes its estimator's raw measurements, shared truth and stamp
properties to a standalone FITS file. `shearnet-merge` joins the two complete
catalogs after checking that both jobs observed the same things.

Schema 2 keeps the scientific columns and adds SHA-256 object and observation
UIDs plus fingerprints of the original and metacal images. It is the current
format; readers must use schema 2. Its random streams differ from schema 1, so
an old simulation realization should not be compared byte for byte with a new
one.

## Rows

Every per-record table (`TRUTH`, `STAMP`, `SHEARNET`, `NGMIX`) has one row per
**catalog object x scene x ring station**, in the same order:

    record_id = (scene_id * n_rotations + rotation_id) * n_objects + catalog_row

The tables line up row for row. The merge verifies `observation_uid`, the
shared truth and the image fingerprints; a row number alone is not proof of a
match.

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
| `PRIMARY` | header: `SCHEMA`/`SCHEMAV`, `RUNNAME`, `EVALNAME`, `OBSSHA`, `IMPLSHA`, `CKPTSHA` (ShearNet model), `CKPTEPO`, `NOBJ`, `NSCENE`, `NROT`, `NRECORD`, `BACKEND`, `PSFMODE`, `X64`, `MCALSTEP`, `MCALPSF` |
| `TRUTH` | what each stamp was drawn from |
| `STAMP` | PSF moments and stamp-level observables |
| `SHEARNET` | ShearNet's predictions (if `shearnet` was measured) |
| `NGMIX` | ngmix's fits (if `ngmix` was measured) |
| `SCHEMA` | every column: dtype, shape, unit, meaning |
| `SCENES`, `ROTATIONS` | the scenes and ring stations |
| `BLOCKS` | per scene and station: record coverage and worker count |
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
  with `sigma = det(M)^(1/4)`;
  `STAMP.psf_T_admom` is the adaptive-moment trace `Irr + Icc`. `NGMIX.Tpsf_<t>`
  is the T of ngmix's configured model fit to the PSF of that variant -- for metacal
  products the dilated reconvolution PSF -- which is the `Tpsf_noshear`
  SuperBIT's `T/Tpsf` cut divides by.

## Derived measurements

Select populations by scene (`g1_plus`, `g1_minus`, `g2_plus`, `g2_minus`,
`zero`) and ring stations by `rotation_deg`. Ring averages are means over
stations of each `catalog_row`.

For either `NGMIX` or `SHEARNET`, use `g_noshear` where all nine metacal flags
are zero. Compute the shear response as
`R[:, a, 0] = (g_1p - g_1m)[:, a] / (2 step)` and
`R[:, a, 1] = (g_2p - g_2m)[:, a] / (2 step)`, setting it to NaN where any
flag is set. Compute the PSF response with `_1p_psf`, `_1m_psf`, `_2p_psf`,
and `_2m_psf` instead. ShearNet flags non-finite predictions.
Summary and binned statistics are derived from these measurements, not stored.

**Metacal m and c:** for each +/- pair and each station
separately, keep the objects good in both signs (all flags 0, finite `Rpsf` and
`psf_g`); `Rbar_psf` = mean over them of `(Rpsf_plus + Rpsf_minus) / 2`; the
corrected shape is `g_noshear - Rbar_psf @ psf_g` for both signs. Average the
corrected shapes and `Rgamma` over the stations (an object failed at one
station is failed), then form the paired estimator `<(e+ - e-)/2> / <R>`.
**Rendering and the merge.** Stamps are rendered in float64 on JAX's default
device -- the GPU in the ShearNet job, the CPU in the ngmix job -- and cast to
float32. The two devices agree to roundoff, not bit for bit, so `STAMP` records a
fingerprint of every image both jobs measured -- the original galaxy and PSF
stamps and the galaxy and PSF image of each metacal product: `sum |x|` and the
image's projections on three fixed random +-1 patterns. Roundoff moves them by
~1e-7 of `sum |x|`; a different galaxy, noise field or PSF, or the same image
rotated, flipped or shifted, moves the projections hundreds of times the merge's
tolerance. The merge requires every shared `TRUTH` and `STAMP` value to agree,
each on its own: integers and identifiers exactly, floats to 1e-4 of their own
size, fingerprints to 1e-4 of their image's `sum |x|`, and `flux_stamp` (which
can sit near zero) to 1e-4 of its stamp's `sum |x|`, each with an absolute floor
of 1e-12 for values at zero (a round PSF's `psf_g` differs by ~1e-16 there). The merged file takes
`TRUTH` and `STAMP` from the ngmix job and copies both measurement tables
unchanged.

**Pairing in metacal.** Every scene and station of a galaxy uses the same
metacal fixnoise field. Schema 1 drew a separate one per scene.

**PSF fits.** The PSF is the same in every scene and station, and metacal makes
only six distinct PSF stamps from it (the original, the dilated reconvolution
PSF shared by `noshear` and the four galaxy-shear products, and the four sheared
ones). ngmix fits each of the six once per galaxy, seeded from the stamp, so the
answer is the one a refit would give; the 200 PSF fits of five scenes x four
stations x ten measurements become six.

**Timing.** `NGMIX.fit_cpu_seconds` and `NGMIX.metacal_cpu_seconds` are measured
CPU seconds on one core for one observation; `*_psf_cpu_seconds` is the PSF part.
A PSF fit shared by several observations of a galaxy is charged to the one that
ran it, so the measured columns sum to what the job spent
(`TIMING.ngmix_psf_cpu_seconds` over `TIMING.ngmix_psf_fits` fits). Separately,
`*_psf_uncached_cpu_seconds` is an estimate: the cost of every PSF fit an
observation used, counted where it is used, as a pipeline that refits each PSF
(SuperBIT's) would pay. Its uncached cost is
`*_cpu_seconds - *_psf_cpu_seconds + *_psf_uncached_cpu_seconds`.

**Failures.** Any measurement that raises -- metacal, a PSF fit, the size guess,
the galaxy fit -- is a flagged row (`flags = 2**30`, NaN values), logged with its
error and counted in `TIMING.failures_raised`; the job never stops for one stamp.

The `TIMING` table
also gives ShearNet's inference wall seconds and image count
(`shearnet_inference_seconds`, `shearnet_images`), compilation kept separate
(`shearnet_compile_seconds`), and rendering (`render_seconds`), each summed over
the committed work of every attempt. `BLOCKS` records coverage and the worker
count.

## Columns

<!-- generated:start -->

### TRUTH

| column | type | unit | meaning |
|---|---|---|---|
| `object_uid` | S64 |  | SHA-256 identity of the galaxy realization, the same in every scene and station |
| `observation_uid` | S64 |  | SHA-256 identity of this observation: the object plus its applied shear and rotation |
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
| `object_uid` | S64 |  | SHA-256 identity of the galaxy realization, the same in every scene and station |
| `observation_uid` | S64 |  | SHA-256 identity of this observation: the object plus its applied shear and rotation |
| `record_id` | i8 |  | unique row id: (scene_id * n_rotations + rotation_id) * n_objects + catalog_row |
| `catalog_row` | i8 |  | row of the evaluation catalog this object is drawn from (0-based); the same galaxy in every scene and station |
| `scene_id` | i2 |  | index into the SCENES table |
| `rotation_id` | i2 |  | index into the ROTATIONS table |
| `input_fingerprint` | f4 2x4 |  | sum |x| and three fixed random projections of the galaxy stamp and of the PSF stamp; the merge checks both jobs saw the same images |
| `metacal_fingerprint` | f4 9x2x4 |  | the same for the galaxy and PSF image of each metacal product, in METACAL_TYPES order; NaN where ngmix made none |
| `psf_g` | f8 2 |  | ngmix adaptive-moment ellipticity of the PSF stamp, epsilon convention |
| `psf_T_hsm` | f8 | arcsec2 | GalSim HSM 2 sigma^2 of the PSF stamp, sigma = det(M)^(1/4): a determinant size |
| `psf_T_admom` | f8 | arcsec2 | ngmix adaptive-moment trace Irr + Icc of the PSF stamp |
| `psf_flags` | i4 |  | non-zero where the PSF moments failed |
| `flux_stamp` | f8 | count | sum of the noisy galaxy stamp |
| `s2n_stamp` | f8 |  | sqrt(sum I^2) / noise_sigma on the noisy stamp; not ngmix s2n |

### SHEARNET

| column | type | unit | meaning |
|---|---|---|---|
| `object_uid` | S64 |  | SHA-256 identity of the galaxy realization, the same in every scene and station |
| `observation_uid` | S64 |  | SHA-256 identity of this observation: the object plus its applied shear and rotation |
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
| `object_uid` | S64 |  | SHA-256 identity of the galaxy realization, the same in every scene and station |
| `observation_uid` | S64 |  | SHA-256 identity of this observation: the object plus its applied shear and rotation |
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
| `fit_cpu_seconds` | f8 | s | measured CPU seconds of the plain fit of the original stamp (T guess, galaxy fit, and the PSF fit if this observation ran it), on one core |
| `fit_psf_cpu_seconds` | f8 | s | measured: of fit_cpu_seconds, the PSF fit; zero where the fit was shared from another observation of the galaxy |
| `fit_psf_uncached_cpu_seconds` | f8 | s | estimate: CPU seconds of the PSF fit this observation used, whichever observation ran it |
| `metacal_cpu_seconds` | f8 | s | measured CPU seconds of this observation's metacal (making the nine products, their galaxy fits, and the PSF fits it ran), on one core; once per observation, not per product |
| `metacal_psf_cpu_seconds` | f8 | s | measured: of metacal_cpu_seconds, the PSF fits this observation ran |
| `metacal_image_cpu_seconds` | f8 | s | measured: of metacal_cpu_seconds, making the nine products, which a ShearNet metacal measurement also pays |
| `metacal_psf_uncached_cpu_seconds` | f8 | s | estimate: CPU seconds of all nine PSF fits the products used, each counted where it is used, as a pipeline that refits every PSF would pay |

Every `*_original` column repeats for each metacal product with the suffix `_noshear`, `_1p`, `_1m`, `_2p`, `_2m`, `_1p_psf`, `_1m_psf`, `_2p_psf`, `_2m_psf`.

<!-- generated:end -->
