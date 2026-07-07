# `shearnet.core.dataset`

> Module: `shearnet.core.dataset`

Galaxy/PSF postage-stamp simulation and dataset generation for ShearNet.

<a id="shearnet.core.dataset.MARGIN"></a>

#### `MARGIN`

Margins that I wanna use for PSF Rendering

<a id="shearnet.core.dataset.generate_dataset"></a>

#### `generate_dataset`

```python
def generate_dataset(samples,
                     psf_fwhm,
                     npix=53,
                     scale=0.141,
                     type="exp",
                     exp="ideal",
                     nse_sd=1e-5,
                     seed=42,
                     return_clean=False,
                     return_psf=False,
                     return_obs=False,
                     apply_psf_shear=False,
                     psf_shear_range=0.05,
                     base_shear_g1=0.0,
                     base_shear_g2=0.0,
                     psf_file_or_dir=PSF_DATA_DIR,
                     output_keys=("g1", "g2"),
                     hlr_type="constant",
                     flux_type="constant",
                     cosmos_cat_fname=None)
```

Simulate a dataset of galaxy postage stamps with known shear labels.

Each sample is a GalSim galaxy (exponential or Gaussian) sheared by values
drawn from a COSMOS catalog, convolved with either an analytic Gaussian PSF
(``exp='ideal'``) or an empirical PSFEx SuperBIT PSF (``exp='superbit'``),
and corrupted with Gaussian noise. See `sim_func` for the per-object
simulation details.

**Arguments**:

- `samples` - Number of postage stamps to generate.
- `psf_fwhm` - FWHM (arcsec) of the analytic Gaussian PSF (``exp='ideal'``).
- `npix` - Stamp size in pixels (square).
- `scale` - Pixel scale in arcsec/pixel.
- `type` - Galaxy light profile, ``'exp'`` or ``'gauss'``.
- `exp` - Experiment / PSF mode, ``'ideal'`` or ``'superbit'``.
- `nse_sd` - Standard deviation of the additive Gaussian noise.
- `seed` - Base random seed for reproducibility.
- `return_clean` - Also stack the noise-free galaxy image as an extra channel.
- `return_psf` - Also stack the PSF image as an extra channel.
- `return_obs` - Additionally return the list of ngmix ``Observation`` objects.
- `apply_psf_shear` - Apply a random shear to the PSF (``exp='ideal'`` only).
- `psf_shear_range` - Half-width of the uniform PSF-shear distribution.
  base_shear_g1, base_shear_g2: Constant shear applied to every galaxy.
- `psf_file_or_dir` - PSF file or directory of ``.psf`` files for the
  ``superbit`` mode (defaults to `PSF_DATA_DIR`).
- `output_keys` - Label fields to return per sample; subset of
  ``{"g1", "g2", "hlr", "flux", "psf_e1", "psf_e2", "psf_T"}``.
- `hlr_type` - ``'constant'`` (0.5) or ``'catalog'`` half-light radius.
- `flux_type` - ``'constant'`` or ``'catalog'`` flux.
- `cosmos_cat_fname` - Path to the COSMOS catalog FITS file; a synthetic
  random catalog is used as a fallback (e.g. in CI) when absent.
  

**Returns**:

  ``(images, labels)`` as numpy arrays, or ``(images, labels, obs)`` when
  ``return_obs=True``. ``images`` is ``(samples, npix, npix)`` for a single
  channel, or has a trailing channel axis when ``return_psf``/``return_clean``
  are set. ``labels`` has shape ``(samples, len(output_keys))``.

<a id="shearnet.core.dataset.split_combined_images"></a>

#### `split_combined_images`

```python
def split_combined_images(combined_images, has_psf=False, has_clean=False)
```

Split concatenated images back into separate arrays.

**Arguments**:

- `combined_images` - np.ndarray of shape (samples, height, width, 2 or 3)
- `has_psf` - bool, whether PSF images are included
- `has_clean` - bool, whether clean images are included
  

**Returns**:

  Tuple of arrays depending on combination:
  - If has_psf=True, has_clean=True: (galaxy, psf, clean)
  - If has_psf=True, has_clean=False: (galaxy, psf)
  - If has_psf=False, has_clean=True: (galaxy, clean)

<a id="shearnet.core.dataset.sim_func"></a>

#### `sim_func`

```python
def sim_func(g1,
             g2,
             hlr=1.0,
             flux=1.0,
             psf_fwhm=0.5,
             nse_sd=1e-5,
             type="gauss",
             npix=53,
             scale=0.141,
             seed=42,
             exp="ideal",
             apply_psf_shear=False,
             psf_shear_range=0.05,
             ud=None,
             psf_files=None,
             base_shear_g1=0.0,
             base_shear_g2=0.0)
```

Simulate a single galaxy observation and return an ngmix ``Observation``.

Builds a sheared, randomly-shifted galaxy, convolves it with the chosen PSF,
draws the noisy image, fits the PSF adaptive moments, and packages everything
into an ngmix ``Observation``. The metadata also stores the clean image and
the +/- e1/e2 sheared counterparts used for metacalibration-style responses.

**Arguments**:

  g1, g2: Intrinsic shear of the galaxy.
- `hlr` - Half-light radius (arcsec).
- `flux` - Total flux of the galaxy.
- `psf_fwhm` - Gaussian PSF FWHM (arcsec) for ``exp='ideal'``.
- `nse_sd` - Standard deviation of the additive Gaussian noise.
- `type` - Galaxy profile, ``'exp'`` or ``'gauss'``.
- `npix` - Stamp size in pixels (square).
- `scale` - Pixel scale in arcsec/pixel.
- `seed` - Random seed for this object's noise and shifts.
- `exp` - PSF mode, ``'ideal'`` (analytic Gaussian) or ``'superbit'`` (PSFEx).
- `apply_psf_shear` - Apply a random shear to the analytic PSF.
- `psf_shear_range` - Half-width of the uniform PSF-shear distribution.
- `ud` - A ``galsim.UniformDeviate`` used to sample the SuperBIT PSF.
- `psf_files` - List of ``.psf`` files to draw from for ``exp='superbit'``.
  base_shear_g1, base_shear_g2: Constant shear applied before the PSF.
  

**Returns**:

- `ngmix.Observation` - The noisy galaxy observation, with its ``psf`` set and
  ``meta`` populated with ``snr``, ``clean_image`` and the metacal images.

<a id="shearnet.core.dataset.search_psf_files"></a>

#### `search_psf_files`

```python
def search_psf_files(path)
```

Return a list of all ``*.psf`` files directly under ``path``.

<a id="shearnet.core.dataset.get_background_file"></a>

#### `get_background_file`

```python
def get_background_file(psf_file)
```

Map a PSFEx ``.psf`` file path to its matching sky-background FITS file.

Replaces the ``_starcat.psf`` suffix with ``.bkg_rms.fits`` and swaps the
``psfex-output`` directory for the sibling ``sky_backgrounds`` directory.

<a id="shearnet.core.dataset.import_psf"></a>

#### `import_psf`

```python
def import_psf(psf_files,
               ud,
               xsize=WCS_PARAMS["image_xsize"],
               ysize=WCS_PARAMS["image_ysize"],
               margin=MARGIN)
```

Sample an empirical PSFEx PSF at a random image position.

Picks a random exposure from ``psf_files`` and a random position within the
image (inset by ``margin``), then evaluates the PSFEx model there using the
WCS built from `WCS_PARAMS`.

**Arguments**:

- `psf_files` - List of PSFEx ``.psf`` files to choose from.
- `ud` - A ``galsim.UniformDeviate`` providing the random draws.
  xsize, ysize: Image dimensions (pixels) used to sample the position.
- `margin` - Pixel margin kept clear of the image edges.
  

**Returns**:

  A ``galsim`` PSF object evaluated at the sampled position.
