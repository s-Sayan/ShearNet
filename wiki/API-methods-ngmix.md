# `shearnet.methods.ngmix`

> Module: `shearnet.methods.ngmix`

NGmix-based shear estimation and metacalibration utilities for ShearNet.

<a id="shearnet.methods.ngmix.clean_and_report_nans"></a>

#### `clean_and_report_nans`

```python
def clean_and_report_nans(data_list, name)
```

Return ``data_list`` as an array with NaNs removed, printing how many were dropped.

<a id="shearnet.methods.ngmix.fourier_transform"></a>

#### `fourier_transform`

```python
def fourier_transform(psf)
```

Return the centered (fftshifted) 2-D Fourier transform of an image.

<a id="shearnet.methods.ngmix.inverse_fourier_transform"></a>

#### `inverse_fourier_transform`

```python
def inverse_fourier_transform(ft_psf_shifted)
```

Invert `fourier_transform`, returning the real-space magnitude image.

<a id="shearnet.methods.ngmix.fft_ifft"></a>

#### `fft_ifft`

```python
def fft_ifft(psf)
```

Round-trip an image through FFT and inverse FFT (a numerical sanity check).

<a id="shearnet.methods.ngmix.convolve2d"></a>

#### `convolve2d`

```python
def convolve2d(image, psf, mode="same", boundary="wrap")
```

Convolve an image with a PSF using 2D FFT-based convolution.

Always returns an image of the same dimensions as the input.

**Arguments**:

- `image` _np.ndarray_ - The input image.
- `psf` _np.ndarray_ - The PSF.
- `mode` _str_ - Not used in FFT implementation, kept for API consistency.
- `boundary` _str_ - Not used in FFT implementation, kept for API consistency.
  

**Returns**:

- `np.ndarray` - The convolved image with same dimensions as input.

<a id="shearnet.methods.ngmix.sample_half_gaussian"></a>

#### `sample_half_gaussian`

```python
def sample_half_gaussian(size=1, sigma=0.5, seed=None)
```

Generate samples from a half-Gaussian distribution with given sigma.

Ensures that all values are strictly greater than zero.

**Arguments**:

- `size` _int_ - Number of samples to generate.
- `sigma` _float_ - Standard deviation of the full Gaussian.
- `seed` _int, optional_ - Seed for reproducibility.
  

**Returns**:

- `np.ndarray` - Array of sampled values.

<a id="shearnet.methods.ngmix.g1_g2_sigma_sample"></a>

#### `g1_g2_sigma_sample`

```python
def g1_g2_sigma_sample(num_samples=10000,
                       g_std=0.26,
                       sigma_std=0.5,
                       seed=None)
```

Generate samples for g1, g2, and sigma.

g1 and g2 are sampled from a Gaussian distribution with specified standard
deviation, and sigma from a half-Gaussian distribution.

**Arguments**:

- `num_samples` _int_ - Number of samples to generate.
- `g_std` _float_ - Standard deviation for the Gaussian distribution of g1 and g2. Default is 0.26.
- `sigma_std` _float_ - Standard deviation for the half-Gaussian distribution of
  sigma. Default is 0.5.
- `seed` _int, optional_ - Seed for reproducibility.
  

**Returns**:

- `tuple` - Arrays of sampled g1, g2, and sigma values.

<a id="shearnet.methods.ngmix.make_struct"></a>

#### `make_struct`

```python
def make_struct(res, obs, shear_type)
```

Make the data structure.

Parameters
----------
res: dict
    With keys 's2n', 'e', 'T', and 'g_cov'
obs: ngmix.Observation
    The observation for this shear type
shear_type: str
    The shear type

Returns
-------
1-element array with fields

<a id="shearnet.methods.ngmix.get_em_ngauss"></a>

#### `get_em_ngauss`

```python
def get_em_ngauss(name)
```

Parse the Gaussian count from an EM model name, e.g. ``'em3'`` -> ``3``.

<a id="shearnet.methods.ngmix.get_coellip_ngauss"></a>

#### `get_coellip_ngauss`

```python
def get_coellip_ngauss(name)
```

Parse the Gaussian count from a coelliptical model name, e.g. ``'coellip3'`` -> ``3``.

<a id="shearnet.methods.ngmix.process_obs"></a>

#### `process_obs`

```python
def process_obs(obs, boot)
```

Run a metacal bootstrapper on one observation; return its struct and obs dict.

<a id="shearnet.methods.ngmix.mp_fit_one"></a>

#### `mp_fit_one`

```python
def mp_fit_one(obslist,
               prior,
               rng,
               psf_model="gauss",
               gal_model="gauss",
               mcal_pars={
                   "psf": "dilate",
                   "mcal_shear": 0.01
               },
               weight_fwhm=1.0)
```

Run metacalibration on an object (multiprocessing version of ``_fit_one``).

Returns the unsheared ellipticities of each galaxy, as well as entries for
each shear step.

inputs:
- obslist: Observation list for MEDS object of given ID
- prior: ngmix mcal priors
- mcal_pars: mcal running parameters
- weight_fwhm is just for gaussmom psf and gal models

TO DO: add a label indicating whether the galaxy passed the selection
cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).

<a id="shearnet.methods.ngmix.ngmix_pred"></a>

#### `ngmix_pred`

```python
def ngmix_pred(data_list, return_bad_indices=False)
```

Extract ``(g1, g2, sigma, flux)`` predictions from NGmix metacal results.

Reads the unsheared (``noshear``) entry of each galaxy's result struct and
converts the moment size ``T`` to ``sigma = sqrt(T/2)``.

**Arguments**:

- `data_list` - Per-galaxy metacal structs as produced by `mp_fit_one`.
- `return_bad_indices` - If ``True``, also return the indices with invalid
  (non-positive or NaN) ``T``.
  

**Returns**:

  np.ndarray of shape ``(N, 4)``, or ``(preds, bad_indices)`` when requested.

<a id="shearnet.methods.ngmix.mp_fit_one_single"></a>

#### `mp_fit_one_single`

```python
def mp_fit_one_single(obslist,
                      prior,
                      rng,
                      psf_model="gauss",
                      gal_model="gauss",
                      mcal_pars={
                          "psf": "dilate",
                          "mcal_shear": 0.01
                      })
```

Run metacalibration on an object (multiprocessing version of ``_fit_one``).

Returns the unsheared ellipticities of each galaxy, as well as entries for
each shear step.

inputs:
- obslist: Observation list for MEDS object of given ID
- prior: ngmix mcal priors
- mcal_pars: mcal running parameters

TO DO: add a label indicating whether the galaxy passed the selection
cuts for each shear step (i.e. no_shear,1p,1m,2p,2m).

<a id="shearnet.methods.ngmix.get_memory_usage"></a>

#### `get_memory_usage`

```python
def get_memory_usage(obj)
```

Print the memory usage (MB) of each attribute of an object, and the total.

<a id="shearnet.methods.ngmix.response_calculation"></a>

#### `response_calculation`

```python
def response_calculation(data_list, mcal_shear)
```

Compute per-galaxy metacalibration response and PSF-leakage terms.

From the sheared (``1p/1m/2p/2m``) entries of each galaxy's metacal struct,
finite-differences the shear response matrix elements (R11, R22, R12, R21) and
the additive/PSF terms, using a step of ``mcal_shear``.

**Returns**:

  tuple of eight lists: ``(r11, r22, r12, r21, c1, c2, c1_psf, c2_psf)``.
