# `shearnet.methods.mcal`

> Module: `shearnet.methods.mcal`

Moment-based metacalibration shear estimation for ShearNet.

<a id="shearnet.methods.mcal.measure_moments"></a>

#### `measure_moments`

```python
def measure_moments(im)
```

Measure second-order moments for a GalSim image object or a NumPy array.

**Arguments**:

- `im` - Either a GalSim image object or a 2D NumPy array.
  

**Returns**:

  M_xx, M_yy, M_xy: Second-order moments.

<a id="shearnet.methods.mcal.measure_e1e2"></a>

#### `measure_e1e2`

```python
def measure_e1e2(g1=None, g2=None, im=None, sigma=1, scale=0.2, npix=53)
```

Calculate ellipticity components e1 and e2.

**Arguments**:

  g1, g2: Shear components. The image-rendering path for these is not
  implemented; pass ``im`` instead.
- `im` - 2D image array. If provided, moments are computed via
  `measure_moments`.
- `sigma` - Standard deviation for Gaussian smoothing (used with g1, g2).
- `scale` - Pixel scale (used with g1, g2).
- `npix` - Number of pixels along one dimension (used with g1, g2).
  

**Returns**:

  e1, e2: Ellipticity components.

<a id="shearnet.methods.mcal.calculate_responsivity"></a>

#### `calculate_responsivity`

```python
def calculate_responsivity(psf_sigma, seed, h=0.01)
```

Estimate the shear responsivity (R1, R2) by finite differences.

Simulates galaxies sheared by +/- ``h`` in each component and measures how the
recovered ellipticity changes, giving the multiplicative response used to
de-bias moment-based shear estimates.

**Arguments**:

- `psf_sigma` - Gaussian PSF FWHM (arcsec) passed to `sim_func`.
- `seed` - Random seed for the simulated galaxies.
- `h` - Finite-difference step in shear.
  

**Returns**:

- `tuple` - ``(R1, R2)`` responsivities for g1 and g2.

<a id="shearnet.methods.mcal.obs_g1g2"></a>

#### `obs_g1g2`

```python
def obs_g1g2(im, psf_sigma)
```

Calculate the observed g1 and g2 values for a given image and PSF FWHM.

**Arguments**:

- `im` _numpy.ndarray_ - The input image.
- `psf_sigma` _float_ - The Sigma of the PSF.
  

**Returns**:

- `tuple` - A tuple containing the observed g1 and g2 values.

<a id="shearnet.methods.mcal.mcal_preds"></a>

#### `mcal_preds`

```python
def mcal_preds(images, psf_sigma)
```

Calculate observed g1 and g2 for a list of images and a given PSF FWHM.

**Arguments**:

- `images` _list_ - A list of input images.
- `psf_sigma` _float_ - The Sigma of the PSF.
  

**Returns**:

- `tuple` - Two lists, the observed g1 values and the observed g2 values.
