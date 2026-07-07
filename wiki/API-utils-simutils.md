# `shearnet.utils.simutils`

> Module: `shearnet.utils.simutils`

GalSim WCS construction helpers for ShearNet simulations.

<a id="shearnet.utils.simutils.create_wcs_from_params"></a>

#### `create_wcs_from_params`

```python
def create_wcs_from_params(params)
```

Create a GalSim TanWCS from a dictionary of image and pointing parameters.

Parameters
----------
params : dict
    Dictionary with keys:
        - image_xsize : int
        - image_ysize : int
        - pixel_scale : float  (in arcsec/pixel)
        - center_ra   : float  (in degrees)
        - center_dec  : float  (in degrees)
        - theta       : float, optional (rotation angle in degrees, default=0.0)

Returns
-------
galsim.TanWCS
    World Coordinate System object.
