import galsim
import numpy as np

def create_wcs_from_params(params):
    """
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
    """
    # Extract parameters
    xsize = params["image_xsize"]
    ysize = params["image_ysize"]
    pixel_scale = params["pixel_scale"]
    center_ra = params["center_ra"] * galsim.degrees
    center_dec = params["center_dec"] * galsim.degrees
    theta = params.get("theta", 0.0) * galsim.degrees

    # Create fiducial image for coordinate origin
    fiducial_full_image = galsim.ImageF(xsize, ysize)

    # Build affine transform
    dudx = np.cos(theta) * pixel_scale
    dudy = -np.sin(theta) * pixel_scale
    dvdx = np.sin(theta) * pixel_scale
    dvdy = np.cos(theta) * pixel_scale

    affine = galsim.AffineTransform(
        dudx, dudy, dvdx, dvdy,
        origin=fiducial_full_image.true_center
    )

    # Create sky center and WCS
    sky_center = galsim.CelestialCoord(ra=center_ra, dec=center_dec)
    wcs = galsim.TanWCS(affine, sky_center, units=galsim.arcsec)

    return wcs