import numpy as np
from shearnet.dataset import sim_func

def measure_moments(im):
    """
    Measures moments for a GalSim image object or a NumPy array.
    
    Args:
        - im: Either a GalSim image object or a 2D NumPy array.
        
    Returns:
        - M_xx, M_yy, M_xy: Second-order moments.
    """
    if hasattr(im, 'bounds') and hasattr(im, 'array'):
        # Handle GalSim image object
        xmin, xmax, ymin, ymax = im.bounds.xmin, im.bounds.xmax, im.bounds.ymin, im.bounds.ymax
        x, y = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
        I = im.array
    elif isinstance(im, np.ndarray):
        # Handle NumPy array
        ny, nx = im.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        I = im
    else:
        raise TypeError("Input 'im' must be a GalSim image object or a 2D NumPy array.")

    I_sum = np.sum(I)
    M_x = np.sum(I * x) / I_sum
    M_y = np.sum(I * y) / I_sum
    M_xx = np.sum(I * (x - M_x) ** 2) / I_sum
    M_yy = np.sum(I * (y - M_y) ** 2) / I_sum
    M_xy = np.sum(I * (x - M_x) * (y - M_y)) / I_sum
    return M_xx, M_yy, M_xy

def measure_e1e2(g1=None, g2=None, im=None, sigma=1, scale=0.2, npix=53):
    """
    Calculate ellipticity components e1 and e2.
    
    Parameters:
    - g1, g2: Arrays for shear components. If provided, moments are calculated using plot_shear_image.
    - im: 2D array for image. If provided, moments are calculated using measure_moments.
    - sigma: Standard deviation for Gaussian smoothing (used with g1, g2).
    - scale: Pixel scale (used with g1, g2).
    - npix: Number of pixels along one dimension (used with g1, g2).
    
    Returns:
    - e1, e2: Ellipticity components.
    """
    if im is not None:
        M_xx, M_yy, M_xy = measure_moments(im)
    elif g1 is not None and g2 is not None:
        M_xx, M_yy, M_xy = plot_shear_image(g1, g2, sigma=sigma, nx=npix, ny=npix, scale=scale, verbose=False, return_moments=True, plot=False)
    else:
        raise ValueError("Either (g1 and g2) or im must be provided.")
    
    Mr = M_xx + M_yy
    Mplus = M_xx - M_yy
    Mcross = 2 * M_xy
    e1 = Mplus / Mr
    e2 = Mcross / Mr
    return e1, e2

def calculate_responsivity(psf_fwhm, seed, h=0.01):
    obj_im_p = sim_func(h, 0, seed=seed, psf_sigma=psf_fwhm)
    obj_im_m = sim_func(-h, 0, seed=seed, psf_sigma=psf_fwhm)
    e1p, _ = measure_e1e2(im=obj_im_p.image)
    e1m, _ = measure_e1e2(im=obj_im_m.image)
    R1 = (e1p - e1m) / (2 * h)
    obj_im_p = sim_func(0, h, seed=seed, psf_sigma=psf_fwhm)
    obj_im_m = sim_func(0, -h, seed=seed, psf_sigma=psf_fwhm)
    _, e2p = measure_e1e2(im=obj_im_p.image)
    _, e2m  = measure_e1e2(im=obj_im_m.image)
    R2 = (e2p - e2m) / (2 * h)
    return R1, R2

def obs_g1g2(im, psf_fwhm):
    """
    Calculates the observed g1 and g2 values for a given image and PSF FWHM.

    Args:
        im (numpy.ndarray): The input image.
        psf_fwhm (float): The FWHM of the PSF.

    Returns:
        tuple: A tuple containing the observed g1 and g2 values.
    """
    R1, R2 = calculate_responsivity(psf_fwhm, 1234)
    e1, e2 = measure_e1e2(im=im)
    obs_g1 = e1 / R1
    obs_g2 = e2 / R2
    
    return obs_g1, obs_g2

def mcal_preds(images, psf_fwhm):
    """
    Calculates the observed g1 and g2 values for a list of images and a given PSF FWHM.

    Args:
        images (list): A list of input images.
        psf_fwhm (float): The FWHM of the PSF.

    Returns:
        tuple: A tuple containing two lists, one for the observed g1 values and one for the observed g2 values.
    """
    preds = []
    for image in images:
        g1, g2 = obs_g1g2(image, psf_fwhm)
        g1g2 = np.array([g1, g2])
        preds.append(g1g2)
    
    return np.array(preds)