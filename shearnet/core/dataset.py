import os
from glob import glob
import numpy as np
import galsim
import ngmix
from scipy.signal import convolve2d
from tqdm import tqdm
from ..methods.ngmix import g1_g2_sigma_sample
from ..utils import create_wcs_from_params
import galsim.des

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SHEARNET_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

WCS_PARAMS = {
    "image_xsize": 9600,
    "image_ysize": 6422,
    "pixel_scale": 0.1408,     # arcsec/pixel
    "center_ra": 13.3,
    "center_dec": 33.1,
    "theta": 0.0            # optional
}

MARGIN = 200 # Margins that I wanna use for PSF Rendering

PSF_DATA_DIR = "/home/adfield/SHEARNET_DATA"

# PSF_DATA_DIR = os.path.join(SHEARNET_ROOT, "psf_data", "emp_psfs_best")

def generate_dataset(samples, psf_sigma, npix=53, scale=0.141, type='gauss', exp='ideal', nse_sd=1e-5, seed=42, return_clean=False, return_psf=False,return_obs=False,apply_psf_shear=False, psf_shear_range=0.05):
    images = []
    labels = []
    obs = []
    g1_list, g2_list, sigma_list = g1_g2_sigma_sample(num_samples=samples, seed=seed)
    ud = galsim.UniformDeviate(seed)
    if exp=="superbit":
        psf_files = search_psf_files(path=PSF_DATA_DIR)
        if len(psf_files)==0:
            raise FileNotFoundError(f"No PSF files found in {PSF_DATA_DIR}")
    else:
        psf_files = None
    for i in tqdm(range(samples)):
        g1, g2 = g1_list[i], g2_list[i]
        sigma = sigma_list[i]
        #g1, g2 = np.random.uniform(-0.5, 0.5, size=2)  # Random shears
        #sigma = np.random.uniform(0.5, 1.5)  # Random sigma  
        flux=np.random.uniform(1, 5)  # Random flux
        #psf_sigma = np.random.uniform(0.5, 1.5)
        
        obj_obs = sim_func(g1, g2, sigma=sigma, flux=flux, psf_sigma=psf_sigma, 
        nse_sd=nse_sd, type=type, npix=npix, scale=scale, seed=i, exp=exp, apply_psf_shear=apply_psf_shear, psf_shear_range=psf_shear_range, ud=ud, psf_files=psf_files)
        
        galaxy_images = obj_obs.image
        psf_images = obj_obs.psf.image
        clean_images = obj_obs.clean_image

        if return_psf and return_clean:
            # Create (height, width, 3) array: [galaxy, psf, clean]
            combined_images = np.stack([galaxy_images, psf_images, clean_images], axis=-1)
            images.append(combined_images)
        elif return_psf:
            # Create (height, width, 2) array: [galaxy, psf]
            combined_images = np.stack([galaxy_images, psf_images], axis=-1)
            images.append(combined_images)
        elif return_clean:
            # Create (height, width, 2) array: [galaxy, clean]
            combined_images = np.stack([galaxy_images, clean_images], axis=-1)
            images.append(combined_images)
        else:
            # Just galaxy images
            images.append(galaxy_images)

        labels.append(np.array([g1, g2, sigma, flux], dtype=np.float32))
        obs.append(obj_obs)
    
    if return_obs:
        return np.array(images), np.array(labels), obs    
    
    return np.array(images), np.array(labels)

def split_combined_images(combined_images, has_psf=False, has_clean=False):
    """
    Split concatenated images back into separate arrays.
    
    Args:
        combined_images: np.ndarray of shape (samples, height, width, 2 or 3)
        has_psf: bool, whether PSF images are included
        has_clean: bool, whether clean images are included
        
    Returns:
        Tuple of arrays depending on combination:
        - If has_psf=True, has_clean=True: (galaxy, psf, clean)
        - If has_psf=True, has_clean=False: (galaxy, psf)
        - If has_psf=False, has_clean=True: (galaxy, clean)
    """
    if combined_images.shape[-1] == 2:
        if has_psf and not has_clean:
            # Galaxy + PSF
            galaxy_images = combined_images[..., 0]
            psf_images = combined_images[..., 1]
            return galaxy_images, psf_images
        elif has_clean and not has_psf:
            # Galaxy + Clean
            galaxy_images = combined_images[..., 0]
            clean_images = combined_images[..., 1]
            return galaxy_images, clean_images
        else:
            raise ValueError("Invalid combination: 2 channels requires either PSF or clean images, not both")
            
    elif combined_images.shape[-1] == 3:
        if has_psf and has_clean:
            # Galaxy + PSF + Clean
            galaxy_images = combined_images[..., 0]
            psf_images = combined_images[..., 1]
            clean_images = combined_images[..., 2]
            return galaxy_images, psf_images, clean_images
        else:
            raise ValueError("3 channels requires both PSF and clean images")
    else:
        raise ValueError(f"Unexpected number of channels: {combined_images.shape[-1]}")

def sim_func(g1, g2, sigma=1.0, flux=1.0, psf_sigma=0.5, nse_sd = 1e-5,  type='gauss', npix=53, scale=0.141, seed=42, exp="ideal", apply_psf_shear=False, psf_shear_range=0.05, ud=None, psf_files=None):

    rng = np.random.RandomState(seed=seed)

    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma, flux=flux).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    # Generate PSF shear values if requested
    psf_g1, psf_g2 = 0.0, 0.0
    if apply_psf_shear:
        psf_g1 = rng.uniform(-psf_shear_range, psf_shear_range)
        psf_g2 = rng.uniform(-psf_shear_range, psf_shear_range)

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    sheared_gal = gal.shift(dx, dy)

    # Convolve with PSF
    if exp == 'ideal':
        #gsp = galsim.GSParams(maximum_fft_size=8192000)
        psf = galsim.Gaussian(sigma=psf_sigma)

        if apply_psf_shear:
            psf = psf.shear(g1=psf_g1, g2=psf_g2)

        obj = galsim.Convolve(sheared_gal, psf)
    elif exp == 'superbit':
        psf = import_psf(psf_files, ud)
        gsp=galsim.GSParams(maximum_fft_size=32768)
        obj = galsim.Convolve([psf, sheared_gal], gsparams=gsp)
    else:
        raise ValueError("For now only supported experiments are 'ideal' or 'superbit'")

    # Draw images
    obj_im = obj.drawImage(nx=npix, ny=npix, scale=scale).array
    psf_im = psf.drawImage(nx=npix, ny=npix, scale=scale).array

    # Add noise
    nse = rng.normal(size=obj_im.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.shape, scale=nse_sd)

    cen = npix // 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen + dy / scale, col=cen + dx / scale)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)

    # Add small noise to PSF for stability
    target_psf_noise = psf_im.max() / 1000.0

    psf_obs = ngmix.Observation(
        image=psf_im,
        weight=np.ones_like(psf_im) / target_psf_noise**2,
        jacobian=psf_jac,
    )
    obj_obs = ngmix.Observation(
        image=obj_im + nse,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    # Calculate SNR using ngmix built-in method
    snr = obj_obs.get_s2n()
    # Store SNR in metadata
    obj_obs.update_meta_data({'snr': snr})


    # Store the clean image as an attribute
    try :
        sheared_im = sheared_gal.drawImage(nx=npix, ny=npix, scale=scale).array
    except (galsim.errors.GalSimFFTSizeError, galsim.errors.GalSimError) as e:
        print(f"GalSim error drawing clean galaxy: {e}")
        # Create a fallback zero image
        sheared_im = np.zeros((npix, npix))
        
    obj_obs.clean_image = sheared_im

    return obj_obs

def search_psf_files(path):
    all_psf_files = []
    search_path = os.path.join(path, 'psfex-output', '*.psf')
    all_psf_files.extend(glob(search_path))
    return all_psf_files

def get_background_file(psf_file):
    # Extract base name without directory
    fname = os.path.basename(psf_file)

    # Remove the "_starcat.psf" suffix
    stem = fname.replace("_starcat.psf", "")

    # Build the new file name
    new_fname = stem + ".bkg_rms.fits"

    # Replace psfex-output with sky_backgrounds
    new_dir = os.path.dirname(psf_file).replace("psfex-output", "sky_backgrounds")

    return os.path.join(new_dir, new_fname)

def import_psf(psf_files, ud, xsize=WCS_PARAMS['image_xsize'], ysize=WCS_PARAMS['image_ysize'], margin=MARGIN):
    maxexp = len(psf_files)
    # random position
    x = margin + (xsize - 2*margin) * ud()
    y = margin + (ysize - 2*margin) * ud()

    # random integer between 1 and maxexp
    exp = int(1 + (maxexp) * ud())  
    image_pos = galsim.PositionD(x=x, y=y)  
    psf_file = psf_files[exp-1]

    wcs = create_wcs_from_params(WCS_PARAMS)
    psf = galsim.des.DES_PSFEx(psf_file, wcs=wcs)
    this_psf = psf.getPSF(image_pos)

    return this_psf
