import os
from glob import glob
import numpy as np
import galsim
import ngmix
from scipy.signal import convolve2d
from tqdm import tqdm
from ..methods.ngmix import g1_g2_sigma_sample

XIMAGE_SIZE, YIMAGE_SIZE = 9600, 6422 # Usual size of SuperBIT single exposures in pixel unit
MARGIN = 200 # Margins that I wanna use for PSF Rendering

PSF_DATA_DIR = "/projects/mccleary_group/superbit/SHEARNET_DATA"

def generate_dataset(samples, psf_sigma, npix=53, scale=0.141, type='gauss', exp='ideal', nse_sd=1e-5, seed=42, process_psf=False,return_obs=False):
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
        
        obj_obs = sim_func(g1, g2, sigma=sigma, flux=flux, psf_sigma=psf_sigma, nse_sd = nse_sd,  type=type, npix=npix, scale=scale, seed=i, exp=exp, ud=ud, psf_files=psf_files)
        
        galaxy_images = obj_obs.image
        psf_images = obj_obs.psf.image

        if process_psf :
            # Create (height, width, 2) array
            combined_images = np.stack([galaxy_images, psf_images], axis=-1)
            
            images.append(combined_images)
        else :
            images.append(galaxy_images)
        labels.append(np.array([g1, g2, sigma, flux], dtype=np.float32))
        obs.append(obj_obs)
    
    if return_obs:
        return np.array(images), np.array(labels), obs    
    
    return np.array(images), np.array(labels)

def split_combined_images(combined_images):
    """
    Split concatenated images back into separate galaxy and PSF arrays.
    
    Args:
        combined_images: np.ndarray of shape (samples, height, width, 2)
        
    Returns:
        galaxy_images: np.ndarray of shape (samples, height, width)
        psf_images: np.ndarray of shape (samples, height, width)
    """
    galaxy_images = combined_images[..., 0]  # Channel 0
    psf_images = combined_images[..., 1]     # Channel 1
    return galaxy_images, psf_images

def sim_func(g1, g2, sigma=1.0, flux=1.0, psf_sigma=0.5, nse_sd = 1e-5,  type='gauss', npix=53, scale=0.141, seed=42, exp="ideal", ud=None, psf_files=None):

    rng = np.random.RandomState(seed=seed)

    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma, flux=flux).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    sheared_gal = gal.shift(dx, dy)

    # Convolve with PSF
    if exp == 'ideal':
        #gsp = galsim.GSParams(maximum_fft_size=8192000)
        psf = galsim.Gaussian(sigma=psf_sigma)
        obj = galsim.Convolve(sheared_gal, psf)

    elif exp == 'superbit':
        this_psf = import_psf(psf_files, ud)
        gsp=galsim.GSParams(maximum_fft_size=32768)
        obj = galsim.Convolve([this_psf, sheared_gal], gsparams=gsp)
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
    return obj_obs

def search_psf_files(path):
    all_psf_files = []
    search_path = os.path.join(path, 'psfex-output', '*.psf')
    return all_psf_files.extend(glob(search_path))

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

def import_psf(psf_files, ud, xsize=XIMAGE_SIZE, ysize=YIMAGE_SIZE, margin=MARGIN):
    maxexp = len(psf_files)
    # random position
    x = margin + (xsize - 2*margin) * ud()
    y = margin + (ysize - 2*margin) * ud()

    # random integer between 1 and maxexp
    exp = int(1 + (maxexp) * ud())  
    image_pos = galsim.PositionD(x=x, y=y)  
    psf_file = psf_files[exp-1]

    bk_image = get_background_file(psf_file)
    psf = galsim.des.DES_PSFEx(psf_file, bk_image)
    this_psf = psf.getPSF(image_pos)

    return this_psf

def sim_func_dual_shear(g1, g2, seed, psf_sigma, g1_sh=0, g2_sh=0, sigma=1.0, flux=1.0, type='exp', npix=53, scale=0.2):
    rng = np.random.RandomState(seed=seed)

    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma, flux=flux).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    # Create second shear on the elliptic objects
    gal = gal.shear(g1=g1_sh, g2=g2_sh) 

    my_gaussian_image = gal.drawImage(nx=npix, ny=npix, scale=scale)
    my_moments = galsim.hsm.FindAdaptiveMom(my_gaussian_image)
    g1_admom, g2_admom = my_moments.observed_shape.g1, my_moments.observed_shape.g2

    # Create a PSF
    psf = galsim.Gaussian(sigma=psf_sigma)

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    #print(f"Center Shift: {dx, dy}")
    #print(f"Galaxy size: {2 * sigma**2}")
    #print(f"PSF Size: {2 * psf_sigma**2}")
    sheared_gal = gal.shift(dx, dy)
    #sheared_gal = gal

    # Convolve with PSF
    obj = galsim.Convolve(sheared_gal, psf)

    # Draw images
    obj_im = obj.drawImage(nx=npix, ny=npix, scale=scale)
    psf_im = psf.drawImage(nx=npix, ny=npix, scale=scale)

    # Add noise
    nse_sd = 1e-5
    nse = rng.normal(size=obj_im.array.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.array.shape, scale=nse_sd)

    cen = npix // 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen + dy / scale, col=cen + dx / scale)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)

    # Add small noise to PSF for stability
    target_psf_s2n = 500.0
    target_psf_noise = np.sqrt(np.sum(psf_im.array**2)) / target_psf_s2n
    #print(target_psf_noise)
    psf_obs = ngmix.Observation(
        image=psf_im.array,
        weight=np.ones_like(psf_im.array) / target_psf_noise**2,
        jacobian=psf_jac,
    )

    obj_obs = ngmix.Observation(
        image=obj_im.array + nse,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    return obj_obs, g1_admom, g2_admom

def sim_func_decrepted(g1, g2, seed, psf_sigma, sigma=1.0, flux=1.0, type='exp', npix=53, scale=0.2):
    rng = np.random.RandomState(seed=seed)

    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma, flux=flux).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    # Create a PSF
    psf = galsim.Gaussian(sigma=psf_sigma)

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    #print(f"Center Shift: {dx, dy}")
    #print(f"Galaxy size: {2 * sigma**2}")
    #print(f"PSF Size: {2 * psf_sigma**2}")
    sheared_gal = gal.shift(dx, dy)
    #sheared_gal = gal

    # Convolve with PSF
    obj = galsim.Convolve(sheared_gal, psf)

    # Draw images
    obj_im = obj.drawImage(nx=npix, ny=npix, scale=scale)
    psf_im = psf.drawImage(nx=npix, ny=npix, scale=scale)

    # Add noise
    nse_sd = 1e-5
    nse = rng.normal(size=obj_im.array.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.array.shape, scale=nse_sd)

    cen = npix // 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen + dy / scale, col=cen + dx / scale)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)

    # Add small noise to PSF for stability
    target_psf_s2n = 500.0
    target_psf_noise = np.sqrt(np.sum(psf_im.array**2)) / target_psf_s2n
    #print(target_psf_noise)
    psf_obs = ngmix.Observation(
        image=psf_im.array,
        weight=np.ones_like(psf_im.array) / target_psf_noise**2,
        jacobian=psf_jac,
    )

    obj_obs = ngmix.Observation(
        image=obj_im.array + nse,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    return obj_obs

def sim_func_superbit_dual_shear(g1, g2, seed, psf_sigma, g1_sh=0, g2_sh=0, sigma=1.0, flux=1.0, superbit_psf_fnmae=psf_fnmae, type='exp', npix=53, scale=0.2):
    rng = np.random.RandomState(seed=seed)
    
    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma, flux=flux).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    # Create second shear on the elliptic objects
    #gal = gal.shear(g1=g1_sh, g2=g2_sh) 

    my_gaussian_image = gal.drawImage(nx=npix, ny=npix, scale=scale)
    my_moments = galsim.hsm.FindAdaptiveMom(my_gaussian_image)
    g1_admom, g2_admom = my_moments.observed_shape.g1, my_moments.observed_shape.g2

    # Create a PSF
    psf = galsim.Gaussian(sigma=psf_sigma)

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    #print(f"Center Shift: {dx, dy}")
    #print(f"Galaxy size: {2 * sigma**2}")
    #print(f"PSF Size: {2 * psf_sigma**2}")
    sheared_gal = gal.shift(dx, dy)
    #sheared_gal = gal

    obj_im = sheared_gal.drawImage(nx=npix, ny=npix, scale=scale).array
    psf_images = np.load(superbit_psf_fnmae)

    random_psf_index = rng.randint(0, psf_images.shape[0])  # Random index in the range [0, n)
    psf_image = psf_images[random_psf_index].copy()

    # Convolve the object image with the PSF
    psf_convolved = convolve2d(obj_im, psf_image, mode='same', boundary='wrap')

    # Add the noise map to the convolved image


    # Add noise
    nse_sd = 1e-5
    nse = rng.normal(size=obj_im.shape, scale=nse_sd)
    nse_im = rng.normal(size=obj_im.shape, scale=nse_sd)
    final_image = psf_convolved + nse_im

    cen = npix // 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen + dy / scale, col=cen + dx / scale)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)

    # Add small noise to PSF for stability
    target_psf_s2n = 500.0
    target_psf_noise = np.sqrt(np.sum(psf_image**2)) / target_psf_s2n
    #print(target_psf_noise)
    psf_obs = ngmix.Observation(
        image=psf_image,
        weight=np.ones_like(psf_image) / target_psf_noise**2,
        jacobian=psf_jac,
    )

    obj_obs = ngmix.Observation(
        image=final_image,
        noise=nse_im,
        weight=np.ones_like(nse_im) / nse_sd**2,
        jacobian=jac,
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    return obj_obs, g1_admom, g2_admom

def sim_func_superbit(g1, g2, seed, sigma=1.0, type='exp', npix=53, scale=0.2, superbit_psf_fnmae=psf_fnmae, superbit_weight_fname=weight_fname):

    rng = np.random.RandomState(seed=seed)
    
    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    sheared_gal = gal.shift(dx, dy)

    # Draw images
    obj_im = sheared_gal.drawImage(nx=npix, ny=npix, scale=scale).array
    psf_images = np.load(superbit_psf_fnmae)
    weight_images = np.load(superbit_weight_fname)

    # Randomly choose one image using rng
    random_psf_index = rng.randint(0, psf_images.shape[0])  # Random index in the range [0, n)
    psf_image = psf_images[random_psf_index]
    random_weight_index = rng.randint(0, weight_images.shape[0])  # Random index in the range [0, n)
    weight_image = weight_images[random_weight_index]

    # Resize or pad the weight image to match the object image size
    weight_shape = weight_image.shape
    obj_shape = obj_im.shape
    if weight_shape != obj_shape:
        if weight_shape[0] > obj_shape[0] or weight_shape[1] > obj_shape[1]:
            # Crop the center of the weight image if it's larger than obj_im
            crop_x_start = (weight_shape[0] - obj_shape[0]) // 2
            crop_y_start = (weight_shape[1] - obj_shape[1]) // 2
            weight_image_cropped = weight_image[crop_x_start:crop_x_start + obj_shape[0], 
                                               crop_y_start:crop_y_start + obj_shape[1]]
        else:
            # Pad the weight image with random values sampled from the existing weight distribution
            pad_x = (obj_shape[0] - weight_shape[0]) // 2
            pad_y = (obj_shape[1] - weight_shape[1]) // 2
            
            weight_image_flat = weight_image.flatten()
            weight_mean = np.mean(weight_image_flat)
            weight_std = np.std(weight_image_flat)
            
            random_pad_values = rng.normal(loc=weight_mean, scale=weight_std, size=((obj_shape[0] - weight_shape[0]) * (obj_shape[1] - weight_shape[1])))
            
            weight_image_padded = np.pad(weight_image, 
                                          ((pad_x, obj_shape[0] - weight_shape[0] - pad_x),
                                           (pad_y, obj_shape[1] - weight_shape[1] - pad_y)), 
                                          mode='constant', constant_values=0)
            weight_image_padded[pad_x:pad_x + random_pad_values.shape[0] // obj_shape[1], 
                                pad_y:pad_y + random_pad_values.shape[0] % obj_shape[1]] = random_pad_values.reshape(weight_image_padded[pad_x:pad_x + random_pad_values.shape[0] // obj_shape[1], pad_y:].shape)
            weight_image_cropped = weight_image_padded

    else:
        weight_image_cropped = weight_image

    # Generate noise map using the inverse variance (1/sqrt(weight_image))
    noise_map = rng.normal(size=obj_im.shape, scale=1.0 / np.sqrt(weight_image_cropped))

    # Convolve the object image with the PSF
    psf_convolved = convolve2d(obj_im, psf_image, mode='same', boundary='wrap')

    # Add the noise map to the convolved image
    final_image = psf_convolved + noise_map

    cen = npix // 2
    jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen + dy / scale, col=cen + dx / scale)
    psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)

    # Add small noise to PSF for stability
    target_psf_s2n = 500.0
    target_psf_noise = np.sqrt(np.sum(psf_image**2)) / target_psf_s2n
    psf_obs = ngmix.Observation(
        image=psf_image,
        weight=np.ones_like(psf_image) / target_psf_noise**2,
        jacobian=psf_jac,
    )

    obj_obs = ngmix.Observation(
        image=final_image,
        noise=noise_map,
        weight=weight_image_cropped,
        jacobian=jac,
#        bmask=np.zeros_like(nse_im, dtype=np.int32),
#        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    return obj_obs
    
