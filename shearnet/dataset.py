import numpy as np
import galsim
import ngmix
from scipy.signal import convolve2d

psf_fnmae = '/work/mccleary_group/saha/codes/.empty/psf_cutouts_superbit.npy'
weight_fname = '/work/mccleary_group/saha/codes/.empty/weights_cutouts_superbit.npy'

def generate_dataset(samples, psf_fwhm, npix=53, scale=0.2, exp='ideal'):
    images = []
    labels = []

    for seed in range(samples):
        g1, g2 = np.random.uniform(-0.1, 0.1, size=2)  # Random shears
        if exp == 'superbit':
            obj_obs = sim_func_superbit(g1, g2, seed, npix=npix, scale=scale)
        elif exp == 'ideal':
            obj_obs = sim_func(g1, g2, seed, psf_fwhm, npix=npix, scale=scale)
        else:
            raise ValueError("For now only supported experiments are 'ideal' or 'superbit'")
        
        images.append(obj_obs.image)
        labels.append([g1, g2])  # e1, e2 labels

    return np.array(images), np.array(labels)


def sim_func(g1, g2, seed, psf_fwhm, sigma=1.0, type='exp', npix=53, scale=0.2):
    rng = np.random.RandomState(seed=seed)

    # Create a galaxy object
    if type == 'exp':
        gal = galsim.Exponential(half_light_radius=sigma).shear(g1=g1, g2=g2)
    elif type == 'gauss':
        gal = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)
    else:
        raise ValueError("type must be 'exp' or 'gauss'")

    # Create a PSF
    psf = galsim.Gaussian(fwhm=psf_fwhm)

    # Apply a random shift
    dx, dy = 2.0 * (rng.uniform(size=2) - 0.5) * 0.2
    sheared_gal = gal.shift(dx, dy)

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

def sim_func_superbit(g1, g2, seed, sigma=1.0, type='exp', npix= 53, scale=0.2, superbit_psf_fnmae=psf_fnmae, superbit_weight_fname=weight_fname):
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
        bmask=np.zeros_like(nse_im, dtype=np.int32),
        ormask=np.zeros_like(nse_im, dtype=np.int32),
        psf=psf_obs,
    )

    return obj_obs
    
