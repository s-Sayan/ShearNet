import numpy as np
import galsim
import ngmix

def generate_dataset(samples, psf_fwhm, npix=53, scale=0.2):
    images = []
    labels = []

    for seed in range(samples):
        g1, g2 = np.random.uniform(-0.1, 0.1, size=2)  # Random shears
        obj_obs = sim_func(g1, g2, seed, psf_fwhm, npix=npix, scale=scale)
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
    psf = galsim.Gaussian(sigma=psf_fwhm)

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