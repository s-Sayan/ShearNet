# ========================== IMPORTS ==========================
import galsim
import ngmix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import Table
from shearnet.methods.ngmix import _get_priors, mp_fit_one, ngmix_pred, response_calculation, mp_fit_one_single

# ----- Simulation controls -----
hlr = 0.5
flux = 12258.97
psf_fwhm  = 0.5
scale = 0.141
npix = 63
nse_sd = 12.719674
seed = 150
n_obs = 100000
Njack = 20
shear_true = 0.02    # true applied shear for + and -
pdf_name = "sample_galaxies_triplet.pdf"

# ----- Models -----
psf_model = "gauss"
gal_model = "gauss"

# ----- Catalog -----
cosmos_cat_fname = "/projects/mccleary_group/superbit/galsim_data/cosmos15_superbit2023_phot_shapes_with_sigma.csv"
cosmos_cat = Table.read(cosmos_cat_fname, format="csv")

def main():

    # ========================= SIMULATE OBSERVATIONS =========================
    rng = np.random.RandomState(seed=seed)

    observations_0 = []  # no extra shear
    observations_p = []  # g1 = +shear_true
    observations_m = []  # g1 = -shear_true

    for _ in range(n_obs):
        index = rng.randint(len(cosmos_cat))
        phi = cosmos_cat[index]['c10_sersic_fit_phi'] * galsim.radians
        q   = cosmos_cat[index]['c10_sersic_fit_q']

        base_gal = galsim.Exponential(half_light_radius=hlr, flux=flux)
        if q > 1.0:
            q = 1/q
        base_gal = base_gal.shear(q=q, beta=phi)
        psf = galsim.Gaussian(fwhm=psf_fwhm)

        gal_0 = base_gal
        gal_p = base_gal.shear(g1=+shear_true, g2=0.0)
        gal_m = base_gal.shear(g1=-shear_true, g2=0.0)

        final_0 = galsim.Convolve([psf, gal_0])
        final_p = galsim.Convolve([psf, gal_p])
        final_m = galsim.Convolve([psf, gal_m])

        stamp_0 = final_0.drawImage(nx=npix, ny=npix, scale=scale).array
        stamp_p = final_p.drawImage(nx=npix, ny=npix, scale=scale).array
        stamp_m = final_m.drawImage(nx=npix, ny=npix, scale=scale).array

        psf_im = psf.drawImage(nx=npix, ny=npix, scale=scale).array
        nse    = rng.normal(size=stamp_0.shape, scale=nse_sd)
        nse_im = rng.normal(size=stamp_0.shape, scale=nse_sd)

        cen = npix // 2
        jac     = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)
        psf_jac = ngmix.jacobian.DiagonalJacobian(scale=scale, row=cen, col=cen)
        target_psf_noise = psf_im.max() / 1000.0

        psf_obs = ngmix.Observation(
            image=psf_im,
            weight=np.ones_like(psf_im) / target_psf_noise**2,
            jacobian=psf_jac,
        )

        def build_obs(stamp):
            return ngmix.Observation(
                image=stamp + nse,
                noise=nse_im,
                weight=np.ones_like(nse_im) / nse_sd**2,
                jacobian=jac,
                bmask=np.zeros_like(nse_im, dtype=np.int32),
                ormask=np.zeros_like(nse_im, dtype=np.int32),
                psf=psf_obs,
            )

        observations_0.append(build_obs(stamp_0))
        observations_p.append(build_obs(stamp_p))
        observations_m.append(build_obs(stamp_m))

    # ================= Checking out a few galaxies =================

    n_show = 5 # how many random galaxies to show

    all_idx = np.arange(len(observations_0))

    rng_show = np.random.RandomState(seed + 123) 
    choice = rng_show.choice(all_idx, size=min(n_show, len(all_idx)), replace=False)

    with PdfPages(pdf_name) as pdf:
        fig, axes = plt.subplots(
            nrows=len(choice), ncols=3,
            figsize=(6, 2.0 * len(choice)),
        )

        # if only one galaxy, axes will be 1D, so make it 2D for simplicity
        if len(choice) == 1:
            axes = np.expand_dims(axes, axis=0)

        col_sets  = [observations_0, observations_p, observations_m]
        col_titles = ["0-shear", "g1=+0.02", "g1=-0.02"]

        for r, idx in enumerate(choice):
            for c, (obs_list, title) in enumerate(zip(col_sets, col_titles)):
                ax = axes[r, c]
                img = obs_list[idx].image  # ngmix.Observation image array
                ax.imshow(img, origin="lower", cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(title)

        plt.tight_layout()
        pdf.savefig(fig, dpi=600)
        plt.close(fig)

    # ======================= SHEAR ESTIMATION PIPELINE =======================
    def run_shear_pipeline(observations, prior, rng, psf_model, gal_model, step=0.01):
        datalist = mp_fit_one(
            observations,
            prior,
            rng,
            psf_model=psf_model,
            gal_model=gal_model,
        )

        r11_list, r22_list, r12_list, r21_list, c1_list, c2_list, c1_psf_list, c2_psf_list = response_calculation(datalist, step)

        preds = ngmix_pred(datalist)
        g1 = preds[:, 0]
        g2 = preds[:, 1]

        return dict(
            datalist=datalist,
            r11=r11_list,
            r22=r22_list,
            r12=r12_list,
            r21=r21_list,
            c1=c1_list,
            c2=c2_list,
            c1_psf=c1_psf_list,
            c2_psf=c2_psf_list,
            g1=g1,
            g2=g2,
        )

    # Run all three sets
    rng = np.random.RandomState(seed=seed)
    prior = _get_priors(seed)

    obs_sets = {"g0": observations_0, "gp": observations_p, "gm": observations_m}
    results = {key: run_shear_pipeline(obs_list, prior, rng, psf_model, gal_model)
            for key, obs_list in obs_sets.items()}

    # ======================= m ESTIMATION (PAIRWISE) =======================
    res_0 = results["g0"]
    res_p = results["gp"]
    res_m = results["gm"]

    g1_p = res_p["g1"]
    g1_m = res_m["g1"]
    R1_p = np.array(res_p["r11"])
    R1_m = np.array(res_m["r11"])

    mask = np.isfinite(g1_p) & np.isfinite(g1_m) & np.isfinite(R1_p) & np.isfinite(R1_m)
    g1_p, g1_m, R1_p, R1_m = g1_p[mask], g1_m[mask], R1_p[mask], R1_m[mask]

    R1_pair = 0.5 * (R1_p + R1_m)
    gamma1_per = (g1_p - g1_m) / (2.0 * R1_pair)
    gamma1_est = np.mean(gamma1_per)

    m_est = gamma1_est / shear_true - 1

    # --- Jackknife on m ---
    indices = np.arange(len(gamma1_per))
    chunks = np.array_split(indices, Njack)

    m_jack_values = []
    for chunk in chunks:
        mask_j = np.ones(len(gamma1_per), dtype=bool)
        mask_j[chunk] = False
        gamma1_est_jack = np.mean(gamma1_per[mask_j])
        m_jack_values.append(gamma1_est_jack / shear_true - 1)

    m_jack_values = np.array(m_jack_values)
    m_mean = np.mean(m_jack_values)
    m_err = np.sqrt((Njack - 1) / Njack * np.sum((m_jack_values - m_mean)**2))

    # ======================= c ESTIMATION =======================
    g1_0 = res_0["g1"]
    mask0 = np.isfinite(g1_p) & np.isfinite(g1_m) & np.isfinite(g1_0)
    g1_p, g1_m, g1_0 = g1_p[mask0], g1_m[mask0], g1_0[mask0]

    chunks = np.array_split(np.arange(g1_p.size), Njack)
    c1_jack = []

    for chunk in chunks:
        mask_j = np.ones(g1_p.size, dtype=bool)
        mask_j[chunk] = False
        e_p_jack = np.mean(g1_p[mask_j])
        e_m_jack = np.mean(g1_m[mask_j])
        e_0_jack = np.mean(g1_0[mask_j])
        c1_jack.append(0.5 * (e_p_jack + e_m_jack) - e_0_jack)

    c1_jack = np.array(c1_jack)
    c1_mean = np.mean(c1_jack)
    c1_err  = np.sqrt((Njack - 1) / Njack * np.sum((c1_jack - c1_mean)**2))

    # ======================= FINAL OUTPUT =======================
    print(f"m = {m_mean:.6e} ± {m_err:.6e}")
    print(f"c = {c1_mean:.6e} ± {c1_err:.6e}")


if __name__ == "__main__":
    main()