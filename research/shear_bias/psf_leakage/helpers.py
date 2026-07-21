# helpers_ngmix_mcal.py
import numpy as np
import ngmix
from astropy.table import Table
import shearnet.utils.superbit as utils  # vendored from superbit-lensing (see shearnet/utils/superbit.py)
from ngmix.gaussmom import GaussMom


## the trained state should be imported here
# state = checkpoints.restore_checkpoint(ckpt_dir=model_dir, target=state)

def get_sn_output(state, obs):
    obs_im = obs.image
    psf_im = obs.psf.image
    preds = state.apply_fn(state.params, obs_im, psf_im, deterministic=True)
    return preds

def get_em_ngauss(name):
    ngauss=int( name[2:] )
    return ngauss

def get_coellip_ngauss(name):
    ngauss=int( name[7:] )
    return ngauss

def get_init_guess(obs):
    gm = GaussMom(1.2).go(obs)
    if gm['flags'] == 0:
        flux_guess = gm['flux']
        Tguess = gm['T']
    else:
        gm = GaussMom(1.2).go(obs.psf)
        if gm['flags'] == 0:
            Tguess = 2 * gm['T']
        else:
            jacobian = obs._jacobian
            Tguess = 4*jacobian.get_scale()**2
        flux_guess = np.sum(obs.image)    
    return Tguess, flux_guess

def _get_priors(seed):
    rng = np.random.RandomState(seed)

    g_sigma = 0.3
    g_prior = ngmix.priors.GPriorBA(g_sigma, rng=rng)

    row, col = 0.0, 0.0
    row_sigma, col_sigma = 0.2, 0.2
    cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng=rng)

    Tminval = -1.0
    Tmaxval = 1000
    T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

    Fminval = -1.0e1
    Fmaxval = 1.0e5
    F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

    priors = ngmix.joint_prior.PriorSimpleSep(cen_prior, g_prior, T_prior, F_prior)
    return priors


def progress(total, miniters=1):
    last_print_n = 0
    last_printed_len = 0
    sl = str(len(str(total)))
    mf = "%" + sl + "d/%" + sl + "d %3d%%"
    for i in range(total):
        yield i

        num = i + 1
        if i == 0 or num == total or num - last_print_n >= miniters:
            meter = mf % (num, total, 100 * float(num) / total)
            nspace = max(last_printed_len - len(meter), 0)

            print("\r" + meter + " " * nspace, flush=True, end="")
            last_printed_len = len(meter)
            if i > 0:
                last_print_n = num

    print(flush=True)


def make_struct(res, obs, state=None):
    """
    make the data structure

    Parameters
    ----------
    res: dict
        With keys 's2n', 'e', and 'T'
    obs: ngmix.Observation
        The observation for this shear type
    shear_type: str
        The shear type

    Returns
    -------
    1-element array with fields
    """
    dt = [
        ('flags', 'i4'),
        ('s2n', 'f8'),
        ('g', 'f8', 2),
        ('T', 'f8'),
        ('Tpsf', 'f8'),
        ('gpsf', 'f8', 2),
        ("g_sn", "f8", 2),        
    ]
    data = np.zeros(1, dtype=dt)
    data['flags'] = res['flags']
    if res['flags'] == 0:
        data['s2n'] = res['s2n']
        # for moments we are actually measureing e, the elliptity
        try:
            data['g'] = res['e']
        except KeyError:
            data['g'] = res['g']
        data['T'] = res['T']
    else:
        data['s2n'] = np.nan
        data['g'] = np.nan
        data['T'] = np.nan
        data['Tpsf'] = np.nan

    # we only have one epoch and band, so we can get the psf T from the
    # observation rather than averaging over epochs/bands
    admom_dict = utils.get_admoms_ngmix_fit(obs.psf, reduced=True)
    if admom_dict['flags'] == 0:
        g1psf, g2psf, Tpsf = admom_dict['e1'], admom_dict['e2'], admom_dict['T']
    else:
        g1psf, g2psf, Tpsf = np.nan, np.nan, np.nan
    data['Tpsf'] = Tpsf
    data['gpsf'] = np.array([g1psf, g2psf])

    if state is not None:
        sn_preds = get_sn_output(state, obs)
        g_sn  = np.array(sn_preds) 
        data["g_sn"] = g_sn
    else:
        data['g_sn'] = np.nan

    return data

def process_obs(obs, boot, return_images=False):
    resdict = boot.go(obs)
    dlist = make_struct(res=resdict, obs=obs)
    struct = np.hstack(dlist)
    if return_images:
        return struct, obs.image.copy(), obs.psf.image.copy()
    return struct


# ============================================================
# PSF-response (metacal) correction
#
# Mirrors the reference implementation in
#   ngmix/tests/test_metacal_galsim_psf_response.py
# where the PSF leakage is removed via the metacal PSF response:
#
#     R11_psf = (g['1p_psf'] - g['1m_psf']) / (2 * step)
#     g_corrected = g['noshear'] - g_psf * R11_psf
#
# Here R11_psf is a scalar (PSF sheared in the 1-direction only) and is
# applied to both components of the measured PSF ellipticity g_psf, exactly
# as in the reference (`correction += g_psf.mean(axis=0) * R11_psf`).
# ============================================================
def make_struct_response(res, obs, shear_type):
    """
    Like ``make_struct`` but carries a ``shear_type`` field so the metacal
    shear types ('noshear', '1p_psf', '1m_psf') can be disentangled later.
    ShearNet predictions (``g_sn``) default to NaN and are filled in the main
    process to avoid running JAX inside worker processes.
    """
    dt = [
        ("flags", "i4"),
        ("shear_type", "U7"),
        ("s2n", "f8"),
        ("g", "f8", 2),
        ("T", "f8"),
        ("Tpsf", "f8"),
        ("gpsf", "f8", 2),
        ("g_sn", "f8", 2),
    ]
    data = np.zeros(1, dtype=dt)
    data["shear_type"] = shear_type
    data["flags"] = res["flags"]

    if res["flags"] == 0:
        data["s2n"] = res["s2n"]
        try:
            data["g"] = res["e"]
        except KeyError:
            data["g"] = res["g"]
        data["T"] = res["T"]
    else:
        data["s2n"] = np.nan
        data["g"] = np.nan
        data["T"] = np.nan
        data["Tpsf"] = np.nan

    admom_dict = utils.get_admoms_ngmix_fit(obs.psf, reduced=True)
    if admom_dict["flags"] == 0:
        g1psf, g2psf, Tpsf = admom_dict["e1"], admom_dict["e2"], admom_dict["T"]
    else:
        g1psf, g2psf, Tpsf = np.nan, np.nan, np.nan
    data["Tpsf"] = Tpsf
    data["gpsf"] = np.array([g1psf, g2psf])

    # inference happens in the main process
    data["g_sn"] = np.nan

    return data


def process_obs_response(obs, boot):
    """
    Run a metacal bootstrap (with PSF-shear types) on ``obs`` and return the
    per-shear-type struct array together with the sheared images needed to run
    ShearNet on each shear type in the main process.
    """
    resdict, obsdict = boot.go(obs)
    dlist = [
        make_struct_response(res=sres, obs=obsdict[stype], shear_type=stype)
        for stype, sres in resdict.items()
    ]
    struct = np.hstack(dlist)
    mcal_images = {
        stype: (obsdict[stype].image.copy(), obsdict[stype].psf.image.copy())
        for stype in obsdict
    }
    return struct, mcal_images


# ============================================================
# Skip-deconvolution ("direct") PSF response
#
# Instead of measuring the PSF response through metacal (which deconvolves the
# galaxy from its PSF and reconvolves it with a *dilated, sheared* target PSF),
# we shear the real PSF directly and convolve the ORIGINAL galaxy with it. This
# bypasses the deconvolution/dilation step entirely, so the images stay in the
# form the shape measurement software (ngmix or ShearNet) was designed for:
#   galaxy (x) PSF, galaxy (x) PSF sheared +g1, galaxy (x) PSF sheared -g1.
# The per-object PSF response is then the usual finite difference
#   R11_psf = (g['1p_psf'] - g['1m_psf']) / (2 * step),
# assembled downstream by ``leakage_response_to_table`` (the arithmetic is
# identical to the metacal path; only the images differ).
#
# See ``bypass_deconvolution_test.md`` in this directory for the rationale.
# ============================================================
def render_psf_shear_variants(
    obj0, psf, *, step, noise, scale, npix, npix_psf, rng, gsp=None, dy=0.0, dx=0.0
):
    """
    Render the same galaxy convolved with the PSF and with the PSF sheared by
    ``+/-step`` in g1, returning ngmix Observations keyed by shear type
    ('noshear', '1p_psf', '1m_psf').

    A single noise realisation is shared across the three renders so it cancels
    in the +/- difference (common random numbers). No deconvolution/metacal is
    involved -- the reduced-shear ``step`` is applied straight to the real PSF
    before convolution, matching ngmix/metacal's ``Shape(step, 0)`` convention.

    Parameters
    ----------
    obj0 : galsim.GSObject
        The (already shifted, if desired) pre-PSF galaxy profile.
    psf : galsim.GSObject
        The PSF profile to shear and convolve with.
    step : float
        Reduced-shear step applied to the PSF in the g1 direction.
    noise, scale, npix, npix_psf : float/int
        Image noise sigma, pixel scale, galaxy stamp size, PSF stamp size.
    rng : np.random.RandomState
        Source of the shared noise field.
    gsp : galsim.GSParams, optional
    dy, dx : float
        Sub-pixel galaxy shift (arcsec), used to offset the jacobian centre so
        it matches how ``make_data`` places the object.
    """
    import galsim

    if gsp is None:
        gsp = galsim.GSParams(maximum_fft_size=32768)

    # one shared noise realisation for all three renders
    im_noise = rng.normal(scale=noise, size=(npix, npix))

    psf_variants = {
        "noshear": psf,
        "1p_psf": psf.shear(g1=+step, g2=0.0),
        "1m_psf": psf.shear(g1=-step, g2=0.0),
    }

    obs_dict = {}
    for stype, psf_used in psf_variants.items():
        obj_psf = galsim.Convolve(psf_used, obj0, gsparams=gsp)

        psf_im = psf_used.withGSParams(gsp).drawImage(
            nx=npix_psf, ny=npix_psf, scale=scale
        ).array
        im = obj_psf.withGSParams(gsp).drawImage(
            nx=npix, ny=npix, scale=scale
        ).array
        im = im + im_noise

        cen = (np.array(im.shape) - 1.0) / 2.0
        psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0

        jacobian = ngmix.DiagonalJacobian(
            row=cen[0] + dy / scale, col=cen[1] + dx / scale, scale=scale,
        )
        psf_jacobian = ngmix.DiagonalJacobian(
            row=psf_cen[0], col=psf_cen[1], scale=scale,
        )

        wt = im * 0 + 1.0 / noise**2
        psf_noise = psf_im.max() / 1000.0
        psf_wt = psf_im * 0 + 1.0 / psf_noise**2

        psf_obs = ngmix.Observation(psf_im, weight=psf_wt, jacobian=psf_jacobian)
        obs_dict[stype] = ngmix.Observation(
            im, weight=wt, jacobian=jacobian, psf=psf_obs
        )

    return obs_dict


def process_obs_direct(obs_dict, boot=None, do_ngmix=True):
    """
    Skip-deconvolution counterpart of ``process_obs_response``.

    Given the pre-rendered PSF-shear-variant observations from
    ``render_psf_shear_variants``, optionally fit each with ngmix and return the
    per-shear-type struct array plus the (galaxy, psf) images so ShearNet can be
    evaluated on each variant in the main process.

    ``do_ngmix=False`` skips the ngmix fit entirely (ShearNet-only run): the
    ngmix shape ``g`` is left NaN while ``gpsf`` and the images are still
    populated. This is what makes the two shape-measurement softwares
    independently switchable from the config.
    """
    dlist = []
    for stype, obs in obs_dict.items():
        if do_ngmix and boot is not None:
            res = boot.go(obs)
        else:
            # skip the ngmix fit -> make_struct_response leaves g/T/s2n NaN
            res = {"flags": 1}
        dlist.append(make_struct_response(res=res, obs=obs, shear_type=stype))
    struct = np.hstack(dlist)
    mcal_images = {
        stype: (obs.image.copy(), obs.psf.image.copy())
        for stype, obs in obs_dict.items()
    }
    return struct, mcal_images


def leakage_response_to_table(data_list, step=0.01, apply_shearnet=False):
    """
    Assemble the PSF-leakage table from metacal per-object struct arrays and
    apply the PSF-response correction.

    The metacal PSF response is an *ensemble* quantity, computed exactly as in
    ngmix/tests/test_metacal_galsim_psf_response.py -- a ratio of ensemble
    means, NOT a per-object finite difference::

        Rbar_psf    = (<g['1p_psf'][0]>    - <g['1m_psf'][0]>)    / (2*step)
        Rbar_psf_sn = (<g_sn['1p_psf'][0]> - <g_sn['1m_psf'][0]>) / (2*step)

    and the (single, constant) response is then applied per object::

        g    = g['noshear']    - gpsf * Rbar_psf
        g_sn = g_sn['noshear'] - gpsf * Rbar_psf_sn        (only if apply_shearnet)

    Using the ensemble mean rather than a per-object response is essential: the
    per-object response ``(g_1p_psf - g_1m_psf)/(2*step)`` is a difference of two
    noisy shear estimates divided by ``2*step`` and has enormous variance, so
    subtracting ``gpsf * r_per_object`` injects that variance (and a noise bias,
    because the metacal shear types share the object's noise realization) into
    every corrected shear.

    IMPORTANT: metacal PSF response only equals the physical PSF leakage for a
    metacal-family estimator that explicitly deconvolves the PSF (e.g. ngmix).
    For a black-box network such as ShearNet the response to shearing the
    (dilated) reconvolution PSF measures out-of-distribution sensitivity to the
    metacal manipulation, not physical leakage, so it does NOT match the raw
    leakage slope and must not be used to "correct" ShearNet. The ShearNet
    response is therefore reported as a diagnostic (``r11_psf_sn``,
    ``rbar_psf_sn``) but only applied to ``g_sn`` when ``apply_shearnet=True``.

    Raw (uncorrected) estimates are kept as ``g_raw`` / ``g_sn_raw``. Returns
    ``(table, Rbar_psf, Rbar_psf_sn)``.
    """
    dg = 2.0 * step

    # ---- pass 1: gather per-object stores + ensemble response accumulators ----
    stores = []
    g1p_ng, g1m_ng, g1p_sn, g1m_sn = [], [], [], []
    for arr in data_list:
        good = bool(np.all(arr["flags"] == 0))
        g_store, g_sn_store = {}, {}
        gpsf = np.array([np.nan, np.nan])
        meta = {}
        for rec in arr:
            stype = rec["shear_type"]
            g_store[stype] = np.array(rec["g"], dtype=float)
            g_sn_store[stype] = np.array(rec["g_sn"], dtype=float)
            if stype == "noshear":
                gpsf = np.array(rec["gpsf"], dtype=float)
                meta = {"s2n": rec["s2n"], "T": rec["T"], "Tpsf": rec["Tpsf"]}
        stores.append((good, g_store, g_sn_store, gpsf, meta))

        if good and "1p_psf" in g_store and "1m_psf" in g_store:
            g1p_ng.append(g_store["1p_psf"][0])
            g1m_ng.append(g_store["1m_psf"][0])
        if good and "1p_psf" in g_sn_store and "1m_psf" in g_sn_store:
            g1p_sn.append(g_sn_store["1p_psf"][0])
            g1m_sn.append(g_sn_store["1m_psf"][0])

    def _rbar(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size == 0 or b.size == 0:
            return np.nan
        return (np.nanmean(a) - np.nanmean(b)) / dg

    rbar_psf = _rbar(g1p_ng, g1m_ng)
    rbar_psf_sn = _rbar(g1p_sn, g1m_sn)

    # ---- pass 2: build rows, applying the single ensemble response ----
    rows = []
    for good, g_store, g_sn_store, gpsf, meta in stores:
        row = {"flag": 0 if good else 1, "gpsf": gpsf}
        row.update(meta)

        # per-object response kept for diagnostics only
        if "1p_psf" in g_store and "1m_psf" in g_store:
            row["r11_psf"] = (g_store["1p_psf"][0] - g_store["1m_psf"][0]) / dg
        else:
            row["r11_psf"] = np.nan
        if "1p_psf" in g_sn_store and "1m_psf" in g_sn_store:
            row["r11_psf_sn"] = (g_sn_store["1p_psf"][0] - g_sn_store["1m_psf"][0]) / dg
        else:
            row["r11_psf_sn"] = np.nan
        row["rbar_psf"] = rbar_psf
        row["rbar_psf_sn"] = rbar_psf_sn

        g_noshear = g_store.get("noshear", np.array([np.nan, np.nan]))
        row["g_raw"] = g_noshear
        row["g"] = g_noshear - gpsf * rbar_psf if np.isfinite(rbar_psf) else g_noshear

        g_sn_noshear = g_sn_store.get("noshear", np.array([np.nan, np.nan]))
        row["g_sn_raw"] = g_sn_noshear
        # ShearNet: metacal response is not physical leakage -> off by default
        if apply_shearnet and np.isfinite(rbar_psf_sn):
            row["g_sn"] = g_sn_noshear - gpsf * rbar_psf_sn
        else:
            row["g_sn"] = g_sn_noshear

        rows.append(row)

    return Table(rows), rbar_psf, rbar_psf_sn


def leakage_response_direct_to_table(data_list, step=0.01,
                                     apply_ngmix=False, apply_shearnet=False):
    """
    Assemble the skip-deconvolution ("direct") PSF-response table.

    The ngmix and ShearNet ensemble responses are each computed over the objects
    where *that* software produced a finite shape -- so the two shape-measurement
    softwares can be enabled independently from the config.

    Correction application (mirrors the metacal path and the m/bias pipeline):
    when ``apply_ngmix`` / ``apply_shearnet`` is set, the constant ensemble
    response is subtracted from the corresponding noshear shape
    (``g = g_noshear - gpsf * Rbar_psf`` and/or
    ``g_sn = g_sn_noshear - gpsf * Rbar_psf_sn``). The uncorrected shapes are
    always kept in ``g_raw`` / ``g_sn_raw``. With both flags off this is a pure
    measurement (``g == g_raw``, ``g_sn == g_sn_raw``).

    Column names match the metacal path (``r11_psf``, ``r11_psf_sn``,
    ``rbar_psf``, ``rbar_psf_sn``, ``gpsf``, ``g``, ``g_raw``, ``g_sn``,
    ``g_sn_raw``, ``s2n``, ``T``, ``Tpsf``).

    Returns ``(table, Rbar_psf, Rbar_psf_sn)``.
    """
    dg = 2.0 * step

    stores = []
    g1p_ng, g1m_ng, g1p_sn, g1m_sn = [], [], [], []
    for arr in data_list:
        g_store, g_sn_store = {}, {}
        gpsf = np.array([np.nan, np.nan])
        meta = {}
        for rec in arr:
            stype = rec["shear_type"]
            g_store[stype] = np.array(rec["g"], dtype=float)
            g_sn_store[stype] = np.array(rec["g_sn"], dtype=float)
            if stype == "noshear":
                gpsf = np.array(rec["gpsf"], dtype=float)
                meta = {"s2n": rec["s2n"], "T": rec["T"], "Tpsf": rec["Tpsf"]}
        stores.append((g_store, g_sn_store, gpsf, meta))

        # accumulate each software's ensemble only where its own shape is finite
        if ("1p_psf" in g_store and "1m_psf" in g_store
                and np.isfinite(g_store["1p_psf"][0])
                and np.isfinite(g_store["1m_psf"][0])):
            g1p_ng.append(g_store["1p_psf"][0])
            g1m_ng.append(g_store["1m_psf"][0])
        if ("1p_psf" in g_sn_store and "1m_psf" in g_sn_store
                and np.isfinite(g_sn_store["1p_psf"][0])
                and np.isfinite(g_sn_store["1m_psf"][0])):
            g1p_sn.append(g_sn_store["1p_psf"][0])
            g1m_sn.append(g_sn_store["1m_psf"][0])

    def _rbar(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size == 0 or b.size == 0:
            return np.nan
        return (np.nanmean(a) - np.nanmean(b)) / dg

    rbar_psf = _rbar(g1p_ng, g1m_ng)
    rbar_psf_sn = _rbar(g1p_sn, g1m_sn)

    rows = []
    for g_store, g_sn_store, gpsf, meta in stores:
        row = {"flag": 0, "gpsf": gpsf}
        row.update(meta)

        if "1p_psf" in g_store and "1m_psf" in g_store:
            row["r11_psf"] = (g_store["1p_psf"][0] - g_store["1m_psf"][0]) / dg
        else:
            row["r11_psf"] = np.nan
        if "1p_psf" in g_sn_store and "1m_psf" in g_sn_store:
            row["r11_psf_sn"] = (g_sn_store["1p_psf"][0] - g_sn_store["1m_psf"][0]) / dg
        else:
            row["r11_psf_sn"] = np.nan
        row["rbar_psf"] = rbar_psf
        row["rbar_psf_sn"] = rbar_psf_sn

        g_noshear = g_store.get("noshear", np.array([np.nan, np.nan]))
        row["g_raw"] = g_noshear
        row["g"] = (g_noshear - gpsf * rbar_psf
                    if apply_ngmix and np.isfinite(rbar_psf) else g_noshear)
        g_sn_noshear = g_sn_store.get("noshear", np.array([np.nan, np.nan]))
        row["g_sn_raw"] = g_sn_noshear
        row["g_sn"] = (g_sn_noshear - gpsf * rbar_psf_sn
                       if apply_shearnet and np.isfinite(rbar_psf_sn) else g_sn_noshear)

        rows.append(row)

    return Table(rows), rbar_psf, rbar_psf_sn


def shear_data_to_table(data_list):
    """
    data_list: list of structured numpy arrays, typically each of length 1
    Returns an Astropy Table with one row per element of data_list.
    """
    rows = []
    for arr in data_list:
        a = arr[0] 
        row = {name: a[name] for name in a.dtype.names} 
        row["flag"] = row.pop("flags")  
        rows.append(row)
    return Table(rows)
