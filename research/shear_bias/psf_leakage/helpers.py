# helpers_ngmix_mcal.py
import numpy as np
import ngmix
from astropy.table import Table
import superbit_lensing.utils as utils
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


def leakage_response_to_table(data_list, step=0.01):
    """
    Assemble the PSF-leakage table from metacal per-object struct arrays and
    apply the PSF-response correction.

    For every object:
        R11_psf     = (g['1p_psf'][0]     - g['1m_psf'][0])     / (2*step)
        R11_psf_sn  = (g_sn['1p_psf'][0]  - g_sn['1m_psf'][0])  / (2*step)
        g          = g['noshear']    - gpsf * R11_psf      (PSF-response corrected)
        g_sn       = g_sn['noshear'] - gpsf * R11_psf_sn   (PSF-response corrected)

    The raw (uncorrected) noshear estimates are retained as ``g_raw`` /
    ``g_sn_raw`` and the responses as ``r11_psf`` / ``r11_psf_sn`` for
    diagnostics. ``g`` / ``g_sn`` hold the corrected values so the existing
    plotting notebook shows the corrected leakage directly.
    """
    dg = 2.0 * step
    rows = []

    for arr in data_list:
        row = {}
        g_store, g_sn_store = {}, {}
        gpsf = np.array([np.nan, np.nan])

        row["flag"] = 0 if np.all(arr["flags"] == 0) else 1

        for rec in arr:
            stype = rec["shear_type"]
            g_store[stype] = np.array(rec["g"], dtype=float)
            g_sn_store[stype] = np.array(rec["g_sn"], dtype=float)
            if stype == "noshear":
                gpsf = np.array(rec["gpsf"], dtype=float)
                row["s2n"] = rec["s2n"]
                row["T"] = rec["T"]
                row["Tpsf"] = rec["Tpsf"]

        row["gpsf"] = gpsf

        # ---- ngmix PSF response + correction ----
        if "1p_psf" in g_store and "1m_psf" in g_store:
            r11_psf = (g_store["1p_psf"][0] - g_store["1m_psf"][0]) / dg
        else:
            r11_psf = np.nan
        g_noshear = g_store.get("noshear", np.array([np.nan, np.nan]))
        row["r11_psf"] = r11_psf
        row["g_raw"] = g_noshear
        row["g"] = g_noshear - gpsf * r11_psf

        # ---- ShearNet PSF response + correction ----
        if "1p_psf" in g_sn_store and "1m_psf" in g_sn_store:
            r11_psf_sn = (g_sn_store["1p_psf"][0] - g_sn_store["1m_psf"][0]) / dg
        else:
            r11_psf_sn = np.nan
        g_sn_noshear = g_sn_store.get("noshear", np.array([np.nan, np.nan]))
        row["r11_psf_sn"] = r11_psf_sn
        row["g_sn_raw"] = g_sn_noshear
        row["g_sn"] = g_sn_noshear - gpsf * r11_psf_sn

        rows.append(row)

    return Table(rows)

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
