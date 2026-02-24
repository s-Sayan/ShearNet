# helpers_ngmix_mcal.py
import numpy as np
import ngmix
from astropy.table import Table
import superbit_lensing.utils as utils
from ngmix.gaussmom import GaussMom

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
