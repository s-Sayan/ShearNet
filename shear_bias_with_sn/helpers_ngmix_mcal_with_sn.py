# helpers_ngmix_mcal.py
import numpy as np
import ngmix
from astropy.table import Table
import superbit_lensing.utils as utils


## the trained state should be imported here
# state = checkpoints.restore_checkpoint(ckpt_dir=model_dir, target=state)

def get_sn_output(state, obs):
    obs_im = obs.image
    psf_im = obs.psf.image
    preds = state.apply_fn(state.params, obs_im, psf_im, deterministic=True)
    return preds

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


def make_struct(res, obs, shear_type, state=None):
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

    if state is not None:
        sn_preds = get_sn_output(state, obs)
        g_sn  = np.array(sn_preds) 
        data["g_sn"] = g_sn
    else:
        data['g_sn'] = np.nan

    return data

def process_obs(obs, boot):
    resdict, obsdict = boot.go(obs)
    dlist = [
        make_struct(res=sres, obs=obsdict[stype], shear_type=stype)
        for stype, sres in resdict.items()
    ]
    struct = np.hstack(dlist)
    mcal_images = {
        stype: (obsdict[stype].image.copy(), obsdict[stype].psf.image.copy())
        for stype in obsdict
    }
    raw_image = obs.image.copy()
    psf_image = obs.psf.image.copy()
    return struct, mcal_images, raw_image, psf_image

def shear_data_to_table(data_list, mcal_shear=0.01):
    rows = []

    for arr in data_list:
        row = {}
        g_store = {}
        g_sn_store = {}

        row["flag"] = 0 if np.all(arr["flags"] == 0) else 1

        for rec in arr:
            stype = rec["shear_type"]

            for name in arr.dtype.names:
                if name in ("shear_type", "flags"):
                    continue

                val = rec[name]

                if name == "g":
                    g_store[stype] = np.array(val, dtype=float)
                    row[f"g_{stype}"] = np.array(val, dtype=float)
                    continue
                elif name == "g_sn":
                    g_sn_store[stype] = np.array(val, dtype=float)
                    row[f"g_sn_{stype}"] = np.array(val, dtype=float)
                    continue

                row[f"{name}_{stype}"] = val

        dg = 2 * mcal_shear

        if "1p" in g_store and "1m" in g_store:
            row["r11"] = (g_store["1p"][0] - g_store["1m"][0]) / dg

        if "2p" in g_store and "2m" in g_store:
            row["r22"] = (g_store["2p"][1] - g_store["2m"][1]) / dg

        if "1p" in g_sn_store and "1m" in g_sn_store:
            row["r11_sn"] = (g_sn_store["1p"][0] - g_sn_store["1m"][0]) / dg

        if "2p" in g_sn_store and "2m" in g_sn_store:
            row["r22_sn"] = (g_sn_store["2p"][1] - g_sn_store["2m"][1]) / dg

        rows.append(row)

    return Table(rows)


def jackknife_mc_v2(tab_p, tab_m, shear_true, njac=20, g_col="g_noshear", r11_col="r11"):
    N = len(tab_p)
    if njac < 2:
        raise ValueError("njac must be >= 2")
    if njac > N:
        raise ValueError(f"njac ({njac}) cannot exceed number of rows ({N})")

    g_arr_p = np.asarray(tab_p[g_col])
    R_arr_p = np.asarray(tab_p[r11_col])

    g_arr_m = np.asarray(tab_m[g_col])
    R_arr_m = np.asarray(tab_m[r11_col])

    gamma1_per = (g_arr_p[:, 0] - g_arr_m[:, 0]) / 2.0
    c_per = (g_arr_p[:, 1] + g_arr_m[:, 1]) / 2.0

    R1_pair = 0.5 * (R_arr_p + R_arr_m)

    indices = np.arange(N)
    chunks = np.array_split(indices, njac)

    shear_est = np.nanmean(gamma1_per) / np.nanmean(R1_pair)
    m_full = shear_est / shear_true - 1
    c_full = np.nanmean(c_per)

    m_jack, c_jack = [], []
    r11_jack, r11_p_jack, r11_m_jack = [], [], []

    for chunk in chunks:
        mask = np.ones(N, dtype=bool)
        mask[chunk] = False

        g_mean_chunk = np.nanmean(gamma1_per[mask])
        R_mean_chunk = np.nanmean(R1_pair[mask])

        shear_chunk = g_mean_chunk / R_mean_chunk
        m_chunk = shear_chunk / shear_true - 1
        c_chunk = np.nanmean(c_per[mask])

        m_jack.append(m_chunk)
        c_jack.append(c_chunk)
        r11_jack.append(R_mean_chunk)
        r11_p_jack.append(np.nanmean(R_arr_p[mask]))
        r11_m_jack.append(np.nanmean(R_arr_m[mask]))

    m_jack = np.array(m_jack)
    c_jack = np.array(c_jack)
    r11_jack = np.array(r11_jack)
    r11_p_jack = np.array(r11_p_jack)
    r11_m_jack = np.array(r11_m_jack)

    m_mean = np.mean(m_jack)
    c_mean = np.mean(c_jack)
    r11_mean = np.mean(r11_jack)
    r11_p_mean = np.mean(r11_p_jack)
    r11_m_mean = np.mean(r11_m_jack)

    m_err = np.sqrt((njac - 1) * np.mean((m_jack - m_mean) ** 2))
    c_err = np.sqrt((njac - 1) * np.mean((c_jack - c_mean) ** 2))
    r11_err = np.sqrt((njac - 1) * np.mean((r11_jack - r11_mean) ** 2))
    r11_p_err = np.sqrt((njac - 1) * np.mean((r11_p_jack - r11_p_mean) ** 2))
    r11_m_err = np.sqrt((njac - 1) * np.mean((r11_m_jack - r11_m_mean) ** 2))

    return (
        m_full,
        c_full,
        m_mean,
        m_err,
        c_mean,
        c_err,
        m_jack,
        c_jack,
        r11_mean,
        r11_err,
        r11_p_mean,
        r11_p_err,
        r11_m_mean,
        r11_m_err,
    )


def superscript(n):
    sup_map = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(n).translate(sup_map)
