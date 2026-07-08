"""Plotting helpers copied verbatim from the superbit-lensing repository.

These were previously imported as ``superbit_lensing.plotter`` by the research
notebooks under ``research/shear_bias``. To remove the external
``superbit-lensing`` dependency, the specific plotting entry points ShearNet
relies on -- :func:`make_psfex_shape_maps` and :class:`PSFLeakagePanelMaker` --
are copied here **unchanged**.

Source: https://github.com/superbit-collaboration/superbit-lensing
        (``superbit_lensing/plotter.py``)

Nothing in this module is original to ShearNet; it is a vendored copy so that
the ``research`` code can run without installing ``superbit-lensing``. Any
change to the upstream implementation should be mirrored from that repository.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .superbit import get_admoms

# NOTE: ``PSFWrapper`` (and its ``piff`` / ``fitsio`` dependencies) is imported
# lazily inside ``make_psfex_shape_maps`` so that importing this module for the
# common case -- ``PSFLeakagePanelMaker`` -- does not require those extra
# packages. In the upstream ``superbit_lensing.plotter`` it was a top-level
# import; the function body below is otherwise unchanged.


def make_psfex_shape_maps(
    psfex_file,
    image_file = None,
    image_xsize=9600,
    image_ysize=6400,
    step=200,
    margin=0,
    smooth=True,
    scale=0.141,
    mode="ngmix",
    reduced=True,
    show=True,
    return_vals=False
):
    """
    Sample PSFEx model across the detector on a coarse grid and plot e1, e2, T maps.

    Returns
    -------
    (e1_map, e2_map, T_map, xx, yy)
      where maps have shape (Ny, Nx) and xx,yy are the coordinate grids.
    """
    from .superbit_psf import PSFWrapper  # lazy import (pulls piff/fitsio)

    interpolation = "bicubic" if smooth else "nearest"

    # ---- load PSFEx model ----
    model = PSFWrapper(psf_file=psfex_file, image_file=image_file)
    #model = psfex.PSFEx(psfex_file)

    # ---- grid of sample points ----
    x = np.arange(margin, image_xsize - margin, step)
    y = np.arange(margin, image_ysize - margin, step)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    Ny, Nx = xx.shape

    e1_map = np.full((Ny, Nx), np.nan, dtype=float)
    e2_map = np.full((Ny, Nx), np.nan, dtype=float)
    T_map  = np.full((Ny, Nx), np.nan, dtype=float)

    # ---- evaluate PSF + moments ----
    for i in range(Ny):
        for j in range(Nx):
            y_im = int(yy[i, j])
            x_im = int(xx[i, j])

            try:
                psf_im = model.get_rec(y_im, x_im)
                #psf_im = psf_im/np.sum(psf_im)
                res = get_admoms(psf_im, scale=scale, mode=mode, reduced=reduced)
                e1_map[i, j] = res["e1"]
                e2_map[i, j] = res["e2"]
                T_map[i, j]  = res["T"]
            except Exception:
                # keep NaNs if anything fails
                continue

    # ---- colormaps (NaNs -> grey) ----
    cmap_shape = cm.RdBu_r.copy()
    cmap_shape.set_bad(color="lightgray")

    cmap_T = cm.viridis.copy()
    cmap_T.set_bad(color="lightgray")

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5),sharey=True)

    datas  = [e1_map, e2_map, T_map]
    labels = ["$e_1$", "$e_2$", "$T$"]
    cmaps  = [cmap_shape, cmap_shape, cmap_T]

    for ax, data, label, cmap in zip(axes, datas, labels, cmaps):
        im = ax.imshow(
            data,
            origin="lower",
            extent=[margin, image_xsize - margin, margin, image_ysize - margin],
            interpolation=interpolation,
            cmap=cmap,
        )
        ax.set_xlim(0, image_xsize)
        ax.set_ylim(0, image_ysize)
        ax.set_facecolor("lightgray")
        ax.set_xlabel("X [pixels]")
        if ax is axes[0]:
            ax.set_ylabel("Y [pixels]")
        ax.set_title(f"PSF {label}")

        # colorbar same height as image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(label)

    plt.tight_layout()

    if show:
        plt.show()
    if return_vals:
        return e1_map, e2_map, T_map, xx, yy


class PSFLeakagePanelMaker:
    """
    Wraps the existing logic into a reusable class.
    IMPORTANT: logic is unchanged; we only move things into methods and make
    e1_gal/e2_gal/etc. explicit inputs (stored on the instance).
    """

    def __init__(
        self,
        *,
        e1_gal,
        e2_gal,
        weights=None,
        NBIN=10,
        MIN_COUNT=20,
        CALIBRATE=False,
        njac=30,
        x_center="median",
        error_type="sem",
        color_e1="#3B4CC0",
        color_e2="#B40426",
    ):
        # ---- store inputs / config (no logic change) ----
        self.e1_gal = np.asarray(e1_gal)
        self.e2_gal = np.asarray(e2_gal)
        if weights is not None:
            self.weights = np.asarray(weights)
        else:
            self.weights = np.ones_like(self.e1_gal)
        self.NBIN = NBIN
        self.MIN_COUNT = MIN_COUNT
        self.CALIBRATE = CALIBRATE
        self.njac = njac

        self.x_center = x_center
        self.error_type = error_type

        self.color_e1 = color_e1
        self.color_e2 = color_e2

    # ---------------------------------------------------------------------
    # (moved as-is) core stats helpers
    # ---------------------------------------------------------------------
    def percentile_binned_mean(
        self,
        x,
        y,
        nbin=20,
        min_count=10,
        weights=None,
        calibrate=False,
        calib=None,
        subtract_global_mean=True,
        x_center="median",
        error_type="sem",  # "sem" or "std"
    ):
        """
        Percentile-bin by x, compute <y> per bin and its uncertainty.
        Returns x_bin, y_bin, yerr_bin, counts, edges
        """
        x = np.asarray(x)
        y = np.asarray(y)

        m = np.isfinite(x) & np.isfinite(y)
        if weights is not None:
            w = np.asarray(weights)
            m &= np.isfinite(w)
        else:
            w = None

        if calibrate:
            if calib is None:
                raise ValueError("calib must be provided when calibrate=True")
            c = np.asarray(calib)
            m &= np.isfinite(c)
        else:
            c = None

        x = x[m]
        y = y[m]
        if w is not None:
            w = w[m]
        if c is not None:
            c = c[m]

        if subtract_global_mean:
            y = y - (np.average(y, weights=w) if w is not None else np.mean(y))

        edges = np.percentile(x, np.linspace(0, 100, nbin + 1))

        x_bin, y_bin, yerr_bin, counts = [], [], [], []

        for i in range(nbin):
            if i < nbin - 1:
                mbin = (x >= edges[i]) & (x < edges[i + 1])
            else:
                mbin = (x >= edges[i]) & (x <= edges[i + 1])

            n = int(np.sum(mbin))
            if n < min_count:
                print(
                    f"[WARNING] number of points in bin {i}: {n}  didn't pass min count = {min_count}"
                )
                continue

            xb = np.median(x[mbin]) if x_center == "median" else np.mean(x[mbin])
            yvals = y[mbin]

            if w is None:
                yb = np.mean(yvals)
                if error_type == "sem":
                    yerr = np.std(yvals, ddof=1) / np.sqrt(n)
                else:
                    yerr = np.std(yvals, ddof=1)
            else:
                wvals = w[mbin]
                yb = np.average(yvals, weights=wvals)
                yerr = np.sqrt(np.average((yvals - yb) ** 2, weights=wvals)) / np.sqrt(
                    n
                )

            if calibrate:
                cvals = c[mbin]
                cb = np.median(cvals)
                if cb == 0:
                    continue
                yb /= cb
                yerr /= cb

            x_bin.append(xb)
            y_bin.append(yb)
            yerr_bin.append(yerr)
            counts.append(n)

        return (
            np.asarray(x_bin),
            np.asarray(y_bin),
            np.asarray(yerr_bin),
            np.asarray(counts),
            edges,
        )

    def slope_from_catalog(
        self,
        x,
        y,
        nbin=20,
        min_count=10,
        weights=None,
        calibrate=False,
        calib=None,
        subtract_global_mean=True,
        x_center="median",
        error_type="sem",
    ):
        x_bin, y_bin, yerr_bin, _, _ = self.percentile_binned_mean(
            x,
            y,
            nbin=nbin,
            min_count=min_count,
            weights=weights,
            calibrate=calibrate,
            calib=calib,
            subtract_global_mean=subtract_global_mean,
            x_center=x_center,
            error_type=error_type,
        )

        w = 1.0 / yerr_bin**2
        alpha, beta = np.polyfit(x_bin, y_bin, 1, w=w)  # y = beta + alpha x
        return alpha, beta, x_bin, y_bin, yerr_bin

    # ---------------------------------------------------------------------
    # (moved as-is) plot helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def make_panel_legend(
        ax,
        showe1e2_leg,
        loc=None,
        fontsize=None,
        columnspacing=0.8,
        handletextpad=0.3,
    ):
        handles, labels = ax.get_legend_handles_labels()

        if showe1e2_leg:
            order = [2, 3, 0, 1]
            ncol = 2
        else:
            order = [0, 1]
            ncol = 1

        ax.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            ncol=ncol,
            frameon=False,
            columnspacing=columnspacing,
            handletextpad=handletextpad,
            loc=loc,
            fontsize=fontsize,
        )

    @staticmethod
    def latex_sci(x, precision=2):
        """2.34e-3 -> 2.34 \\times 10^{-3}"""
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        mant = x / 10**exp
        return rf"{mant:.{precision}f}\times 10^{{{exp}}}"

    @staticmethod
    def set_log_ticks_with_labels(ax, ticks=(10, 20, 30, 50, 100)):
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.ticklabel_format(axis="x", style="plain")
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
        )
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # ---------------------------------------------------------------------
    # the main thing: make_panel (logic unchanged)
    # ---------------------------------------------------------------------
    def make_panel(
        self,
        ax,
        *,
        x_psf,
        xlab,
        calib_for_e1=None,
        calib_for_e2=None,
        x_log_scale=False,
        showe1e2_leg=False,
        plot_confidence=False,
        return_data=False,         
    ):
        alpha_full_1, beta_full_1, x_bin, y_bin_1, yerr_bin = self.slope_from_catalog(
            x=x_psf,
            y=self.e1_gal,
            nbin=self.NBIN,
            min_count=self.MIN_COUNT,
            weights=self.weights,
            calibrate=self.CALIBRATE,
            calib=calib_for_e1,
            subtract_global_mean=True,
            x_center=self.x_center,
            error_type=self.error_type,
        )
        alpha_full_2, beta_full_2, _, y_bin_2, yerr_bin_2 = self.slope_from_catalog(
            x=x_psf,
            y=self.e2_gal,
            nbin=self.NBIN,
            min_count=self.MIN_COUNT,
            weights=self.weights,
            calibrate=self.CALIBRATE,
            calib=calib_for_e2,
            subtract_global_mean=True,
            x_center=self.x_center,
            error_type=self.error_type,
        )

        N = len(x_psf)
        jk_size = N // self.njac

        alpha_jk_1, alpha_jk_2 = [], []
        beta_jk_1, beta_jk_2 = [], []

        for i in range(self.njac):
            mask = np.ones(N, dtype=bool)
            mask[i * jk_size : (i + 1) * jk_size] = False

            a1, b1, _, _, _ = self.slope_from_catalog(
                x_psf[mask],
                self.e1_gal[mask],
                nbin=self.NBIN,
                min_count=self.MIN_COUNT,
                weights=self.weights[mask],
                calibrate=self.CALIBRATE,
                calib=np.asarray(calib_for_e1)[mask] if calib_for_e1 is not None else None,
                subtract_global_mean=True,
                x_center=self.x_center,
                error_type=self.error_type,
            )
            a2, b2, _, _, _ = self.slope_from_catalog(
                x_psf[mask],
                self.e2_gal[mask],
                nbin=self.NBIN,
                min_count=self.MIN_COUNT,
                weights=self.weights[mask],
                calibrate=self.CALIBRATE,
                calib=np.asarray(calib_for_e2)[mask] if calib_for_e2 is not None else None,
                subtract_global_mean=True,
                x_center=self.x_center,
                error_type=self.error_type,
            )

            alpha_jk_1.append(a1)
            alpha_jk_2.append(a2)
            beta_jk_1.append(b1)
            beta_jk_2.append(b2)


        alpha_jk_1 = np.asarray(alpha_jk_1)
        alpha_jk_2 = np.asarray(alpha_jk_2)
        beta_jk_1 = np.asarray(beta_jk_1)
        beta_jk_2 = np.asarray(beta_jk_2)

        alpha_mean_1 = np.mean(alpha_jk_1)
        alpha_err_1 = np.sqrt(
            (self.njac - 1)
            / self.njac
            * np.sum((alpha_jk_1 - alpha_mean_1) ** 2)
        )

        beta_mean_1 = np.mean(beta_jk_1)
        beta_err_1 = np.sqrt(
            (self.njac - 1)
            / self.njac
            * np.sum((beta_jk_1 - beta_mean_1) ** 2)
        )

        alpha_mean_2 = np.mean(alpha_jk_2)
        alpha_err_2 = np.sqrt(
            (self.njac - 1)
            / self.njac
            * np.sum((alpha_jk_2 - alpha_mean_2) ** 2)
        )

        beta_mean_2 = np.mean(beta_jk_2)
        beta_err_2 = np.sqrt(
            (self.njac - 1)
            / self.njac
            * np.sum((beta_jk_2 - beta_mean_2) ** 2)
        )

        xx = np.linspace(np.min(x_bin), np.max(x_bin), 200)
        yy_1 = beta_full_1 + alpha_full_1 * xx
        yy_2 = beta_full_2 + alpha_full_2 * xx

        ax.errorbar(
            x_bin,
            y_bin_1,
            yerr=yerr_bin,
            c=self.color_e1,
            fmt="o",
            capsize=2,
            elinewidth=1.2,
            label=r"$\langle e_1 \rangle$",
        )
        formatted_alpha = (
            f"{alpha_full_1:.3f}"
            if abs(alpha_full_1) >= 1e-2
            else self.latex_sci(alpha_full_1, precision=2)
        )
        ax.plot(
            xx,
            yy_1,
            linewidth=2,
            c=self.color_e1,
            label=rf"$\alpha_1 = {formatted_alpha}\ ({(abs(alpha_mean_1)/alpha_err_1):.2f}\sigma)$",
        )
        dy_1 = np.sqrt((alpha_err_1 * xx)**2 + beta_err_1**2)
        if plot_confidence:
            ax.fill_between(xx, yy_1 - dy_1, yy_1 + dy_1,
                            color=self.color_e1, alpha=0.15, linewidth=0)


        ax.errorbar(
            x_bin,
            y_bin_2,
            yerr=yerr_bin_2,
            c=self.color_e2,
            fmt="s",
            capsize=2,
            elinewidth=1.2,
            label=r"$\langle e_2 \rangle$",
        )
        formatted_alpha = (
            f"{alpha_full_2:.3f}"
            if abs(alpha_full_2) >= 1e-2
            else self.latex_sci(alpha_full_2, precision=2)
        )
        ax.plot(
            xx,
            yy_2,
            linewidth=2,
            c=self.color_e2,
            label=rf"$\alpha_2 = {formatted_alpha}\ ({(abs(alpha_mean_2)/alpha_err_2):.2f}\sigma)$",
        )
        dy_2 = np.sqrt((alpha_err_2 * xx)**2 + beta_err_2**2)
        if plot_confidence:
            ax.fill_between(xx, yy_2 - dy_2, yy_2 + dy_2,
                            color=self.color_e2, alpha=0.15, linewidth=0)

        if x_log_scale:
            if xlab == r"${\rm SNR}$":
                self.set_log_ticks_with_labels(ax, ticks=(10, 20, 30, 50, 100))
            elif xlab == r"$T_{\rm gal}/T_{\rm PSF}$":
                self.set_log_ticks_with_labels(ax, ticks=(3, 5, 7, 10, 20, 30))
            else:
                ax.set_xscale("log")

        ax.axhline(0, color="0.4", linestyle="--", linewidth=1, zorder=0)
        ax.set_xlabel(xlab)
        self.make_panel_legend(ax, showe1e2_leg)

        if return_data:
            return {
                "x_bin":       x_bin,
                "y_bin_1":     y_bin_1,
                "yerr_bin_1":  yerr_bin,
                "y_bin_2":     y_bin_2,
                "yerr_bin_2":  yerr_bin_2,
                "alpha_full_1": alpha_full_1, "alpha_mean_1": alpha_mean_1, "alpha_err_1": alpha_err_1, "beta_full_1": beta_full_1,
                "alpha_full_2": alpha_full_2, "alpha_mean_2": alpha_mean_2, "alpha_err_2": alpha_err_2, "beta_full_2": beta_full_2,
                "beta_err_1": beta_err_1, "beta_err_2": beta_err_2
            }


        
    def make_single_component_panel(
        self,
        ax,
        *,
        x_psf,
        e_gal,
        component_index,
        method_label,
        color,
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=2,
        elinewidth=1.2,
        calib=None,
        x_log_scale=False,
        xlab=None,
    ):
        e_gal = np.asarray(e_gal)

        alpha, beta, x_bin, y_bin, yerr_bin = self.slope_from_catalog(
            x=x_psf, y=e_gal,
            nbin=self.NBIN, min_count=self.MIN_COUNT,
            weights=self.weights, calibrate=self.CALIBRATE, calib=calib,
            subtract_global_mean=True,
            x_center=self.x_center, error_type=self.error_type,
        )

        N = len(x_psf)
        jk_size = N // self.njac
        alpha_jk = []
        for i in range(self.njac):
            mask = np.ones(N, dtype=bool)
            mask[i * jk_size : (i + 1) * jk_size] = False
            a, _, _, _, _ = self.slope_from_catalog(
                x_psf[mask], e_gal[mask],
                nbin=self.NBIN, min_count=self.MIN_COUNT,
                weights=self.weights[mask], calibrate=self.CALIBRATE,
                calib=np.asarray(calib)[mask] if calib is not None else None,
                subtract_global_mean=True,
                x_center=self.x_center, error_type=self.error_type,
            )
            alpha_jk.append(a)

        alpha_jk = np.asarray(alpha_jk)
        alpha_mean = np.mean(alpha_jk)
        alpha_err = np.sqrt(
            (self.njac - 1) / self.njac * np.sum((alpha_jk - alpha_mean) ** 2)
        )

        xx = np.linspace(np.min(x_bin), np.max(x_bin), 200)

        # data points — no label (handled by proxy legend)
        ax.errorbar(x_bin, y_bin, yerr=yerr_bin, c=color, fmt=marker,
                    capsize=capsize, elinewidth=elinewidth, label='_nolegend_')

        # fit line — no label (alpha goes in text box)
        ax.plot(xx, beta + alpha * xx, linewidth=linewidth, c=color,
                linestyle=linestyle, label='_nolegend_')

        if x_log_scale:
            ax.set_xscale("log")
        if xlab is not None:
            ax.set_xlabel(xlab)
        ax.axhline(0, color="0.4", ls="--", lw=1, zorder=0)

        # return alpha info so the caller can build the text box
        formatted = (f"{alpha:.3f}" if abs(alpha) >= 1e-2
                    else self.latex_sci(alpha, precision=2))
        sig = abs(alpha_mean) / alpha_err
        return component_index, method_label, formatted, sig
