"""Multi-panel comparison plots."""

import numpy as np
import matplotlib.pyplot as plt

from ._common import format_colorbar_scientific


def plot_footprint_comparison(
    fields,
    grids,
    labels,
    meas_pt=None,
    n_levels=6,
    vmin=None,
    vmax=None,
    cmap="turbo",
    figsize=None,
    title=None,
):
    """Multi-panel contour plot comparing footprint models side-by-side.

    Parameters
    ----------
    fields : list of ndarray
        List of 2-D footprint fields to compare.
    grids : list of tuple
        List of (X, Y) coordinate arrays (one per field). Grids may differ.
    labels : list of str
        Subplot titles.
    meas_pt : tuple (x, y), optional
        Measurement point coordinates to mark with a red star on each panel.
    n_levels : int
        Number of contour levels (default 6).
    vmin : float, optional
        Minimum contour level. If None, uses 1e-5.
    vmax : float, optional
        Maximum contour level. If None, uses max across all fields.
    cmap : str
        Colormap name (default "turbo").
    figsize : tuple, optional
        Figure size (width, height).
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes
    """
    n = len(fields)
    if figsize is None:
        figsize = (8, 8)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]

    if vmax is None:
        vmax = max(np.max(f) for f in fields)
    if vmin is None:
        vmin = 1e-5

    levels = np.linspace(vmin, vmax, n_levels, endpoint=False)

    for i, (flx, grid, label, ax) in enumerate(zip(fields, grids, labels, axes)):
        X, Y = grid
        plot = ax.contour(
            X, Y, flx, levels, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=4.0
        )
        ax.set_title(label)
        ax.set_xlabel("x [m]")
        if i == 0:
            ax.set_ylabel("y [m]")

        if meas_pt is not None:
            ax.scatter(meas_pt[0], meas_pt[1], zorder=5, marker="*", color="red", s=300)

    cbar = fig.colorbar(plot, ax=axes, shrink=0.8, location="bottom")
    format_colorbar_scientific(cbar, "$m^{-2}$")

    if title:
        fig.suptitle(title)

    return fig, axes


def plot_field_comparison(fields, domain, src_pt=None, cmap="turbo", figsize=None):
    """2x2 panel comparison: conc, flux, and relative differences.

    Top-left: concentration, top-right: flux.
    Bottom-left: relative difference concentration, bottom-right: relative difference flux.

    Parameters
    ----------
    fields : dict
        Must contain keys: "conc", "flx", "conc_ref", "flx_ref".
        All values are 2-D arrays.
    domain : tuple (xmax, ymax)
        Domain extents for imshow.
    src_pt : tuple (x, y), optional
        Source point coordinates to mark with a red star on top panels.
    cmap : str
        Colormap name (default "turbo").
    figsize : tuple, optional
        Figure size (width, height). Default (10, 6).

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes (2x2)
    """
    if figsize is None:
        figsize = (10, 6)

    conc = fields["conc"]
    flx = fields["flx"]
    conc_ref = fields["conc_ref"]
    flx_ref = fields["flx_ref"]

    diff_conc = (conc - conc_ref) / np.max(conc_ref)
    diff_flx = (flx - flx_ref) / np.max(flx_ref)

    extent = [0, domain[0], 0, domain[1]]
    shrink = 0.7

    fig, axs = plt.subplots(
        2, 2, figsize=figsize, sharex=True, sharey=True, layout="constrained"
    )

    # Top-left: concentration
    plot = axs[0, 0].imshow(conc, origin="lower", cmap=cmap, extent=extent)
    axs[0, 0].set_title("Numerical concentration")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].xaxis.set_tick_params(labelbottom=False)
    cbar = fig.colorbar(plot, ax=axs[0, 0], shrink=shrink, location="bottom")
    format_colorbar_scientific(cbar, "a.u.")

    # Top-right: flux
    plot = axs[0, 1].imshow(flx, origin="lower", cmap=cmap, extent=extent)
    axs[0, 1].set_title("Numerical flux")
    axs[0, 1].xaxis.set_tick_params(labelbottom=False)
    axs[0, 1].yaxis.set_tick_params(labelleft=False)
    cbar = fig.colorbar(plot, ax=axs[0, 1], shrink=shrink, location="bottom")
    format_colorbar_scientific(cbar, "a.u. m/s")

    # Bottom-left: relative difference concentration
    plot = axs[1, 0].imshow(diff_conc, origin="lower", cmap=cmap, extent=extent)
    axs[1, 0].set_title("Relative difference to analytic concentration")
    axs[1, 0].set_xlabel("x [m]")
    axs[1, 0].set_ylabel("y [m]")
    cbar = fig.colorbar(plot, ax=axs[1, 0], shrink=shrink, location="bottom")
    format_colorbar_scientific(cbar)

    # Bottom-right: relative difference flux
    plot = axs[1, 1].imshow(diff_flx, origin="lower", cmap=cmap, extent=extent)
    axs[1, 1].set_title("Relative difference to analytic flux")
    axs[1, 1].set_xlabel("x [m]")
    axs[1, 1].yaxis.set_tick_params(labelleft=False)
    cbar = fig.colorbar(plot, ax=axs[1, 1], shrink=shrink, location="bottom")
    format_colorbar_scientific(cbar)

    if src_pt is not None:
        axs[0, 0].scatter(
            src_pt[0], src_pt[1], zorder=5, marker="*", color="red", s=100
        )
        axs[0, 1].scatter(
            src_pt[0], src_pt[1], zorder=5, marker="*", color="red", s=100
        )

    return fig, axs
