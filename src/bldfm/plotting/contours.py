"""Source area contour visualization functions.

These functions combine the output of ``bldfm.utils.get_source_area()``
with matplotlib contour overlays.  The ``get_source_area`` computation
itself lives in ``bldfm.utils`` -- this module handles only visualization.

Source area contours represent the spatial region that contributes a given
fraction of the measured flux.  Different "base functions" produce different
contour geometries:

- **Contribution contours** (g = f): standard isopleth contours
- **Circular contours** (g = -(r^2)): concentric circles from tower
- **Upwind contours** (g = dot(wind_hat, r)): upwind distance bands
- **Crosswind contours** (g = -(perp distance)^2): crosswind ridges
- **Sector contours** (g = -abs(theta)): angular sectors from upwind axis

See Also
--------
bldfm.utils.get_source_area : Rescale a field so contour levels represent
    cumulative contribution fractions.
"""

import numpy as np
import matplotlib.pyplot as plt

from ._common import ensure_ax, format_colorbar_scientific


def plot_source_area_contours(flx, grid, source_area_field, levels=None,
                              ax=None, contour_colors=None, cmap="RdYlBu_r",
                              title=None, **pcolormesh_kw):
    """Plot footprint field with source area contour overlay.

    The ``source_area_field`` should be the output of
    ``get_source_area(flx, g)`` for some base function ``g``.

    Parameters
    ----------
    flx : ndarray (ny, nx)
        Original footprint field (plotted as pcolormesh background).
    grid : tuple (X, Y, Z) or (X, Y)
        Coordinate arrays from the solver.
    source_area_field : ndarray (ny, nx)
        Rescaled field from ``get_source_area()``, where contour levels
        represent cumulative contribution fractions.
    levels : list of float, optional
        Contribution fraction levels (default [0.25, 0.5, 0.75]).
    ax : matplotlib Axes, optional
    contour_colors : list of str, optional
        Colors for contour lines (default ['white', 'magenta', 'cyan']).
    cmap : str
        Colourmap for background pcolormesh.
    title : str, optional
    **pcolormesh_kw
        Forwarded to ``ax.pcolormesh``.

    Returns
    -------
    ax : matplotlib Axes
    """
    X, Y = grid[:2]

    if levels is None:
        levels = [0.25, 0.5, 0.75]
    if contour_colors is None:
        contour_colors = ["white", "magenta", "cyan"]

    ax = ensure_ax(ax)

    pm = ax.pcolormesh(X, Y, flx, cmap=cmap, shading="auto", **pcolormesh_kw)
    cbar = ax.figure.colorbar(pm, ax=ax)
    format_colorbar_scientific(cbar, label="Footprint [m$^{-2}$]")

    cs = ax.contour(X, Y, source_area_field, levels=levels,
                    colors=contour_colors)
    ax.clabel(cs, fmt=lambda x: f"{x:.0%}", fontsize=8, inline=True)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)

    return ax


def plot_source_area_gallery(flx, grid, meas_pt, wind, levels=None,
                             cmap="RdYlBu_r", figsize=None):
    """Multi-panel plot showing all 5 source area contour types.

    Parameters
    ----------
    flx : ndarray (ny, nx)
        Footprint field.
    grid : tuple (X, Y, Z) or (X, Y)
        Coordinate arrays from the solver.
    meas_pt : tuple (xm, ym)
        Measurement point.
    wind : tuple (u, v)
        Wind components (m/s).
    levels : list of float, optional
        Contribution fraction levels (default [0.25, 0.5, 0.75]).
    cmap : str
        Colourmap for background.
    figsize : tuple, optional
        Figure size (default (18, 10)).

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes (2, 3)
    """
    from ..utils import (
        get_source_area,
        source_area_contribution,
        source_area_circular,
        source_area_upwind,
        source_area_crosswind,
        source_area_sector,
    )

    X, Y = grid[:2]

    contour_types = [
        ("Contribution", source_area_contribution(flx)),
        ("Circular", source_area_circular(X, Y, meas_pt)),
        ("Upwind", source_area_upwind(X, Y, meas_pt, wind)),
        ("Crosswind", source_area_crosswind(X, Y, meas_pt, wind)),
        ("Sector", source_area_sector(X, Y, meas_pt, wind)),
    ]

    if figsize is None:
        figsize = (18, 10)

    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")
    axes_flat = axes.ravel()

    for i, (name, g) in enumerate(contour_types):
        rescaled = get_source_area(flx, g)
        plot_source_area_contours(
            flx, grid, rescaled, levels=levels,
            ax=axes_flat[i], cmap=cmap, title=f"{name} contours",
        )

    # Hide the unused 6th subplot
    axes_flat[5].set_visible(False)

    return fig, axes
