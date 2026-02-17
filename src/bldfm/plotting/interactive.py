"""Interactive (non-matplotlib) plotting backends."""

import numpy as np

from ._common import optional_import


def plot_footprint_interactive(flx, grid, title=None, xlim=None, ylim=None):
    """Create an interactive Plotly heatmap of the footprint.

    Requires the optional ``plotly`` package.

    Parameters
    ----------
    flx : ndarray (ny, nx)
        Footprint field.
    grid : tuple (X, Y, Z)
        Coordinate arrays.
    title : str, optional
    xlim : tuple of float, optional
        (xmin, xmax) axis range.  Useful for excluding halo padding.
    ylim : tuple of float, optional
        (ymin, ymax) axis range.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    go = optional_import("plotly.graph_objects", "plotly")

    X, Y, _ = grid
    x = X[0, :] if X.ndim == 2 else X
    y = Y[:, 0] if Y.ndim == 2 else Y

    if xlim is not None:
        xmask = (x >= xlim[0]) & (x <= xlim[1])
        x, flx = x[xmask], flx[:, xmask]
    if ylim is not None:
        ymask = (y >= ylim[0]) & (y <= ylim[1])
        y, flx = y[ymask], flx[ymask, :]

    fig = go.Figure(data=go.Heatmap(
        z=flx,
        x=x,
        y=y,
        colorscale="RdYlBu_r",
        colorbar=dict(title="Footprint [m\u207b\u00b2]"),
    ))
    x_range = float(x.max() - x.min()) or 1.0
    y_range = float(y.max() - y.min()) or 1.0
    base_width = 650
    margin_px = 150
    fig_height = int(base_width * (y_range / x_range) + margin_px)

    fig.update_layout(
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        title=title or "BLDFM Footprint",
        yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
        width=base_width + margin_px,
        height=fig_height,
    )
    return fig
