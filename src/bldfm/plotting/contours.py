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
- **Sector contours** (g = -|theta|): angular sectors from upwind axis

See Also
--------
bldfm.utils.get_source_area : Rescale a field so contour levels represent
    cumulative contribution fractions.
"""

import numpy as np
import matplotlib.pyplot as plt

from ._common import ensure_ax


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
    grid : tuple (X, Y, Z)
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
    raise NotImplementedError(
        "Source area contour plotting is not yet implemented. "
        "See extension/source_area_contours branch for the upcoming feature."
    )
