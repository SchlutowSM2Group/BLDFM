"""Core footprint plots and percentile contour computation."""

import sys

import numpy as np
import matplotlib.pyplot as plt

from ._common import ensure_ax, optional_import


def extract_percentile_contour(flx, grid, pct=0.8):
    """Find the contour level that encloses *pct* of the cumulative footprint.

    Parameters
    ----------
    flx : ndarray (ny, nx)
        2-D footprint field (non-negative).
    grid : tuple (X, Y, Z)
        Coordinate arrays from the solver.
    pct : float
        Fraction of the total footprint to enclose (0-1).

    Returns
    -------
    level : float
        Contour level value.
    area : float
        Approximate area [m^2] enclosed by that contour.
    """
    X, Y, _ = grid
    dx = np.abs(X[0, 1] - X[0, 0]) if X.ndim == 2 else np.abs(X[1] - X[0])
    dy = np.abs(Y[1, 0] - Y[0, 0]) if Y.ndim == 2 else np.abs(Y[1] - Y[0])
    cell_area = dx * dy

    flat = flx.ravel()
    idx = np.argsort(flat)[::-1]
    sorted_vals = flat[idx]
    cumsum = np.cumsum(sorted_vals) * cell_area
    total = cumsum[-1]

    target = pct * total
    k = np.searchsorted(cumsum, target)
    level = sorted_vals[min(k, len(sorted_vals) - 1)]
    area = (k + 1) * cell_area

    return float(level), float(area)


def plot_footprint_field(flx, grid, ax=None, contour_pcts=None,
                         cmap="RdYlBu_r", title=None, **pcolormesh_kw):
    """Plot a 2-D footprint field with optional percentile contours.

    Parameters
    ----------
    flx : ndarray (ny, nx)
        Footprint (or concentration) field.
    grid : tuple (X, Y, Z)
        Coordinate arrays from the solver.
    ax : matplotlib Axes, optional
    contour_pcts : list of float, optional
        Percentile fractions to overlay as contours (e.g. [0.5, 0.8]).
    cmap : str
        Colourmap name.
    title : str, optional
    **pcolormesh_kw
        Forwarded to ``ax.pcolormesh``.

    Returns
    -------
    ax : matplotlib Axes
    """
    X, Y, _ = grid
    ax = ensure_ax(ax)

    pm = ax.pcolormesh(X, Y, flx, cmap=cmap, shading="auto", **pcolormesh_kw)
    ax.figure.colorbar(pm, ax=ax, label="Footprint [m$^{-2}$]")

    if contour_pcts is not None:
        levels = []
        pct_labels = {}
        for p in sorted(contour_pcts):
            lvl, _ = extract_percentile_contour(flx, grid, p)
            levels.append(lvl)
            pct_labels[lvl] = f"{int(p * 100)}%"
        cs = ax.contour(X, Y, flx, levels=sorted(levels), colors="k",
                        linewidths=0.8, linestyles="--")
        ax.clabel(cs, fmt=lambda x: pct_labels.get(x, f"{x:.2e}"),
                  fontsize=8, inline=True)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def plot_footprint_on_map(flx, grid, config, tower=None, ax=None,
                          contour_pcts=None, tile_source=None,
                          land_cover=False, land_cover_size=(512, 512),
                          alpha=0.5, cmap="RdYlBu_r", title=None):
    """Overlay footprint contours and tower marker(s) on map tiles.

    Requires ``contextily`` (default) or ``owslib`` (when *land_cover=True*).

    Parameters
    ----------
    flx : ndarray (ny, nx)
        Footprint field.
    grid : tuple (X, Y, Z)
        Coordinate arrays (local metres).
    config : BLDFMConfig
        Configuration (for ref_lat/ref_lon and tower metadata).
    tower : TowerConfig, optional
        Specific tower to highlight.  If None, all towers are plotted.
    ax : matplotlib Axes, optional
    contour_pcts : list of float, optional
        Percentile contours to draw (default [0.5, 0.8]).
    tile_source : contextily tile source, optional
        Defaults to OpenStreetMap.  Ignored when *land_cover=True* unless
        explicitly provided (in which case both layers are rendered).
    land_cover : bool
        If True, use ESA WorldCover 2021 as the basemap instead of OSM
        tiles.  Requires the optional ``owslib`` package.
    land_cover_size : tuple of int
        Pixel dimensions (width, height) for the WMS request (default
        (512, 512)).  Increase for higher resolution.
    alpha : float
        Transparency for footprint fill.
    cmap : str
        Colourmap for the footprint fill.
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    from . import _geo

    ref_lat = config.domain.ref_lat
    ref_lon = config.domain.ref_lon
    if ref_lat is None or ref_lon is None:
        raise ValueError("config.domain must have ref_lat and ref_lon for map plots")

    X, Y, _ = grid
    lats, lons = _geo.xy_to_latlon(X, Y, ref_lat, ref_lon)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if contour_pcts is None:
        contour_pcts = [0.5, 0.8]

    # Filled contour of the footprint
    levels_fill = []
    pct_labels = {}
    for p in sorted(contour_pcts):
        lvl, _ = extract_percentile_contour(flx, grid, p)
        levels_fill.append(lvl)
        pct_labels[lvl] = f"{int(p * 100)}%"
    levels_fill = sorted(levels_fill)

    ax.contourf(lons, lats, flx, levels=20, cmap=cmap, alpha=alpha, zorder=2)
    cs = ax.contour(lons, lats, flx, levels=levels_fill, colors="k",
                    linewidths=1.0, linestyles="--", zorder=3)
    ax.clabel(cs, fmt=lambda x: pct_labels.get(x, f"{x:.2e}"),
              fontsize=8, inline=True)

    # Tower markers
    towers_to_plot = [tower] if tower is not None else config.towers
    for t in towers_to_plot:
        ax.plot(t.lon, t.lat, "k^", markersize=10, markeredgecolor="white",
                markeredgewidth=1.5, zorder=4)
        ax.annotate(t.name, (t.lon, t.lat), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight="bold",
                    color="black", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Basemap layer
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    if land_cover:
        bbox = (lon_min, lat_min, lon_max, lat_max)
        # Access through package module for monkeypatch compatibility
        _fetch = sys.modules[__package__]._fetch_land_cover
        lc_img, lc_extent = _fetch(bbox, size=land_cover_size)
        ax.imshow(lc_img, extent=lc_extent, origin="upper",
                  aspect="auto", zorder=0)
        _geo.land_cover_legend(ax)

        if tile_source is not None:
            ctx = optional_import("contextily", "contextily")
            ctx.add_basemap(ax, crs="EPSG:4326", source=tile_source,
                            zorder=1)
    else:
        ctx = optional_import("contextily", "contextily")
        if tile_source is None:
            tile_source = ctx.providers.OpenStreetMap.Mapnik
        ctx.add_basemap(ax, crs="EPSG:4326", source=tile_source, zorder=1)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(useOffset=False, style="plain")
    if title:
        ax.set_title(title)
    return ax
