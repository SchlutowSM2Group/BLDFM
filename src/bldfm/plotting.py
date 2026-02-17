"""
Plotting module for BLDFM footprint visualisation.

All matplotlib-based functions accept an optional ``ax`` parameter for
composability.  Optional dependencies (contextily, windrose, plotly) are
imported lazily and raise helpful messages when missing.
"""

import numpy as np
import matplotlib.pyplot as plt

from .utils import get_logger

logger = get_logger("plotting")


# ---------------------------------------------------------------------------
# ESA WorldCover 2021 configuration
# ---------------------------------------------------------------------------

_WORLDCOVER_WMS_URL = "https://services.terrascope.be/wms/v2"
_WORLDCOVER_LAYER = "WORLDCOVER_2021_MAP"
_WORLDCOVER_CLASSES = {
    10: ("#006400", "Tree cover"),
    20: ("#ffbb22", "Shrubland"),
    30: ("#ffff4c", "Grassland"),
    40: ("#f096ff", "Cropland"),
    50: ("#fa0000", "Built-up"),
    60: ("#b4b4b4", "Bare / sparse vegetation"),
    70: ("#f0f0f0", "Snow and ice"),
    80: ("#0064c8", "Permanent water bodies"),
    90: ("#0096a0", "Herbaceous wetland"),
    95: ("#00cf75", "Mangroves"),
    100: ("#fae6a0", "Moss and lichen"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xy_to_latlon(x, y, ref_lat, ref_lon):
    """Inverse of latlon_to_xy: local metres back to decimal degrees.

    Accepts scalars or arrays (vectorized).
    """
    R = 6_371_000.0
    lats = ref_lat + np.degrees(y / R)
    lons = ref_lon + np.degrees(x / (R * np.cos(np.radians(ref_lat))))
    return lats, lons


def _latlon_to_webmercator(lon, lat):
    """Convert EPSG:4326 (lon, lat) to EPSG:3857 (x, y) Web Mercator."""
    x = lon * 20037508.34 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / np.pi * 20037508.34
    return float(x), float(y)


def _fetch_land_cover(bbox_latlon, size=(512, 512), timeout=10):
    """Fetch ESA WorldCover 2021 raster for a lat/lon bounding box.

    Parameters
    ----------
    bbox_latlon : tuple (lon_min, lat_min, lon_max, lat_max)
        Bounding box in EPSG:4326.
    size : tuple (width, height)
        Pixel dimensions of the returned image.
    timeout : int
        WMS request timeout in seconds.

    Returns
    -------
    img : ndarray
        RGBA image array.
    extent : tuple (lon_min, lon_max, lat_min, lat_max)
        Extent suitable for ``ax.imshow()`` in lon/lat space.
    """
    try:
        from owslib.wms import WebMapService
    except ImportError:
        raise ImportError(
            "owslib is required for land cover overlays. "
            "Install it with: pip install owslib"
        )
    import io

    # Terrascope WMS only supports EPSG:3857 — convert bbox
    lon_min, lat_min, lon_max, lat_max = bbox_latlon
    x_min, y_min = _latlon_to_webmercator(lon_min, lat_min)
    x_max, y_max = _latlon_to_webmercator(lon_max, lat_max)

    wms = WebMapService(_WORLDCOVER_WMS_URL, version="1.1.1", timeout=timeout)
    response = wms.getmap(
        layers=[_WORLDCOVER_LAYER],
        srs="EPSG:3857",
        bbox=(x_min, y_min, x_max, y_max),
        size=size,
        format="image/png",
        transparent=True,
    )
    img = plt.imread(io.BytesIO(response.read()))
    # Return extent in lat/lon for plotting in geographic coordinates
    extent = (lon_min, lon_max, lat_min, lat_max)
    return img, extent


def _land_cover_legend(ax, classes=None):
    """Add a categorical legend for ESA WorldCover classes.

    Parameters
    ----------
    ax : matplotlib Axes
    classes : list of int, optional
        Subset of class values to show.  If None, shows all 11.
    """
    from matplotlib.patches import Patch

    if classes is None:
        classes = sorted(_WORLDCOVER_CLASSES.keys())

    patches = [
        Patch(facecolor=_WORLDCOVER_CLASSES[c][0],
              label=_WORLDCOVER_CLASSES[c][1])
        for c in classes if c in _WORLDCOVER_CLASSES
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7,
              framealpha=0.8, title="Land cover", title_fontsize=8)


def extract_percentile_contour(flx, grid, pct=0.8):
    """Find the contour level that encloses *pct* of the cumulative footprint.

    Parameters
    ----------
    flx : ndarray (ny, nx)
        2-D footprint field (non-negative).
    grid : tuple (X, Y, Z)
        Coordinate arrays from the solver.
    pct : float
        Fraction of the total footprint to enclose (0–1).

    Returns
    -------
    level : float
        Contour level value.
    area : float
        Approximate area [m²] enclosed by that contour.
    """
    X, Y, _ = grid
    dx = np.abs(X[0, 1] - X[0, 0]) if X.ndim == 2 else np.abs(X[1] - X[0])
    dy = np.abs(Y[1, 0] - Y[0, 0]) if Y.ndim == 2 else np.abs(Y[1] - Y[0])
    cell_area = dx * dy

    flat = flx.ravel()
    # Sort descending (peak first)
    idx = np.argsort(flat)[::-1]
    sorted_vals = flat[idx]
    cumsum = np.cumsum(sorted_vals) * cell_area
    total = cumsum[-1]

    # Find threshold where cumulative sum reaches pct * total
    target = pct * total
    k = np.searchsorted(cumsum, target)
    level = sorted_vals[min(k, len(sorted_vals) - 1)]
    area = (k + 1) * cell_area

    return float(level), float(area)


# ---------------------------------------------------------------------------
# Core plot functions
# ---------------------------------------------------------------------------

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
    if ax is None:
        _, ax = plt.subplots()

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
    ref_lat = config.domain.ref_lat
    ref_lon = config.domain.ref_lon
    if ref_lat is None or ref_lon is None:
        raise ValueError("config.domain must have ref_lat and ref_lon for map plots")

    X, Y, _ = grid
    # Convert grid to lat/lon (vectorized)
    lats, lons = _xy_to_latlon(X, Y, ref_lat, ref_lon)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if contour_pcts is None:
        contour_pcts = [0.5, 0.8]

    # --- Plot data first (sets axes extent for contextily) ---

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

    # --- Basemap layer (added after data so axes extent is correct) ---
    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    if land_cover:
        bbox = (lon_min, lat_min, lon_max, lat_max)
        lc_img, lc_extent = _fetch_land_cover(bbox, size=land_cover_size)
        ax.imshow(lc_img, extent=lc_extent, origin="upper",
                  aspect="auto", zorder=0)
        _land_cover_legend(ax)

        # If user also explicitly passed tile_source, add that too
        if tile_source is not None:
            try:
                import contextily as ctx
            except ImportError:
                raise ImportError(
                    "contextily is required when combining land_cover with "
                    "tile_source. Install it with: pip install contextily"
                )
            ctx.add_basemap(ax, crs="EPSG:4326", source=tile_source,
                            zorder=1)
    else:
        try:
            import contextily as ctx
        except ImportError:
            raise ImportError(
                "contextily is required for map plots. "
                "Install it with: pip install contextily"
            )
        if tile_source is None:
            tile_source = ctx.providers.OpenStreetMap.Mapnik
        ctx.add_basemap(ax, crs="EPSG:4326", source=tile_source, zorder=1)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(useOffset=False, style="plain")
    if title:
        ax.set_title(title)
    return ax


def plot_wind_rose(wind_speed, wind_dir, ax=None, bins=None, title=None):
    """Plot a wind rose from meteorological data.

    Requires the optional ``windrose`` package.

    Parameters
    ----------
    wind_speed : array-like
        Wind speed values.
    wind_dir : array-like
        Wind direction values (degrees, meteorological convention).
    ax : WindroseAxes, optional
    bins : array-like, optional
        Speed bins.
    title : str, optional

    Returns
    -------
    ax : WindroseAxes
    """
    try:
        from windrose import WindroseAxes
    except ImportError:
        raise ImportError(
            "windrose is required for wind rose plots. "
            "Install it with: pip install windrose"
        )

    ws = np.asarray(wind_speed)
    wd = np.asarray(wind_dir)

    if ax is None:
        fig = plt.figure()
        ax = WindroseAxes.from_ax(fig=fig)

    if bins is None:
        bins = np.linspace(0, np.ceil(ws.max()), 6)

    ax.bar(wd, ws, bins=bins, normed=True, opening=0.8, edgecolor="white")
    ax.set_legend(title="Wind speed [m/s]")
    if title:
        ax.set_title(title)
    return ax


def plot_footprint_timeseries(results, grid, pcts=None, ax=None, title=None):
    """Plot temporal evolution of footprint extent.

    Parameters
    ----------
    results : list of dict
        Output of run_bldfm_timeseries (list of result dicts for one tower).
    grid : tuple (X, Y, Z)
        Coordinate arrays (from first result, assumed constant).
    pcts : list of float, optional
        Percentile fractions to track (default [0.5, 0.8]).
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    if pcts is None:
        pcts = [0.5, 0.8]

    if ax is None:
        _, ax = plt.subplots()

    timestamps = [r["timestamp"] for r in results]
    x_pos = np.arange(len(timestamps))

    for pct in pcts:
        areas = []
        for r in results:
            _, area = extract_percentile_contour(r["flx"], grid, pct)
            areas.append(area / 1e6)  # m² -> km²
        ax.plot(x_pos, areas, "o-", label=f"{int(pct*100)}%")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(t) for t in timestamps], rotation=45, ha="right")
    ax.set_ylabel("Footprint area [km$^2$]")
    ax.set_xlabel("Timestamp")
    ax.legend()
    if title:
        ax.set_title(title)
    ax.figure.tight_layout()
    return ax


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
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plots. "
            "Install it with: pip install plotly"
        )

    X, Y, _ = grid
    x = X[0, :] if X.ndim == 2 else X
    y = Y[:, 0] if Y.ndim == 2 else Y

    # Crop data to physical domain (removes halo padding)
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
        colorbar=dict(title="Footprint [m⁻²]"),
    ))
    # Size figure to match data aspect ratio
    x_range = float(x.max() - x.min()) or 1.0
    y_range = float(y.max() - y.min()) or 1.0
    base_width = 650
    margin_px = 150  # title, labels, colorbar
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


def plot_footprint_comparison(fields, grids, labels, meas_pt=None, n_levels=6,
                               vmin=None, vmax=None, cmap="turbo", figsize=None,
                               title=None):
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

    Example
    -------
    >>> fields = [flx1, flx2, flx3]
    >>> grids = [(X1, Y1), (X2, Y2), (X3, Y3)]
    >>> labels = ["BLDFM", "BLDFM-SP", "KM01"]
    >>> fig, axes = plot_footprint_comparison(fields, grids, labels, meas_pt=(25, 20))
    """
    n = len(fields)
    if figsize is None:
        figsize = (8, 8)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True, layout="constrained")
    if n == 1:
        axes = [axes]

    # Compute vmin/vmax
    if vmax is None:
        vmax = max(np.max(f) for f in fields)
    if vmin is None:
        vmin = 1e-5

    levels = np.linspace(vmin, vmax, n_levels, endpoint=False)

    for i, (flx, grid, label, ax) in enumerate(zip(fields, grids, labels, axes)):
        X, Y = grid
        plot = ax.contour(X, Y, flx, levels, cmap=cmap, vmin=vmin, vmax=vmax,
                          linewidths=4.0)
        ax.set_title(label)
        ax.set_xlabel("x [m]")
        if i == 0:
            ax.set_ylabel("y [m]")

        if meas_pt is not None:
            ax.scatter(meas_pt[0], meas_pt[1], zorder=5, marker="*",
                       color="red", s=300)

    # Shared colorbar at bottom
    cbar = fig.colorbar(plot, ax=axes, shrink=0.8, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("$m^{-2}$")

    if title:
        fig.suptitle(title)

    return fig, axes


def plot_field_comparison(fields, domain, src_pt=None, cmap="turbo", figsize=None):
    """2×2 panel comparison: conc, flux, and relative differences.

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

    Example
    -------
    >>> fields = {"conc": conc, "flx": flx, "conc_ref": conc_ana, "flx_ref": flx_ana}
    >>> fig, axes = plot_field_comparison(fields, domain=(200, 100), src_pt=(10, 10))
    """
    if figsize is None:
        figsize = (10, 6)

    conc = fields["conc"]
    flx = fields["flx"]
    conc_ref = fields["conc_ref"]
    flx_ref = fields["flx_ref"]

    # Compute relative differences
    diff_conc = (conc - conc_ref) / np.max(conc_ref)
    diff_flx = (flx - flx_ref) / np.max(flx_ref)

    extent = [0, domain[0], 0, domain[1]]
    shrink = 0.7

    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True,
                            layout="constrained")

    # Top-left: concentration
    plot = axs[0, 0].imshow(conc, origin="lower", cmap=cmap, extent=extent)
    axs[0, 0].set_title("Numerical concentration")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].xaxis.set_tick_params(labelbottom=False)
    cbar = fig.colorbar(plot, ax=axs[0, 0], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("a.u.")

    # Top-right: flux
    plot = axs[0, 1].imshow(flx, origin="lower", cmap=cmap, extent=extent)
    axs[0, 1].set_title("Numerical flux")
    axs[0, 1].xaxis.set_tick_params(labelbottom=False)
    axs[0, 1].yaxis.set_tick_params(labelleft=False)
    cbar = fig.colorbar(plot, ax=axs[0, 1], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("a.u. m/s")

    # Bottom-left: relative difference concentration
    plot = axs[1, 0].imshow(diff_conc, origin="lower", cmap=cmap, extent=extent)
    axs[1, 0].set_title("Relative difference to analytic concentration")
    axs[1, 0].set_xlabel("x [m]")
    axs[1, 0].set_ylabel("y [m]")
    cbar = fig.colorbar(plot, ax=axs[1, 0], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    # Bottom-right: relative difference flux
    plot = axs[1, 1].imshow(diff_flx, origin="lower", cmap=cmap, extent=extent)
    axs[1, 1].set_title("Relative difference to analytic flux")
    axs[1, 1].set_xlabel("x [m]")
    axs[1, 1].yaxis.set_tick_params(labelleft=False)
    cbar = fig.colorbar(plot, ax=axs[1, 1], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    # Mark source point on top panels
    if src_pt is not None:
        axs[0, 0].scatter(src_pt[0], src_pt[1], zorder=5, marker="*",
                          color="red", s=100)
        axs[0, 1].scatter(src_pt[0], src_pt[1], zorder=5, marker="*",
                          color="red", s=100)

    return fig, axs


def plot_convergence(grid_sizes, errors, fits=None, marker="o", ax=None,
                     xlabel="$h$ [m]", ylabel="Relative RMSE", title=None):
    """Log-log convergence plot.

    Parameters
    ----------
    grid_sizes : array-like
        Effective grid spacings (h).
    errors : array-like
        Error values (e.g., RMSE).
    fits : list of (func, params, label), optional
        Curve fits to overlay. Each element is a tuple:
        (function, parameter_dict, label_string).
        Example: (lambda h, a, b: a * h**b, {"a": 0.1, "b": 2.0}, "$O(h^2)$")
    marker : str
        Marker style for data points (default "o").
    ax : matplotlib Axes, optional
    xlabel : str
        X-axis label (default "$h$ [m]").
    ylabel : str
        Y-axis label (default "Relative RMSE").
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes

    Example
    -------
    >>> h = np.array([10, 5, 2.5, 1.25])
    >>> err = np.array([1e-2, 2.5e-3, 6.25e-4, 1.56e-4])
    >>> fits = [(lambda x, a: a * x**2, {"a": 1e-4}, "$O(h^2)$")]
    >>> ax = plot_convergence(h, err, fits=fits)
    """
    if ax is None:
        _, ax = plt.subplots()

    grid_sizes = np.asarray(grid_sizes)
    errors = np.asarray(errors)

    # Plot data
    ax.loglog(grid_sizes, errors, marker=marker, linestyle="-", label="Data")

    # Overlay fits
    if fits is not None:
        for func, params, label in fits:
            h_fit = np.linspace(grid_sizes.min(), grid_sizes.max(), 100)
            y_fit = func(h_fit, **params)
            ax.loglog(h_fit, y_fit, linestyle="--", label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    if title:
        ax.set_title(title)

    return ax


def plot_vertical_profiles(z_list, profiles_list, labels, meas_height=None,
                            figsize=None, title=None):
    """Multi-panel (1×2) profile plot: velocity and diffusivity vs height.

    Left panel: u vs z.
    Right panel: Kz vs z.

    Parameters
    ----------
    z_list : list of array-like
        List of z arrays (one per stability condition).
    profiles_list : list of tuple
        List of profile tuples (u, v, Kx, Ky, Kz), one per stability condition.
    labels : list of str
        Legend labels for each profile (e.g., "L = -10 m").
    meas_height : float, optional
        Measurement height to mark with a horizontal dashed line.
    figsize : tuple, optional
        Figure size (width, height). Default (10, 5).
    title : str, optional
        Overall figure title.

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes (1x2)

    Example
    -------
    >>> z1, profs1 = vertical_profiles(64, 10, wind=(0, -5), z0=0.5, mol=-10)
    >>> z2, profs2 = vertical_profiles(64, 10, wind=(0, -5), z0=0.5, mol=10)
    >>> fig, axes = plot_vertical_profiles([z1, z2], [profs1, profs2],
    ...                                     labels=["L = -10 m", "L = +10 m"],
    ...                                     meas_height=10)
    """
    if figsize is None:
        figsize = (10, 5)

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")

    # Generate colormap-based colors
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(z_list))]

    for i, (z, profs, label, color) in enumerate(zip(z_list, profiles_list, labels, colors)):
        u, v, Kx, Ky, Kz = profs

        # Left: u vs z
        axes[0].plot(u, z, marker="+", linestyle="-", label=label, color=color)

        # Right: Kz vs z
        axes[1].plot(Kz, z, marker="", linestyle="-", label=label, color=color)

    # Mark measurement height
    if meas_height is not None:
        for ax in axes:
            ax.axhline(meas_height, color="gray", linestyle="--", linewidth=1, zorder=0)

    axes[0].set_xlabel("u [m/s]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("Velocity profile")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Kz [m²/s]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title("Diffusivity profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    return fig, axes


def plot_vertical_slice(field, grid, slice_axis, slice_index, ax=None,
                        cmap="viridis", title=None, xlabel=None, ylabel=None):
    """2D slice from a 3D field.

    Parameters
    ----------
    field : ndarray (nz, ny, nx)
        3-D field to slice.
    grid : tuple (X, Y, Z)
        3-D coordinate arrays from the solver.
    slice_axis : str
        Axis to slice along: "x", "y", or "z".
    slice_index : int
        Index along the slice axis.
    ax : matplotlib Axes, optional
    cmap : str
        Colormap name (default "viridis").
    title : str, optional
    xlabel : str, optional
        X-axis label (auto-generated if None).
    ylabel : str, optional
        Y-axis label (auto-generated if None).

    Returns
    -------
    ax : matplotlib Axes

    Example
    -------
    >>> grid, conc, flx = steady_state_transport_solver(...)
    >>> # Y-slice at x-index 32
    >>> ax = plot_vertical_slice(conc, grid, slice_axis="x", slice_index=32)
    """
    X, Y, Z = grid

    if ax is None:
        _, ax = plt.subplots()

    if slice_axis == "y":
        # Slice at fixed y: plot X vs Z
        slice_data = field[:, slice_index, :]
        x_coord = X[:, slice_index, :] if X.ndim == 3 else X
        z_coord = Z[:, slice_index, :] if Z.ndim == 3 else Z
        pm = ax.pcolormesh(x_coord, z_coord, slice_data, cmap=cmap, shading="auto")
        if xlabel is None:
            xlabel = "x [m]"
        if ylabel is None:
            ylabel = "z [m]"

    elif slice_axis == "x":
        # Slice at fixed x: plot Y vs Z
        slice_data = field[:, :, slice_index]
        y_coord = Y[:, :, slice_index] if Y.ndim == 3 else Y
        z_coord = Z[:, :, slice_index] if Z.ndim == 3 else Z
        pm = ax.pcolormesh(y_coord, z_coord, slice_data, cmap=cmap, shading="auto")
        if xlabel is None:
            xlabel = "y [m]"
        if ylabel is None:
            ylabel = "z [m]"

    elif slice_axis == "z":
        # Slice at fixed z: plot X vs Y
        slice_data = field[slice_index, :, :]
        x_coord = X[slice_index, :, :] if X.ndim == 3 else X
        y_coord = Y[slice_index, :, :] if Y.ndim == 3 else Y
        pm = ax.pcolormesh(x_coord, y_coord, slice_data, cmap=cmap, shading="auto")
        if xlabel is None:
            xlabel = "x [m]"
        if ylabel is None:
            ylabel = "y [m]"

    else:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got {slice_axis}")

    ax.figure.colorbar(pm, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return ax
