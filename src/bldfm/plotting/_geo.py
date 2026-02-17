"""Geospatial coordinate transforms and WMS helpers for map-based plots."""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# ESA WorldCover 2021 configuration
# ---------------------------------------------------------------------------

WORLDCOVER_WMS_URL = "https://services.terrascope.be/wms/v2"
WORLDCOVER_LAYER = "WORLDCOVER_2021_MAP"
WORLDCOVER_CLASSES = {
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


def xy_to_latlon(x, y, ref_lat, ref_lon):
    """Inverse of latlon_to_xy: local metres back to decimal degrees.

    Accepts scalars or arrays (vectorized).
    """
    R = 6_371_000.0
    lats = ref_lat + np.degrees(y / R)
    lons = ref_lon + np.degrees(x / (R * np.cos(np.radians(ref_lat))))
    return lats, lons


def latlon_to_webmercator(lon, lat):
    """Convert EPSG:4326 (lon, lat) to EPSG:3857 (x, y) Web Mercator."""
    x = lon * 20037508.34 / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) / np.pi * 20037508.34
    return float(x), float(y)


def fetch_land_cover(bbox_latlon, size=(512, 512), timeout=10):
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
    x_min, y_min = latlon_to_webmercator(lon_min, lat_min)
    x_max, y_max = latlon_to_webmercator(lon_max, lat_max)

    wms = WebMapService(WORLDCOVER_WMS_URL, version="1.1.1", timeout=timeout)
    response = wms.getmap(
        layers=[WORLDCOVER_LAYER],
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


def land_cover_legend(ax, classes=None):
    """Add a categorical legend for ESA WorldCover classes.

    Parameters
    ----------
    ax : matplotlib Axes
    classes : list of int, optional
        Subset of class values to show.  If None, shows all 11.
    """
    from matplotlib.patches import Patch

    if classes is None:
        classes = sorted(WORLDCOVER_CLASSES.keys())

    patches = [
        Patch(facecolor=WORLDCOVER_CLASSES[c][0], label=WORLDCOVER_CLASSES[c][1])
        for c in classes
        if c in WORLDCOVER_CLASSES
    ]
    ax.legend(
        handles=patches,
        loc="lower left",
        fontsize=7,
        framealpha=0.8,
        title="Land cover",
        title_fontsize=8,
    )
