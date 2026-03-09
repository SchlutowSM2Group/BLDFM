"""Geospatial coordinate transforms and WMS helpers for map-based plots.

.. deprecated::
    All functions and constants have moved to ``abltk.plotting.geo``.
    These wrappers will be removed in a future release.
"""

import warnings

from abltk.plotting.geo import (
    WORLDCOVER_WMS_URL,
    WORLDCOVER_LAYER,
    WORLDCOVER_CLASSES,
)


def xy_to_latlon(x, y, ref_lat, ref_lon):
    warnings.warn(
        "bldfm.plotting._geo.xy_to_latlon is deprecated, "
        "use abltk.plotting.geo.xy_to_latlon",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.geo import xy_to_latlon as _new

    return _new(x, y, ref_lat, ref_lon)


def latlon_to_webmercator(lon, lat):
    warnings.warn(
        "bldfm.plotting._geo.latlon_to_webmercator is deprecated, "
        "use abltk.plotting.geo.latlon_to_webmercator",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.geo import latlon_to_webmercator as _new

    return _new(lon, lat)


def fetch_land_cover(bbox_latlon, size=(512, 512), timeout=10):
    warnings.warn(
        "bldfm.plotting._geo.fetch_land_cover is deprecated, "
        "use abltk.plotting.geo.fetch_land_cover",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.geo import fetch_land_cover as _new

    return _new(bbox_latlon, size, timeout)


def land_cover_legend(ax, classes=None):
    warnings.warn(
        "bldfm.plotting._geo.land_cover_legend is deprecated, "
        "use abltk.plotting.geo.land_cover_legend",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.geo import land_cover_legend as _new

    return _new(ax, classes)
