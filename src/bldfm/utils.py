import logging
import warnings

import numpy as np
import numba

from datetime import datetime
from pathlib import Path

from bldfm import config


def compute_wind_fields(u_rot, wind_dir):
    """
    Computes the zonal (u) and meridional (v) wind components from a rotated
    wind speed and direction using the meteorological convention.

    Parameters:
        u_rot (float): Rotated wind speed.
        wind_dir (float): Wind direction in degrees (meteorological convention:
            direction the wind is coming FROM, clockwise from north).
            0 = from north, 90 = from east, 180 = from south, 270 = from west.

    Returns:
        tuple: A tuple (u, v) where:
            - u (float): Zonal wind component (east-west, positive = eastward).
            - v (float): Meridional wind component (north-south, positive = northward).
    """
    wind_dir = np.deg2rad(wind_dir)
    u = -u_rot * np.sin(wind_dir)
    v = -u_rot * np.cos(wind_dir)

    return u, v


def ideal_source(nxy, domain, src_loc=None, shape="diamond"):
    """
    Creates a synthetic source field in the shape of a circle or diamond.
    Useful for testing purposes.

    Parameters:
        nxy (tuple): Number of grid points in the x and y directions (nx, ny).
        domain (tuple): Physical dimensions of the domain (xmax, ymax).
        shape (str): Shape of the source field. Options are "circle" or "diamond". Default is "diamond".

    Returns:
        numpy.ndarray: A 2D array representing the source field.
    """

    nx, ny = nxy
    xmx, ymx = domain
    dx = xmx / nx
    dy = ymx / ny

    if src_loc is None:
        # source in the middle of the domain
        src_loc = (xmx / 2, ymx / 2)

    xs, ys = src_loc

    x = np.linspace(0.0, xmx, nx)
    y = np.linspace(0.0, ymx, ny)

    X, Y = np.meshgrid(x, y)

    q0 = np.zeros([ny, nx])

    if shape == "diamond":
        R0 = xmx / 12
        R = np.abs(X - xs) + np.abs(Y - ys)
        q0 = np.where(R < R0, 1.0, 0.0)

    if shape == "circle":
        R0 = xmx / 12
        R = np.sqrt((X - xs) ** 2 + (Y - ys) ** 2)
        q0 = np.where(R < R0, 1.0, 0.0)

    if shape == "point":
        sig = 4.0 * dx
        Rsq = (X - xs) ** 2 + (Y - ys) ** 2
        q0 = np.exp(-Rsq / 2.0 / sig**2) / sig / np.sqrt(2.0 * np.pi)

    return q0


def point_measurement(f, g):
    """
    Computes the convolution of two 2D arrays evaluated at a specific point.

    Parameters:
        f (numpy.ndarray): First 2D array.
        g (numpy.ndarray): Second 2D array.

    Returns:
        float: The result of the convolution at the specified point.
    """

    return np.sum(f * g)


def setup_logging(
    level=None,
    format_string=None,
    log_file=None,
    log_dir="logs",
    auto_file=True,
    run_name=None,
):
    """Set up logging configuration.

    .. deprecated::
        Use ``abltk.logging.setup_logging(namespace="bldfm", ...)`` instead.
    """
    warnings.warn(
        "bldfm.utils.setup_logging is deprecated, "
        "use abltk.logging.setup_logging(namespace='bldfm', ...)",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.logging import setup_logging as _setup_logging

    return _setup_logging(
        namespace="bldfm",
        level=level,
        format_string=format_string,
        log_file=log_file,
        log_dir=log_dir,
        auto_file=auto_file,
        run_name=run_name,
    )


def get_logger(name=None):
    """Get a logger instance for the given module.

    .. deprecated::
        Use ``abltk.logging.get_logger(name, namespace="bldfm")`` instead.
    """
    warnings.warn(
        "bldfm.utils.get_logger is deprecated, "
        "use abltk.logging.get_logger(name, namespace='bldfm')",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.logging import get_logger as _get_logger

    return _get_logger(name=name, namespace="bldfm")


def parallelize(func):
    _compiled = {}

    def wrapper(*args, **kwargs):
        use_parallel = config.NUM_THREADS > 1
        if use_parallel not in _compiled:
            _compiled[use_parallel] = numba.jit(
                nopython=True, parallel=use_parallel, cache=True
            )(func)
        return _compiled[use_parallel](*args, **kwargs)

    return wrapper


def get_source_area(f, g):
    """Rescale g so contour levels represent cumulative contribution of f.

    .. deprecated::
        Use ``abltk.plotting.source_area.get_source_area`` instead.
    """
    warnings.warn(
        "bldfm.utils.get_source_area is deprecated, "
        "use abltk.plotting.source_area.get_source_area",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.source_area import get_source_area as _new

    return _new(f, g)


def source_area_contribution(flx):
    """Base function for contribution (isopleth) contours: g = flx.

    .. deprecated::
        Use ``abltk.plotting.source_area.source_area_contribution`` instead.
    """
    warnings.warn(
        "bldfm.utils.source_area_contribution is deprecated, "
        "use abltk.plotting.source_area.source_area_contribution",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.source_area import source_area_contribution as _new

    return _new(flx)


def source_area_circular(X, Y, meas_pt):
    """Base function for circular contours centered at measurement point.

    .. deprecated::
        Use ``abltk.plotting.source_area.source_area_circular`` instead.
    """
    warnings.warn(
        "bldfm.utils.source_area_circular is deprecated, "
        "use abltk.plotting.source_area.source_area_circular",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.source_area import source_area_circular as _new

    return _new(X, Y, meas_pt)


def source_area_upwind(X, Y, meas_pt, wind):
    """Base function for upwind distance-band contours.

    .. deprecated::
        Use ``abltk.plotting.source_area.source_area_upwind`` instead.
    """
    warnings.warn(
        "bldfm.utils.source_area_upwind is deprecated, "
        "use abltk.plotting.source_area.source_area_upwind",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.source_area import source_area_upwind as _new

    return _new(X, Y, meas_pt, wind)


def source_area_crosswind(X, Y, meas_pt, wind):
    """Base function for crosswind ridge contours.

    .. deprecated::
        Use ``abltk.plotting.source_area.source_area_crosswind`` instead.
    """
    warnings.warn(
        "bldfm.utils.source_area_crosswind is deprecated, "
        "use abltk.plotting.source_area.source_area_crosswind",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.source_area import source_area_crosswind as _new

    return _new(X, Y, meas_pt, wind)


def source_area_sector(X, Y, meas_pt, wind):
    """Base function for angular sector contours from upwind axis.

    .. deprecated::
        Use ``abltk.plotting.source_area.source_area_sector`` instead.
    """
    warnings.warn(
        "bldfm.utils.source_area_sector is deprecated, "
        "use abltk.plotting.source_area.source_area_sector",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.source_area import source_area_sector as _new

    return _new(X, Y, meas_pt, wind)
