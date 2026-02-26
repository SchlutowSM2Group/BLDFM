import logging
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
    """
    Set up logging configuration with customizable options.

    Parameters:
        level (str or int): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string (str): Custom format string for log messages
        log_file (str): Optional specific log file name (overrides auto_file)
        log_dir (str): Directory to store log files
        auto_file (bool): If True, automatically generate timestamped filename
        run_name (str): Optional run name to include in log filename
    """
    if level is None:
        level = logging.INFO

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create handlers - always include console
    handlers = [logging.StreamHandler()]

    # Add file handler with timestamped filename
    if auto_file and log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            log_file = f"bldfm_{run_name}_{timestamp}.log"
        else:
            log_file = f"bldfm_{timestamp}.log"

    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        full_log_path = log_path / log_file
        handlers.append(logging.FileHandler(full_log_path))

    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set specific logger for BLDFM
    logger = logging.getLogger("bldfm")
    logger.setLevel(level)

    if log_file:
        logger.info(f"BLDFM logging initialized - writing to: {log_path / log_file}")

    return logger


def get_logger(name=None):
    """Get a logger instance for the given module."""
    if name is None:
        return logging.getLogger("bldfm")
    return logging.getLogger(f"bldfm.{name}")


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

    For the transformed field, the contour at level R encloses the region
    where sum(f) = R.

    Parameters
    ----------
    f : ndarray
        Function values (e.g., flux footprint).
    g : ndarray
        Function defining level sets (often same as f).

    Returns
    -------
    g_rescaled : ndarray
        Transformed field where contour values equal cumulative contribution.
    """
    f_flat = f.ravel()
    g_flat = g.ravel()

    # sort by g descending
    order = np.argsort(g_flat)[::-1]
    f_sorted = f_flat[order]

    # cumulative sum as we lower threshold
    M_cum = np.cumsum(f_sorted)

    # shift so each point gets sum of f over {g > g[point]}
    M_shifted = np.zeros_like(M_cum)
    M_shifted[1:] = M_cum[:-1]

    # map back to original positions
    g_rescaled = np.empty_like(g_flat)
    g_rescaled[order] = M_shifted

    return g_rescaled.reshape(g.shape)


def source_area_contribution(flx):
    """Base function for contribution (isopleth) contours: g = flx.

    Parameters
    ----------
    flx : ndarray (ny, nx)
        Footprint field.

    Returns
    -------
    g : ndarray (ny, nx)
    """
    return flx.copy()


def source_area_circular(X, Y, meas_pt):
    """Base function for circular contours centered at measurement point.

    g = -(r^2), so contours are concentric circles.

    Parameters
    ----------
    X, Y : ndarray (ny, nx)
        Coordinate grids.
    meas_pt : tuple (xm, ym)
        Measurement (tower) location.

    Returns
    -------
    g : ndarray (ny, nx)
    """
    xm, ym = meas_pt
    return -((X - xm) ** 2 + (Y - ym) ** 2)


def source_area_upwind(X, Y, meas_pt, wind):
    """Base function for upwind distance-band contours.

    g = dot(wind_hat, r), where r is displacement from tower.
    Contours are lines perpendicular to the wind direction.

    Parameters
    ----------
    X, Y : ndarray (ny, nx)
        Coordinate grids.
    meas_pt : tuple (xm, ym)
        Measurement (tower) location.
    wind : tuple (u, v)
        Wind components (m/s).

    Returns
    -------
    g : ndarray (ny, nx)
    """
    xm, ym = meas_pt
    u, v = wind
    speed = np.sqrt(u**2 + v**2)
    u_hat, v_hat = u / speed, v / speed
    return u_hat * (X - xm) + v_hat * (Y - ym)


def source_area_crosswind(X, Y, meas_pt, wind):
    """Base function for crosswind ridge contours.

    g = -(perpendicular distance from wind axis)^2.
    Contours are symmetric ridges parallel to the wind direction.

    Parameters
    ----------
    X, Y : ndarray (ny, nx)
        Coordinate grids.
    meas_pt : tuple (xm, ym)
        Measurement (tower) location.
    wind : tuple (u, v)
        Wind components (m/s).

    Returns
    -------
    g : ndarray (ny, nx)
    """
    xm, ym = meas_pt
    u, v = wind
    speed = np.sqrt(u**2 + v**2)
    u_hat, v_hat = u / speed, v / speed
    return -((-v_hat * (X - xm) + u_hat * (Y - ym)) ** 2)


def source_area_sector(X, Y, meas_pt, wind):
    """Base function for angular sector contours from upwind axis.

    g = -abs(theta), where theta is angular deviation from upwind direction.
    Contours form pie-slice sectors centered on the upwind direction.

    Parameters
    ----------
    X, Y : ndarray (ny, nx)
        Coordinate grids.
    meas_pt : tuple (xm, ym)
        Measurement (tower) location.
    wind : tuple (u, v)
        Wind components (m/s).

    Returns
    -------
    g : ndarray (ny, nx)
    """
    xm, ym = meas_pt
    u, v = wind
    theta = np.arctan2(Y - ym, X - xm)
    theta_upwind = np.arctan2(-v, -u)
    theta_rel = theta - theta_upwind
    theta_rel = np.arctan2(np.sin(theta_rel), np.cos(theta_rel))
    return -np.abs(theta_rel)
