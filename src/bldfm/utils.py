import numpy as np
import numba

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
