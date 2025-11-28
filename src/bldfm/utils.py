import logging
import numpy as np
import scipy.fft as fft
import os
import numba

from datetime import datetime
from pathlib import Path

from bldfm import config


def compute_wind_fields(u_rot, wind_dir):
    """
    Computes the zonal (u) and meridional (v) wind components from a rotated
    wind speed and direction.

    Parameters:
        u_rot (float): Rotated wind speed.
        wind_dir (float): Wind direction in degrees (clockwise from north).

    Returns:
        tuple: A tuple (u, v) where:
            - u (float): Zonal wind component (east-west).
            - v (float): Meridional wind component (north-south).
    """
    wind_dir = np.deg2rad(wind_dir)
    u = u_rot * np.sin(wind_dir)
    v = u_rot * np.cos(wind_dir)

    return u, v


# def point_source(nxy, domain, src_pt):
#    """
#    Generates a point source field in Fourier space and transforms it back
#    to the spatial domain.
#
#    Parameters:
#        nxy (tuple): Number of grid points in the x and y directions (nx, ny).
#        domain (tuple): Physical dimensions of the domain (xmax, ymax).
#        src_pt (tuple): Coordinates of the source point (xs, ys).
#
#    Returns:
#        numpy.ndarray: A 2D array representing the point source field in the spatial domain.
#    """
#    nx, ny = nxy
#    xmx, ymx = domain
#    xs, ys = src_pt
#
#    dx, dy = xmx / nx, ymx / ny
#
#    # Fourier summation index
#    ilx = fft.fftfreq(nx, d=1.0 / nx)
#    ily = fft.fftfreq(ny, d=1.0 / ny)
#
#    # define zonal and meridional wavenumbers
#    lx = 2.0 * np.pi / dx / nx * ilx
#    ly = 2.0 * np.pi / dy / ny * ily
#
#    Lx, Ly = np.meshgrid(lx, ly)
#
#    fftq0 = np.ones((ny, nx), dtype=np.complex128)
#
#    # shift to source point in Fourier space
#    fftq0 = fftq0 * np.exp(-1j * (Lx * xs + Ly * ys)) / nx / ny
#
#    # normalize
#    fftq0 = fftq0 / dx / dy
#
#    return fft.ifft2(fftq0, norm="forward").real


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
    def wrapper(*args, **kwargs):
        if config.NUM_THREADS > 1:
            return numba.jit(nopython=True, parallel=True, cache=True)(func)(
                *args, **kwargs
            )
        else:
            return numba.jit(nopython=True, cache=True)(func)(*args, **kwargs)

    return wrapper
