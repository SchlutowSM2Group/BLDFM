import logging

import numpy as np
import scipy.fft as fft


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


def point_source(nxy, domain, src_pt):
    """
    Generates a point source field in Fourier space and transforms it back
    to the spatial domain.

    Parameters:
        nxy (tuple): Number of grid points in the x and y directions (nx, ny).
        domain (tuple): Physical dimensions of the domain (xmax, ymax).
        src_pt (tuple): Coordinates of the source point (xs, ys).

    Returns:
        numpy.ndarray: A 2D array representing the point source field in the spatial domain.
    """
    nx, ny = nxy
    xmx, ymx = domain
    xs, ys = src_pt

    dx, dy = xmx / nx, ymx / ny

    # Fourier summation index
    ilx = fft.fftfreq(nx, d=1.0 / nx)
    ily = fft.fftfreq(ny, d=1.0 / ny)

    # define zonal and meridional wavenumbers
    lx = 2.0 * np.pi / dx / nx * ilx
    ly = 2.0 * np.pi / dy / ny * ily

    Lx, Ly = np.meshgrid(lx, ly)

    fftq0 = np.ones((ny, nx), dtype=complex)

    # shift to source point in Fourier space
    fftq0 = fftq0 * np.exp(-1j * (Lx * xs + Ly * ys)) / nx / ny

    # normalize
    fftq0 = fftq0 / dx / dy

    return fft.ifft2(fftq0, norm="forward").real


def ideal_source(nxy, domain, shape="diamond"):
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

    x = np.linspace(0.0, xmx, nx)
    y = np.linspace(0.0, ymx, ny)

    X, Y = np.meshgrid(x, y)

    q0 = np.zeros([ny, nx])

    # Circular source
    # R = np.sqrt((X-xmx/2)**2 + (Y-ymx/2)**2)

    # Diamond source
    R = np.abs(X - xmx / 4) + np.abs(Y - ymx / 4)

    R0 = xmx / 12

    q0 = np.where(R < R0, 1.0, 0.0)

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


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )