import numpy as np

from numpy.fft import fftshift, ifftshift, fftfreq
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from numba import set_num_threads

from .utils import get_logger
from .utils import parallelize
from bldfm import config

logger = get_logger(__name__.split("bldfm.")[-1])
logger.info("Loaded solver module for steady-state transport solver.")


def steady_state_transport_solver(
    srf_flx,
    z,
    profiles,
    domain,
    n,
    modes=(512, 512),
    meas_pt=(0.0, 0.0),
    srf_bg_conc=0.0,
    footprint=False,
    analytic=False,
    halo=-1e9,
):
    """
    Solves the steady-state advection-diffusion equation for a concentration
    with flux boundary condition given vertical profiles of wind and eddy diffusivity
    using the Fourier, linear shooting, and semi-implicit Euler methods.

    Parameters
    ----------
    srf_flx : ndarray of float
        2D field of surface kinematic flux at z=z0 [m/s].
    z : ndarray of float
        1D array of vertical grid points from z0 to zm [m].
    profiles : tuple of ndarray
        Tuple containing 1D arrays of vertical profiles of zonal wind, meridional wind [m/s],
        and eddy diffusivity [m²/s].
    domain : tuple of float
        Tuple containing domain sizes (xmax, ymax) [m].
    modes : tuple of int, optional
        Tuple containing the number of zonal and meridional Fourier modes (nlx, nly).
        Default is (512, 512).
    meas_pt : tuple of float, optional
        Coordinates of the measurement point (xm, ym) [m] relative to `srf_flx`,
        where the origin is at `srf_flx[0, 0]`. Default is (0.0, 0.0).
    srf_bg_conc : float, optional
        Surface background concentration at z=z0 [scalar_unit]. Default is 0.0.
    footprint : bool, optional
        If True, activates footprints (Green's function) for output. Default is False.
    analytic : bool, optional
        If True, uses the analytic solution for constant wind and eddy diffusivity.
        Default is False.
    halo : float, optional
        Width of the zero-flux halo around the domain [m]. Default is -1e9, which sets
        the halo to `max(xmax, ymax)`.

    Returns
    -------
    srf_conc : ndarray of float
        2D field of surface concentrations at z=z0.
    bg_conc : float
        Background concentration at z=zm.
    conc : ndarray of float
        2D field of concentration at z=zm or Green's function.
    flx : ndarray of float
        2D field of kinematic flux at z=zm or footprint.
    """

    q0 = srf_flx
    p000 = srf_bg_conc
    u, v, K = profiles
    xmx, ymx = domain
    nlx, nly = modes
    xm, ym = meas_pt

    # number of grid cells
    ny, nx = q0.shape

    # grid increments
    dx, dy = xmx / nx, ymx / ny

    if halo < 0.0:
        halo = max(xmx, ymx)

    # pad width
    px = int(halo / dx)
    py = int(halo / dy)

    # construct zero-flux halo by padding
    q0 = np.pad(q0, ((py, py), (px, px)), mode="constant", constant_values=0.0)

    # extent domain
    nx = nx + 2 * px
    ny = ny + 2 * py

    if (nlx > nx) or (nly > ny):
        logger.info(
            "Warning: Number of Fourier modes must not exeed number of grid cells."
        )
        logger.info("Setting both equal.")
        nlx, nly = nx, ny

    # Deltas for truncated Fourier transform
    dlx, dly = (nx - nlx) // 2, (ny - nly) // 2

    if footprint:
        # Fourier trafo of delta distribution
        tfftq0 = np.ones((nly, nlx), dtype=complex) / nx / ny
    else:
        fftq0 = fft2(q0, norm="forward")  # fft of source

        # shift zero wave number to center of array
        fftq0 = fftshift(fftq0)

        # truncate fourier series by removing higher-frequency components
        tfftq0 = fftq0[dly : ny - dly, dlx : nx - dlx]

        # unshift
        tfftq0 = ifftshift(tfftq0)

    # Fourier summation index
    ilx = fftfreq(nlx, d=1.0 / nlx)
    ily = fftfreq(nly, d=1.0 / nly)

    # define truncated zonal and meridional wavenumbers
    lx = 2.0 * np.pi / dx / nx * ilx
    ly = 2.0 * np.pi / dy / ny * ily

    Lx, Ly = np.meshgrid(lx, ly)

    dz = np.diff(z)
    nz = len(z)

    # define mask to seperate degenerated and non-degenerated system
    msk = np.ones((nly, nlx), dtype=bool)  # all n and m not equal 0
    msk[0, 0] = False

    one = np.ones((nly, nlx), dtype=complex)[msk]
    zero = np.zeros((nly, nlx), dtype=complex)[msk]

    Kinv = 1.0 / K[nz - 1]

    # Eigenvalue determining solution for z > zmx
    eigval = np.sqrt(
        Lx[msk] ** 2
        + Ly[msk] ** 2
        + 1j * u[nz - 1] * Kinv * Lx[msk]
        + 1j * v[nz - 1] * Kinv * Ly[msk]
    )

    # initialization
    tfftp0 = np.zeros((nly, nlx), dtype=complex)
    tfftp = np.zeros((nly, nlx), dtype=complex)
    tfftq = np.zeros((nly, nlx), dtype=complex)

    tfftp0[0, 0] = p000
    tfftq[0, 0] = tfftq0[0, 0]  # conservation by design

    if analytic:

        # constant profiles solution
        # for validation purposes
        h = z[n] - z[0]

        tfftp0[msk] = tfftq0[msk] * Kinv / eigval
        tfftp[0, 0] = p000 - tfftq0[0, 0] * Kinv * h
        tfftq[msk] = tfftq0[msk] * np.exp(-eigval * h)
        tfftp[msk] = tfftq[msk] * Kinv / eigval

    else:

        # solve non-degenerated problem for (n,m) =/= (0,0)
        # by linear shooting method
        # use two auxillary initial value problems
        if config.NUM_THREADS > 1:
            logger.info("BLDFM runnning in parallel mode.")
            set_num_threads(config.NUM_THREADS)

        tfftp1, tfftq1, tfftpm1, tfftqm1 = ivp_solver(
            (one, zero), profiles, z, n, Lx[msk], Ly[msk]
        )

        tfftp2, tfftq2, tfftpm2, tfftqm2 = ivp_solver(
            (zero, tfftq0[msk]), profiles, z, n, Lx[msk], Ly[msk]
        )

        alpha = -(tfftq2 - K[nz - 1] * eigval * tfftp2) / (
            tfftq1 - K[nz - 1] * eigval * tfftp1
        )

        # linear combination of the two solution of the IVP
        tfftp0[msk] = alpha
        tfftp[msk] = alpha * tfftpm1 + tfftpm2
        tfftq[msk] = alpha * tfftqm1 + tfftqm2

        # solve degenerated problem for (n,m) =  (0,0)
        # with Euler forward method
        tfftp[0, 0] = p000
        for i in range(n):
            tfftp[0, 0] = tfftp[0, 0] - tfftq0[0, 0] / K[i] * dz[i]

    # shift green function in Fourier space to measurement point
    if footprint:
        tfftp0 = tfftp0 * np.exp(1j * (Lx * (xm + halo) + Ly * (ym + halo)))
        tfftp = tfftp * np.exp(1j * (Lx * (xm + halo) + Ly * (ym + halo)))
        tfftq = tfftq * np.exp(1j * (Lx * (xm + halo) + Ly * (ym + halo)))
    # shift such that xm, ym are in the middle of the domain
    elif xm**2 + ym**2 > 0.0:
        tfftp0 = tfftp0 * np.exp(1j * (Lx * (xm - xmx / 2) + Ly * (ym - ymx / 2)))
        tfftp = tfftp * np.exp(1j * (Lx * (xm - xmx / 2) + Ly * (ym - ymx / 2)))
        tfftq = tfftq * np.exp(1j * (Lx * (xm - xmx / 2) + Ly * (ym - ymx / 2)))

    # shift zero to center
    tfftp0 = fftshift(tfftp0)
    tfftp = fftshift(tfftp)
    tfftq = fftshift(tfftq)

    # untruncate
    fftp0 = np.pad(
        tfftp0, ((dly, dly), (dlx, dlx)), mode="constant", constant_values=0.0
    )
    fftp = np.pad(tfftp, ((dly, dly), (dlx, dlx)), mode="constant", constant_values=0.0)
    fftq = np.pad(tfftq, ((dly, dly), (dlx, dlx)), mode="constant", constant_values=0.0)

    # unshift
    fftp0 = ifftshift(fftp0)
    fftp = ifftshift(fftp)
    fftq = ifftshift(fftq)

    if footprint:
        # use fft to reverse sign, make green's function to footprint
        p0 = fft2(fftp0, norm="backward").real  # concentration
        p = fft2(fftp, norm="backward").real  # concentration
        q = fft2(fftq, norm="backward").real  # kinematic flux
    else:
        # use ifft as usual
        p0 = ifft2(fftp0, norm="forward").real  # concentration
        p = ifft2(fftp, norm="forward").real  # concentration
        q = ifft2(fftq, norm="forward").real  # kinematic flux

    srf_conc = p0[py : ny - py, px : nx - px]
    bg_conc = fftp[0, 0].real
    conc = p[py : ny - py, px : nx - px]
    flx = q[py : ny - py, px : nx - px]

    return srf_conc, bg_conc, conc, flx


@parallelize
def ivp_solver(fftpq, profiles, z, n, Lx, Ly):
    """
    Solves the initial value problem resulting from the discretization of the
    steady-state advection-diffusion equation using the Fast Fourier Transform.

    Parameters
    ----------
    fftpq : tuple of ndarray
        Tuple containing the initial Fourier-transformed pressure and flux fields (fftp0, fftq0).
    profiles : tuple of ndarray
        Tuple containing 1D arrays of vertical profiles of zonal wind, meridional wind [m/s],
        and eddy diffusivity [m²/s].
    z : ndarray of float
        1D array of vertical grid points from z0 to zm [m].
    Lx : ndarray of float
        2D array of zonal wavenumbers.
    Ly : ndarray of float
        2D array of meridional wavenumbers.

    Returns
    -------
    fftp : ndarray of complex
        Fourier-transformed pressure field after solving the IVP.
    fftq : ndarray of complex
        Fourier-transformed flux field after solving the IVP.
    """

    fftp0, fftq0 = fftpq
    u, v, K = profiles

    fftp, fftq = np.copy(fftp0), np.copy(fftq0)

    nz = len(z)
    dz = np.diff(z)

    for i in range(nz - 1):

        Ti = -K[i] * (Lx**2 + Ly**2) - 1j * u[i] * Lx - 1j * v[i] * Ly
        Kinv = 1.0 / K[i]
        dzi = dz[i]

        a = 1.0 - 0.5 * Kinv * Ti * dzi**2
        b = -Kinv * dzi - 1.0 / 6.0 * Kinv**2 * Ti * dzi**3
        c = Ti * dzi - 1.0 / 6.0 * Kinv * Ti**2 * dzi**3
        d = 1.0 - 0.5 * Kinv * Ti * dzi**2

        dum = a * fftp + b * fftq
        fftq = c * fftp + d * fftq
        fftp = dum

        if i == n - 1:
            fftpm, fftqm = fftp, fftq

    return fftp, fftq, fftpm, fftqm
