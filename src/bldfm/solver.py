import numpy as np
import gc

from numpy.fft import fftshift, ifftshift, fftfreq
from .fft_manager import fft2, ifft2, get_fft_manager
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
    levels,
    modes=(512, 512),
    meas_pt=(0.0, 0.0),
    srf_bg_conc=0.0,
    footprint=False,
    analytic=False,
    halo=None,
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
    levels : float or ndarray of float
        Vertical level for output or optionally 1D array of output levels.
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
    z : ndarray of float
        Heights [m] at levels.
    conc : ndarray of float
        2D or 3D field of concentration at levels or Green's function.
    flx : ndarray of float
        2D or 3D field of kinematic flux at levels or footprint.
    """

    q0 = srf_flx
    p000 = srf_bg_conc
    u, v, K = profiles
    xmx, ymx = domain
    nlx, nly = modes
    xm, ym = meas_pt

    # Check if modes are even
    if (nlx % 2 > 0) or (nly % 2 > 0):
        raise Exception("modes must consist of even numbers.")

    # number of grid cells
    ny, nx = q0.shape
    nz = len(z)

    # grid increments
    dx, dy = xmx / nx, ymx / ny
    dz = np.diff(z)

    # Check output levels
    if np.ndim(levels) == 0:
        levels = [levels]

    nlvls = len(levels)

    # halo to deal with periodicity of FFT
    if halo is None:
        halo = max(xmx, ymx)

    # pad width
    px = int(halo / dx)
    py = int(halo / dy)

    # construct zero-flux halo by padding
    q0 = np.pad(q0, ((py, py), (px, px)), mode="constant", constant_values=0.0)

    # extent domain
    nxe = nx + 2 * px
    nye = ny + 2 * py

    if (nlx > nxe) or (nly > nye):
        logger.info(
            "Warning: Number of Fourier modes must not exeed number of grid cells."
        )
        logger.info("Setting both equal.")
        nlx, nly = nxe, nye

    # Deltas for truncated Fourier transform
    dlx, dly = (nxe - nlx) // 2, (nye - nly) // 2

    if footprint:
        # Fourier trafo of delta distribution
        tfftq0 = np.ones((nly, nlx), dtype=complex) / nxe / nye
    else:
        fftq0 = fft2(q0, norm="forward")  # fft of source

        # shift zero wave number to center of array
        fftq0 = fftshift(fftq0)

        # truncate fourier series by removing higher-frequency components
        tfftq0 = fftq0[dly : nye - dly, dlx : nxe - dlx]

        # unshift
        tfftq0 = ifftshift(tfftq0)

    # Fourier summation index
    ilx = fftfreq(nlx, d=1.0 / nlx)
    ily = fftfreq(nly, d=1.0 / nly)

    # define truncated zonal and meridional wavenumbers
    lx = 2.0 * np.pi / dx / nxe * ilx
    ly = 2.0 * np.pi / dy / nye * ily

    Lx, Ly = np.meshgrid(lx, ly)

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

    # initialization of output arrays
    tfftp = np.zeros((nlvls, nly, nlx), dtype=complex)
    tfftq = np.zeros((nlvls, nly, nlx), dtype=complex)

    tfftp[0, 0, 0] = p000
    tfftq[:, 0, 0] = tfftq0[0, 0]  # conservation by design

    if analytic:

        # constant profiles solution
        # for validation purposes
        h = z[levels] - z[0]

        tfftp[0, msk] = tfftq0[msk] * Kinv / eigval
        tfftp[:, 0, 0] = p000 - tfftq0[0, 0] * Kinv * h
        tfftq[:, msk] = tfftq0[msk] * np.exp(-eigval * h)
        tfftp[:, msk] = tfftq[:, msk] * Kinv / eigval

    else:

        # solve non-degenerated problem for (n,m) =/= (0,0)
        # by linear shooting method
        # use two auxillary initial value problems
        if config.NUM_THREADS > 1:
            logger.info("BLDFM runnning with Numba parallelization.")
            set_num_threads(config.NUM_THREADS)
            # Initialize FFT manager with thread count
            get_fft_manager(num_threads=config.NUM_THREADS)
        else:
            # Initialize FFT manager for single-threaded operation
            get_fft_manager(num_threads=1)

        tfftp1, tfftq1, tfftpm1, tfftqm1 = ivp_solver(
            (one, zero), profiles, z, levels, Lx[msk], Ly[msk]
        )

        tfftp2, tfftq2, tfftpm2, tfftqm2 = ivp_solver(
            (zero, tfftq0[msk]), profiles, z, levels, Lx[msk], Ly[msk]
        )

        alpha = -(tfftq2 - K[nz - 1] * eigval * tfftp2) / (
            tfftq1 - K[nz - 1] * eigval * tfftp1
        )

        # linear combination of the two solution of the IVP
        tfftp[0, msk] = alpha
        tfftp[:, msk] = alpha * tfftpm1 + tfftpm2
        tfftq[:, msk] = alpha * tfftqm1 + tfftqm2

        # solve degenerated problem for (n,m) =  (0,0)
        # with Euler forward method
        lvl = 0
        tfftp00 = p000
        for i in range(nz - 1):

            if i in levels:
                tfftp[lvl, 0, 0] = tfftp00
                lvl += 1

            tfftp00 = tfftp00 - tfftq0[0, 0] / K[i] * dz[i]

        if nz - 1 in levels:
            tfftp[lvl, 0, 0] = tfftp00

    # shift green function in Fourier space to measurement point
    if footprint:
        shift = np.exp(1j * (Lx * (xm + halo) + Ly * (ym + halo)))
        tfftp = tfftp * shift
        tfftq = tfftq * shift
    # shift such that xm, ym are in the middle of the domain
    elif xm**2 + ym**2 > 0.0:
        shift = np.exp(1j * (Lx * (xm - xmx / 2) + Ly * (ym - ymx / 2)))
        tfftp = tfftp * shift
        tfftq = tfftq * shift

    # shift zero to center?
    tfftp = fftshift(tfftp, axes=(1, 2))
    tfftq = fftshift(tfftq, axes=(1, 2))

    # untruncate
    fftp = np.pad(
        tfftp, ((0, 0), (dly, dly), (dlx, dlx)), mode="constant", constant_values=0.0
    )
    fftq = np.pad(
        tfftq, ((0, 0), (dly, dly), (dlx, dlx)), mode="constant", constant_values=0.0
    )

    # unshift
    fftp = ifftshift(fftp, axes=(1, 2))
    fftq = ifftshift(fftq, axes=(1, 2))

    if footprint:
        # use fft to reverse sign, make green's function to footprint
        p = fft2(fftp, norm="backward").real  # concentration
        q = fft2(fftq, norm="backward").real  # kinematic flux
    else:
        # use ifft as usual
        p = ifft2(fftp, norm="forward").real  # concentration
        q = ifft2(fftq, norm="forward").real  # kinematic flux

    conc = p[:, py : nye - py, px : nxe - px]
    flx = q[:, py : nye - py, px : nxe - px]

    x = np.linspace(0, xmx, nx, endpoint=False)
    y = np.linspace(0, ymx, ny, endpoint=False)

    Z, Y, X = np.meshgrid(z[levels], y, x, indexing="ij")
    grid = (np.squeeze(Z), np.squeeze(Y), np.squeeze(X))

    return grid, np.squeeze(conc), np.squeeze(flx)


@parallelize
def ivp_solver(fftpq, profiles, z, levels, Lx, Ly):
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
    nxy = fftp0.shape[0]

    nlvls = len(levels)
    nz = len(z)
    dz = np.diff(z)

    # Initialize arrays
    fftpi, fftqi = np.copy(fftp0), np.copy(fftq0)
    fftp = np.zeros((nlvls, nxy), dtype=np.complex128)
    fftq = np.zeros((nlvls, nxy), dtype=np.complex128)

    lvl = 0

    for i in range(nz - 1):

        if i in levels:
            fftp[lvl, ...] = fftpi
            fftq[lvl, ...] = fftqi
            lvl += 1

        Ti = -K[i] * (Lx**2 + Ly**2) - 1j * u[i] * Lx - 1j * v[i] * Ly
        Kinv = 1.0 / K[i]
        dzi = dz[i]

        a = 1.0 - 0.5 * Kinv * Ti * dzi**2
        b = -Kinv * dzi - 1.0 / 6.0 * Kinv**2 * Ti * dzi**3
        c = Ti * dzi - 1.0 / 6.0 * Kinv * Ti**2 * dzi**3
        d = 1.0 - 0.5 * Kinv * Ti * dzi**2

        dum = a * fftpi + b * fftqi
        fftqi = c * fftpi + d * fftqi
        fftpi = dum

    if nz - 1 in levels:
        fftp[lvl, ...] = fftpi
        fftq[lvl, ...] = fftqi

    return fftpi, fftqi, fftp, fftq
