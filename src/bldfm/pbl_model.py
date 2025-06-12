import numpy as np
from .utils import get_logger

logger = get_logger(__name__.split("bldfm.")[-1])
logger.info("Loaded PBL model module.")


def vertical_profiles(
    n,
    meas_height,
    wind,
    ustar=None,
    z0=None,
    mol=1e9,
    prsc=1.0,
    closure="MOST",
    blend_height=None,
    stretch=None,
    z0_min=0.001,
    z0_max=2.0,
    tke=None,
):
    """
    Computes vertical profiles of horizontal wind components and eddy diffusivity in the planetary boundary layer (PBL) based on Monin-Obukhov Similarity Theory (MOST) or other closure models.

    Parameters:
        n (int): Number of vertical grid points between z0 and meas_height.
        meas_height (float): Measurement height above the ground.
        wind (tuple of floats): Zonal (u) and meridional (v) wind components at the measurement height.
        ustar (float or numpy.ndarray): Friction velocity [m/s].
        mol (float or numpy.ndarray, optional): Monin-Obukhov length. Default is 1e9 (neutral conditions).
        prsc (float, optional): Prandtl or Schmidt number. Default is 1.0.
        closure (str, optional): Closure model to use. Options are "MOST", "CONSTANT", or "OAAHOC". Default is "MOST".
        z0 (float or numpy.ndarray, optional): Roughness length. Default is -1e9 (auto-calculated).
        z0_min (float, optional): Minimum allowable roughness length. Default is 0.001.
        z0_max (float, optional): Maximum allowable roughness length. Default is 2.0.
        tke (float or numpy.ndarray, optional): Turbulent kinetic energy [m²/s²] for the "OAAHOC" closure. If not provided, it will default to 1.0.

    Returns:
        tuple:
            - z (numpy.ndarray): 1D array of vertical grid points.
            - profiles (tuple of numpy.ndarray): 1D arrays of horizontal wind components (u, v)
              and eddy diffusivity (K) at each vertical grid point.

    Raises:
        ValueError: If invalid closure type is provided.

    Notes:
        - The "OAAHOC" closure uses a one-and-a-half order closure model based on Schumann-Lilly.
        - The "MOST" closure uses Monin-Obukhov Similarity Theory.

    References:
        - Kormann, R., & Meixner, F. X. (2001). An analytical footprint model for non-neutral stratification. Boundary-Layer Meteorology, 99(2), 207–224. https://doi.org/10.1023/A:1018991015119
        - Schumann, U. (1991). Subgrid length-scales for large-eddy simulation of stratified turbulence. Theoretical and Computational Fluid Dynamics, 2(5), 279–290. https://doi.org/10.1007/BF00271468

    """

    zm, (um, vm) = meas_height, wind

    # make function dimension-agnostic
    # here, we create (Nts x Nz arrays)
    um = np.array(um)
    vm = np.array(vm)
    zm = np.array(zm)

    um = um[..., np.newaxis]
    vm = vm[..., np.newaxis]
    zm = zm[..., np.newaxis]

    kap = 0.4  # Karman constant

    # absolute wind at zm
    absum = np.sqrt(um**2 + vm**2)

    if closure == "CONSTANT" or closure == "MOST":

        if z0 is None:

            # roughness length
            z0 = zm * np.exp(-kap * absum / ustar + psi(zm / mol))

        elif ustar is None:

            ustar = absum * kap / (np.log(zm / z0) + psi(zm / mol))

        else:
            raise ValueError(f"Either z0 or ustar must be provided.")

    # One-and-a-half order closure
    # according to Schumann-Lilly closure (Schumann, 1991)
    elif closure == "OAAHOC":

        cl = 0.845
        cm = 0.0856
        ch = 0.204

        # compute tke from mol by tke balance equation
        if tke is None:
            logging.warning("No tke provided. Setting TKE to 1.0.")
            tke = 1.0
        tke = np.array(tke)[..., np.newaxis]

        # absolute wind at zm
        absum = np.sqrt(um**2 + vm**2)

        # roughness length
        z0 = zm * np.exp(-cm * cl * absum * np.sqrt(tke) / ustar**2)

    else:
        raise ValueError(
            f"Invalid closure type: {closure}. "
            "Supported closures are 'MOST', 'CONSTANT', and 'OAAHOC'."
        )

    # stretched vertical grid
    if stretch is None:
        h = 2 * meas_height
    else:
        h = stretch

    if blend_height is None:
        zmx = 2 * meas_height
    else:
        zmx = blend_height

    bb = zm / (np.exp(-z0 / h) - np.exp(-zm / h))
    aa = bb * np.exp(-z0 / h)

    zetamx = aa - bb * np.exp(-zmx / h)

    dzeta = zm / n

    nzeta = int(zetamx / dzeta)

    zeta = np.array([np.arange(0.0, zzz, dzeta) for zzz in zetamx]).squeeze()

    z = -h * np.log(-(zeta - aa) / bb)

    # Compute wind and eddy diffusivity profiles
    if closure == "CONSTANT":

        Km = kap * ustar * zm / prsc
        u = um * np.ones(len(z))
        v = vm * np.ones(len(z))
        K = Km * np.ones(len(z))

    elif closure == "MOST":

        # computation of MOST profiles
        absu = ustar / kap * (np.log(z / z0) + psi(z / mol))

        u = um / absum * absu
        v = vm / absum * absu

        K = kap * ustar * z / phi(z / mol) / prsc

    elif closure == "OAAHOC":

        absu = ustar**2 / cm / cl / np.sqrt(tke) * np.log(z / z0)

        u = um / absum * absu
        v = vm / absum * absu

        K = ch * cl * z * np.sqrt(tke)

    else:
        raise ValueError(
            f"Invalid closure type: {closure}. "
            "Supported closures are 'MOST', 'CONSTANT', and 'OAAHOC'."
        )

    if len(z) <= 1:
        logger.info("Stats from vertical_profiles")
        logger.info("z0    = %.3f m", z[0])
        logger.info("ustar = %.3f m s-1", ustar)
        logger.info(
            "umax  = %.3f m s-1, vmax = %.3f m s-1, Kmax = %.3f m2 s-1",
            max(u),
            max(v),
            max(K),
        )
        logger.info(
            "umin  = %.3f m s-1, vmin = %.3f m s-1, Kmin = %.3f m2 s-1",
            min(u),
            min(v),
            min(K),
        )

    return z.squeeze(), (u.squeeze(), v.squeeze(), K.squeeze())


def psi(x):
    """
    Stability correction function for Monin-Obukhov Similarity Theory (MOST).

    Parameters:
        x (float or numpy.ndarray): Dimensionless stability parameter (z / mol).

    Returns:
        numpy.ndarray: Stability correction factor for momentum.

    Notes:
        - For stable conditions (x > 0), a linear relationship is used.
        - For unstable conditions (x < 0), the Businger-Dyer formulation is applied.

    """
    xi = np.where(x > 0.0, np.nan, np.power(1.0 - 16.0 * x, 0.25, dtype=complex).real)
    return np.where(
        x > 0.0,
        5.0 * x,
        -2.0 * np.log(0.5 * (1.0 + xi))
        - np.log(0.5 * (1.0 + xi**2))
        + 2.0 * np.arctan(xi)
        - 0.5 * np.pi,
    )


def phi(x):
    """
    Stability correction function for eddy diffusivity in Monin-Obukhov Similarity Theory (MOST).

    Parameters:
        x (float or numpy.ndarray): Dimensionless stability parameter (z / mol).

    Returns:
        numpy.ndarray: Stability correction factor for eddy diffusivity.

    Notes:
        - For stable conditions (x > 0), a linear relationship is used.
        - For unstable conditions (x < 0), the Businger-Dyer formulation is applied.

    """
    return np.where(
        x > 0.0, 1.0 + 5.0 * x, np.power(1.0 - 16.0 * x, -0.5, dtype=complex).real
    )
