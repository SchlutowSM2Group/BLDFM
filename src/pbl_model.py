"""
pbl_model.py

This module provides functions to compute vertical profiles of horizontal wind and eddy diffusivity in the planetary boundary layer (PBL). The calculations are based on Monin-Obukhov Similarity Theory (MOST) and other closure models.

Functions:
----------
    - vertical_profiles: Computes vertical profiles of wind and eddy diffusivity.
    - psi: Stability correction function for MOST.
    - phi: Stability correction function for eddy diffusivity.

References:
-----------
    - Kormann, R., & Meixner, F. X. (2001). An analytical footprint model for non-neutral stratification. Boundary-Layer Meteorology, 99(2), 207–224. https://doi.org/10.1023/A:1018991015119
    - Schumann, U. (1991). Subgrid length-scales for large-eddy simulation of stratified turbulence. Theoretical and Computational Fluid Dynamics, 2(5), 279–290.

"""

import logging
import numpy as np


def vertical_profiles(
    n,
    meas_height,
    wind,
    ustar,
    mol=1e9,
    prsc=1.0,
    closure="MOST",
    z0=-1e9,
    z0_min=0.001,
    z0_max=2.0,
    tke=None,
):
    """
    Computes vertical profiles of horizontal wind components and eddy diffusivity in the planetary boundary layer (PBL) based on Monin-Obukhov Similarity Theory (MOST) or other closure models.

    Parameters:
        n (int): Number of vertical grid points.
        meas_height (float): Measurement height above the ground.
        wind (tuple of floats): Zonal (u) and meridional (v) wind components at the measurement height.
        ustar (float or numpy.ndarray): Friction velocity (in m/s).
        mol (float or numpy.ndarray, optional): Monin-Obukhov length. Default is 1e9 (neutral conditions).
        prsc (float, optional): Prandtl or Schmidt number. Default is 1.0.
        closure (str, optional): Closure model to use. Options are "MOST", "CONSTANT", or "OAAHOC". Default is "MOST".
        z0 (float or numpy.ndarray, optional): Roughness length. Default is -1e9 (auto-calculated).
        z0_min (float, optional): Minimum allowable roughness length. Default is 0.001.
        z0_max (float, optional): Maximum allowable roughness length. Default is 2.0.
        tke (float or numpy.ndarray, optional): Turbulent kinetic energy (in m²/s²) for the "OAAHOC" closure. If not provided, it will default to 1.0.

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
        - Schumann, U. (1991). Subgrid length-scales for large-eddy simulation of stratified turbulence. Theoretical and Computational Fluid Dynamics, 2, 279–290.

    """

    zm, (um, vm) = meas_height, wind

    # make function dimension-agnostic
    # here, we create (Nts x Nz arrays)
    um = np.array(um)
    vm = np.array(vm)
    zm = np.array(zm)
    ustar = np.array(ustar)

    um = um[..., np.newaxis]
    vm = vm[..., np.newaxis]
    zm = zm[..., np.newaxis]
    ustar = ustar[..., np.newaxis]

    kap = 0.4  # Karman constant

    if closure == "CONSTANT":

        Km = kap * ustar * zm / prsc

        z0 = np.zeros_like(um)
        z = np.array([np.linspace(z00, zm, n) for z00 in z0]).squeeze()

        u = um * np.ones(n)
        v = vm * np.ones(n)
        K = Km * np.ones(n)

    elif closure == "MOST":

        # absolute wind at zm
        absum = np.sqrt(um**2 + vm**2)

        if z0 < 0.0:

            # roughness length
            z0 = zm * np.exp(-kap * absum / ustar + psi(zm / mol))

            # sanity checks
            z0[...] = np.clip(z0, z0_min, z0_max)

        else:

            z0 = np.array(z0)[..., np.newaxis]
            ustar = absum * kap / (np.log(zm / z0) + psi(zm / mol))

        # equidistant vertical grid
        # find a way to vectorize this properly
        z = np.array([np.linspace(z00, zm, n) for z00 in z0]).squeeze()

        absu = ustar / kap * (np.log(z / z0) + psi(z / mol))

        u = um / absum * absu
        v = vm / absum * absu

        K = kap * ustar * z / phi(z / mol) / prsc

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

        # equidistant vertical grid
        z = np.array([np.linspace(z00, zm, n) for z00 in z0]).squeeze()

        absu = ustar**2 / cm / cl / np.sqrt(tke) * np.log(z / z0)

        u = um / absum * absu
        v = vm / absum * absu

        K = ch * cl * z * np.sqrt(tke)

    else:
        raise ValueError(
            f"Invalid closure type: {closure}. "
            "Supported closures are 'MOST', 'CONSTANT', and 'OAAHOC'."
        )

    logging.info("Stats from vertical_profiles")
    logging.info("z0    = %.3f m", z[0])
    logging.info("ustar = %.3f m s-1", ustar)
    logging.info(
        "umax  = %.3f m s-1, vmax = %.3f m s-1, Kmax = %.3f m2 s-1",
        max(u),
        max(v),
        max(K),
    )
    logging.info(
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
