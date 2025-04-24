# Kormann & Meixner footprint model (Kormann and Meixner, 2001). The
# implementation is based on a MATLAB script originally by Jakob Sievers
# (05/2013) but revised and annotated by Christian Wille.
#
# Zhan Li, zhanli@gfz-potsdam.de
# Created: Sat Aug 22 18:41:23 CEST 2020

import warnings

import numpy as np
from scipy import special as spsp

von_karman = 0.4


def estimateZ0(zm, ws, wd, ustar, mo_len, half_wd_win=22):
    """Estimate roughness lengths based on Kormann-Meixner footprint model.

    Estimate roughness length (z0) by relating power-law wind profile in
    Kormann & Meixner's footprint model (Kormann and Meixner, 2001) to log wind
    profile in Monin-Obukhov similarity theory.

    The number of input observations at multiple time steps should be large
    enough to cover all possible wind directions because this function outputs
    roughness lengths that are smoothed within a given window of wind
    directions at 1-degree step. The default size of wind direction window for
    smoothing is 45-degree.

    The implementation is based on a MATLAB script originally by Jakob Sievers
    (05/2013) but revised and annotated by Christian Wille.

    Parameters
    ----------
    zm : ndarray of shape (n_obs,)
        List of measurement height (meter) per observation at one time step.

    ws : ndarray of shape (n_obs,)
        List of wind speed (m*s^-1) per observation at one time step.

    wd : ndarray of shape (n_obs,)
        List of wind direction (degree) per observation at one time step.

    ustar : ndarray of shape (n_obs,)
        List of friction velocity (m*s^-1) per observation at one time step.

    mo_len : ndarray of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step.

    half_wd_win : float
        Half window size of wind directions to smooth estimates of roughness
        length per each 1-degree wind direction. Therefore, the minimum half
        window size allowed is 1 degree. No smoothing if a half window size
        less than 1 degree is given.

    Returns
    -------
    z0med : ndarray of shape (n_obs,)
        List of estimated roughness length, z0 (meter) per measurement
        interval.

    Refernces
    ---------
    Kormann, R., Meixner, F.X., 2001. An Analytical Footprint Model For
    Non-Neutral Stratification. Boundary-Layer Meteorology 99, 207–224.
    https://doi.org/10.1023/A:1018991015119
    """
    # von Karman constant
    k = von_karman

    n_obs = len(zm)
    # Check if inputs are of the same length
    if (
        n_obs != len(ws)
        or n_obs != len(wd)
        or n_obs != len(ustar)
        or n_obs != len(mo_len)
    ):
        raise RuntimeError("Input parameters must be of the same length!")

    psi_m = _psiM(zm, mo_len)
    # roughness length z0, a solution for z0 to the Eq. (31) in (Kormann and
    # Meixner, 2001)
    z0 = zm * np.exp(psi_m - (k * ws / ustar))

    # remove outliers
    z0[z0 > 1000] = np.nan

    # NOTE: here z0, raw values of roughness length from solving Eq. (31) in
    # (Kormann and Meixner, 2001) is smoothed using median in a 45-deg window
    # of wind directions at 1-degree step. This is where the size of input
    # observations matter.
    if half_wd_win < 1:
        return z0

    z0med = np.zeros_like(z0) + np.nan
    for kk in range(0, 360):
        wd_wrapped = wd.copy()
        if kk < 90:
            wd_wrapped[wd > 270] = wd[wd > 270] - 360
        elif kk > 270:
            wd_wrapped[wd < 90] = wd[wd < 90] + 360
        idx1 = np.logical_and(wd >= kk, wd < (kk + 1))
        idx2 = np.logical_and(
            wd_wrapped >= (kk - half_wd_win), wd_wrapped < (kk + 1 + half_wd_win)
        )
        z0med[idx1] = np.nanmedian(z0[idx2])
    return z0med


def estimateFootprint(
    zm, z0, ws, ustar, mo_len, sigma_v, grid_domain, grid_res, mxy, wd=None
):
    """Estimate footprint of one measurement at one time step using Kormann &
    Meixner model.

    The estimation of the footprint of one measurement at one time step here
    uses the analytical approach in Kormann & Meixner's footprint model
    (Kormann and Meixner, 2001) to estimate two parameters, m and n for the
    power-law wind profile used by this footprint model.

    The implementation is based on a MATLAB script originally by Jakob Sievers
    (05/2013) but revised and annotated by Christian Wille.

    Parameters
    ----------
    zm : float
        The measurement/receptor height (meter) above zero displacement plane.

    z0 : float
        The roughness length (meter).

    ws : float
        The wind speed (m*s^-1).

    ustar : float
        The friction velocity (m*s^-1).

    mo_len : float
        The Monin-Obukhov length (meter).

    sigma_v : float
        The standard deviation of cross-wind (m*s^-1).

    grid_domain : array-like of shape (4,)
        The rectangular domain of the grid on which the footprint will be
        estimated. The domain is defined by a bounding box [xmin, xmax, ymin,
        ymax] in meters, in a coordinate system with the measurement/receptor
        at the location given by the parameter `mxy` within the `grid_domain`.
        The `[xmin, ymax]` will be the upper left corner of the upper left cell
        in the output grid.  The X and Y directions of this coordinate system
        are set up in one the following two ways.

        If wind direction is given by the optional parameter `wd`, the grid will
        be set up to align with the X-Y axes where wind direction is measured
        (clockwise from positive Y, that is, similar to how azimuth angle is
        measured). For example, if wind direction is measured with regard to
        true/due north. Then the output grid will align with the X axis from
        west towards east and the Y axis from south towards north.

        If no wind direction is given, the output grid aligns with along-wind
        and cross-wind direction.  Specifically in this coordinate system, the
        X axis points to the wind direction angle (where wind comes from, i.e.,
        positive X indicates upwind distances), then the Y axis is defined as
        perpendicular to the wind direction (i.e., cross-wind direction) and
        with the Y axis direction chosen to produce right-handed coordinates.

    grid_res : float
        The resolution (meter) of the grid on which the footprint will be
        estimated.

    mxy : array-like of shape (2,)
        The [x, y] of the measurement/receptor in the coordinate system of
        `grid_domain`.

    wd : float
        The wind direction (degree), an angle where wind comes from. It is
        measured clockwise with regard to the Y axis of a cardinal coordinate
        system. The output grid of flux footprint will align with the axes of
        this coordinate system.

        If you want the output grid to align with a pre-defined raster grid,
        you have to provide an adjusted wind direction angle that is with
        regard to the positive Y or north of this pre-defined grid system. That
        is, you need to change common wind direction angles based on true north
        to an adjusted value based on grid north. True north and grid north are
        usually not the same in many map projections where raster grids are
        defined. The difference between true north and grid north is called
        "meridian convergence" of a map projection. It depends on geolocation
        and map projection.

    Returns
    -------
    grid_x : ndarray of shape (grid_ysize, grid_xsize)
        The X coordinates of cell centers of the grid on which flux footprint
        is estimated. The `grid_ysize` and `grid_xsize` are the number of cells
        along X and Y axes, that is, columns and rows.

    grid_y : ndarray of shape (grid_ysize, grid_xsize)
        The Y coordinates of cell centers of the grid on which flux footprint
        is estimated. The `grid_ysize` and `grid_xsize` are the number of cells
        along X and Y axes, that is, columns and rows.

    grid_ffm : ndarray of shape (grid_ysize, grid_xsize)
        The flux footprint, phi(x, y, z) in (Kormann and Meixner, 2001) that
        describes the flux portion (m^-2) seen by the measurement/receptor. The
        `grid_ysize` and `grid_xsize` are the number of cells along X and Y
        axes, that is, columns and rows.

    Refernces
    ---------
    Kormann, R., Meixner, F.X., 2001. An Analytical Footprint Model For
    Non-Neutral Stratification. Boundary-Layer Meteorology 99, 207–224.
    https://doi.org/10.1023/A:1018991015119
    """
    # von Karman constant
    k = von_karman

    # set up the output grid
    xmin, xmax, ymin, ymax = tuple(grid_domain)
    grid_x, grid_y = np.meshgrid(
        np.arange(xmin + 0.5 * grid_res, xmax, grid_res),
        np.arange(ymax - 0.5 * grid_res, ymin, -grid_res),
    )
    grid_ffm = np.zeros_like(grid_x)

    # phi_m, Eq. (33) in (Kormann and Meixner, 2001), stability function
    phi_m = _phiM(np.asarray([zm]), np.asarray([mo_len]))[0]
    # phi_c, Eq. (34) in (Kormann and Meixner, 2001), stability function
    phi_c = _phiC(np.asarray([zm]), np.asarray([mo_len]))[0]
    # psi_m, Eq. (35) in (Kormann and Meixner, 2001), diabatic integration of
    # the wind profile.
    psi_m = _psiM(np.asarray([zm]), np.asarray([mo_len]))[0]

    m = _mParam(
        np.asarray([zm]), np.asarray([ws]), np.asarray([ustar]), np.asarray([mo_len])
    )[0]
    n = _nParam(np.asarray([zm]), np.asarray([mo_len]))[0]

    # kappa, from solving Eq. (11) & (32) in (Kormann and Meixner, 2001),
    # constant of eddy diffusivity in K(z)=kappa*z^n under the power-law wind
    # profile.
    kappa = k * zm * ustar / (phi_c * zm**n)

    # U, from solving Eq. (11) & (31) in (Kormann and Meixner, 2001), constant
    # of wind velocity in u(z)=U*z^m under the power-law wind profile.
    U = ustar * (np.log(zm / z0) + psi_m) / (k * zm**m)
    if U < 0:
        # Physically impossible
        msg = (
            "U in Eq. (11) of Kormann & Meixner, 2001 is estimated "
            + "negative as {0:.3f}, physically impossible! "
            + "Return empty footprint."
        )
        msg = msg.format(U)
        msg = (
            msg
            + "\n"
            + "zm = {0:.3f}, z0 = {1:.3f}, ws = {2:.3f}, "
            + "ustar = {3:.3f}, L = {4:.3f}, sigma_v = {5:.3f}"
        )
        msg = msg.format(zm, z0, ws, ustar, mo_len, sigma_v)
        warnings.warn(msg)
        return grid_x, grid_y, grid_ffm

    # r, p.213 in (Kormann and Meixner, 2001), shape factor
    r = 2 + m - n
    # mu, p.213 in (Kormann and Meixner, 2001), constant
    mu = (1 + m) / r
    # Xi, Eq. 19 in (Kormann and Meixner, 2001), flux length scale
    Xi = U * zm**r / (r**2 * kappa)

    # aggregate variables for use in the final footprint equation (combination
    # of Eq. (9) and (21) in (Kormann and Meixner, 2001)).

    # gamma, Eq. (18) in (Kormann and Meixner, 2001).
    gmm = spsp.gamma(mu)
    mr = m / r

    # A, a partial term of 1/sigma after combining Eq. (18) and p.212 (lines
    # after Eq. (9)) in (Kormann and Meixner, 2001)
    A = U / (spsp.gamma(1 / r) * sigma_v) * (kappa * r**2 / U) ** mr

    # num, a partial term of footprint phi(x,y,z) after combining Eq. (9) and
    # (21) in (Kormann and Meixner, 2001). By combining these two equations, we
    # will cancel a pair of gamma(mu) and hence reduce some computational
    # errors in the final footprint calculation.
    num = (1 / np.sqrt(2 * np.pi)) * Xi**mu

    if wd is None:
        # No wind direction given, the footprint grid aligns with along-wind
        # and cross-wind directions. grid_x and grid_y coordinates do NOT need
        # to be rotated but ONLY need to be shifted such that the
        # measurement/receptor's coordinates are (0, 0). The post-shift x and y
        # can be used in the footprint function.
        x, y = grid_x - mxy[0], grid_y - mxy[1]
    else:
        # Given a wind direction angle, the footprint grid will align with the
        # coordinate axes where wind direction is measured. We need to
        # transform the grid_x and grid_y in this coordinate system to the x
        # and y in the coordinate system that aligns with along-wind and
        # cross-wind directions for their use in the footprint function.
        #
        # First shift (0, 0) to measurement/receptor location and then rotate
        # coordinates in a polar coordinate system for simplicity.
        x, y = grid_x - mxy[0], grid_y - mxy[1]
        rho = np.sqrt(x**2 + y**2)
        # numpy's arctan2 function defines angular coordinates theta
        # counter-clockwise from X positive. Pay extra attention here because
        # this is different from the grid coordinate system where wind
        # direction angle is defined, clockwise from Y positive.
        theta = np.arctan2(y, x)
        new_theta = theta + np.deg2rad(wd) - np.pi * 0.5
        x = rho * np.cos(new_theta)
        y = rho * np.sin(new_theta)

    # phi, footprint function, flux portion at (x, y) location seen by the
    # measurement/receptor, by combining the Eq. (9) and (21) in (Kormann and
    # Meixner, 2001).
    sflag = x > 0  # Only upwind contributes to flux measurements.
    grid_ffm[sflag] = (
        grid_res**2
        * num
        * A
        * x[sflag] ** (mr - 2 - mu)
        * np.exp(
            -Xi / x[sflag] - 0.5 * (gmm * y[sflag] * A * x[sflag] ** (mr - 1)) ** 2
        )
    )

    return grid_x, grid_y, grid_ffm


def _phiM(zm, mo_len):
    """Calculate phi_m using Eq. (33), the stability function in (Kormann and
    Meixner, 2001)

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of measurement height (meter) per observation at one time steps.

    mo_len : arry-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step.

    Returns
    -------
        phi_m : ndarray of shape (n_obs,)
            Values of phi_m in (Kormann and Meixner, 2001).
    """
    phi_m = np.zeros_like(zm)
    sflag = mo_len < 0
    phi_m[sflag] = (1 - 16 * zm[sflag] / mo_len[sflag]) ** (-0.25)
    sflag = mo_len >= 0
    phi_m[sflag] = 1 + 5 * zm[sflag] / mo_len[sflag]
    return phi_m


def _phiC(zm, mo_len):
    """Calculate phi_c using Eq. (34), the stability function in (Kormann and
    Meixner, 2001)

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of measurement height (meter) per observation at one time steps.

    mo_len : arry-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step.

    Returns
    -------
        phi_c : ndarray of shape (n_obs,)
            Values of phi_c in (Kormann and Meixner, 2001).
    """
    phi_c = np.zeros_like(zm)
    sflag = mo_len < 0
    phi_c[sflag] = (1 - 16 * zm[sflag] / mo_len[sflag]) ** (-0.5)
    sflag = mo_len >= 0
    phi_c[sflag] = 1 + 5 * zm[sflag] / mo_len[sflag]
    return phi_c


def _psiM(zm, mo_len):
    """Calculate psi_m using Eq. (35), the diabatic integration of the wind
    profile (using 1/phi_m as zeta), in (Kormann and Meixner, 2001)

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of measurement height (meter) per observation at one time steps.

    mo_len : arry-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step.

    Returns
    -------
        psi_m : ndarray of shape (n_obs,)
            Values of psi_m in (Kormann and Meixner, 2001).
    """
    psi_m = np.zeros_like(zm)
    sflag = mo_len < 0
    inv_phi_m = (1 - 16 * zm[sflag] / mo_len[sflag]) ** (0.25)
    psi_m[sflag] = (
        -2 * np.log(0.5 * (1 + inv_phi_m))
        - np.log(0.5 * (1 + inv_phi_m**2))
        + 2 * np.arctan(inv_phi_m)
        - np.pi * 0.5
    )
    sflag = mo_len >= 0
    psi_m[sflag] = 5 * zm[sflag] / mo_len[sflag]
    return psi_m


def _mParam(zm, ws, ustar, mo_len):
    """Estimate the parameter m, in Eq. (11) of (Kormann and Meixner, 2001),
    exponent of wind velocity, in u(z)=U*z^m under the assumption of a
    power-law wind profile.

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of measurement height (meter) per observation at one time steps.

    ws : array-like of shape (n_obs,)
        List of wind speed (m*s^-1) per observation at one time step.

    ustar : array-like of shape (n_obs,)
        List of friction velocity (m*s^-1) per observation at one time step.

    mo_len : array-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step.

    Returns
    -------
        m : ndarray of shape (n_obs,)
            Values of m in the Eq. (11) of (Kormann and Meixner, 2001).
    """
    k = von_karman

    # Estimate m using the analytical approach in (Kormann and Meixner, 2001),
    # following the Eq. (36).
    phi_m = _phiM(zm, mo_len)
    m = ustar * phi_m / (k * ws)
    return m


def _nParam(zm, mo_len):
    """Estimate the parameter n, in Eq. (11) of (Kormann and Meixner, 2001),
    constant of eddy diffusivity, in K(z)=kappa*z^n under the assumption of a
    power-law wind profile.

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of measurement height (meter) per observation at one time steps.

    mo_len : array-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step.

    Returns
    -------
        n : ndarray of shape (n_obs,)
            Values of n in the Eq. (11) of (Kormann and Meixner, 2001).
    """
    # Estimate m using the analytical approach in (Kormann and Meixner, 2001),
    # following the Eq. (36).
    n = np.zeros_like(zm)
    sflag = mo_len < 0
    n[sflag] = (1 - 24 * zm[sflag] / mo_len[sflag]) / (
        1 - 16 * zm[sflag] / mo_len[sflag]
    )
    sflag = mo_len >= 0
    n[sflag] = 1 / (1 + 5 * zm[sflag] / mo_len[sflag])
    return n
