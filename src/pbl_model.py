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
):
    """
    Computes profiles of horizontal wind and eddy diffusivity
    depending on the boundary layer's stratifictation
    from point measurements with, e.g., Eddy Covariance systems
    according to Monin-Obukhov Similarity Theory (MOST)
    (Kormann and Meixner, 2001, https://doi.org/10.1023/A:1018991015119).

    Parameters:
        n: scalar(int)
            Number of vertical grid points
        meas_height: scalar(float)
            Measurement height
        wind: array(float)
            List of zonal and meridional wind components u and v and
            the measurement height
        ustar: scalar(float)
            Friction velocity
        mol: scalar(float)
            Monin-Obukhov length
        prsc: scalar(float)
            Prandtl or Schmidt number depending on the scalar
            that is subject to atmospheric dispersion
        constant: scalar(bool)
            Switch for returning constant profiles

    Returns:
        z: array(float)
            1D array of vertical grid points
        profiles: array(float)
            List of 1D arrays of horizontal wind components u and v
            as well as the eddy diffusivity K
    """

    zm, (um, vm) = meas_height, wind

    # make function dimension-agnostic
    # here, we create (Nts x Nz arrays)
    um = np.array(um)
    vm = np.array(vm)
    ustar = np.array(ustar)

    um = um[..., np.newaxis]
    vm = vm[..., np.newaxis]
    ustar = ustar[..., np.newaxis]

    kap = 0.4  # Karman constant

    if closure == "CONSTANT":

        Km = kap * ustar * zm / prsc

        z0 = np.zeros_like(um)
        z = np.array([np.linspace(z00, zm, n) for z00 in z0]).squeeze()

        u = um * np.ones(n)
        v = vm * np.ones(n)
        K = Km * np.ones(n)

    if closure == "MOST":

        # absolute wind at zm
        absum = np.sqrt(um**2 + vm**2)

        if z0 < 0.0:

            # roughness length
            z0 = zm * np.exp(-kap * absum / ustar + psi(zm / mol))

            # sanity checks
            z0[...] = min(min(z0), z0_max)
            z0[...] = max(max(z0), z0_min)

        else:

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
    if closure == "OAAHOC":

        cl = 0.845
        cm = 0.0856
        ch = 0.204

        # compute tke from mol by tke balance equation
        if tke < 0.0:
            print("Warning: No tke available. Setting TKE to 1.0.")
            tke = 1.0

        # absolute wind at zm
        absum = np.sqrt(um**2 + vm**2)

        # roughness length
        z0 = zm * np.exp(-cm * cl * absum * np.sqrt(tke) / ustar**2)

        # equidistant vertical grid
        z = np.linspace(z0, zm, n)

        absu = ustar**2 / cm / cl / np.sqrt(tke) * np.log(z / z0)

        u = um / absum * absu
        v = vm / absum * absu

        K = ch * cl * z * np.sqrt(tke)

    print("Stats from vertical_profiles")
    print("z0    = %.3f m" % z[0])
    print("ustar = %.3f m s-1" % ustar)
    print(
        "umax  = %.3f m s-1, vmax = %.3f m s-1, Kmax = %.3f m2 s-1"
        % (max(u), max(v), max(K))
    )
    print(
        "umin  = %.3f m s-1, vmin = %.3f m s-1, Kmin = %.3f m2 s-1"
        % (min(u), min(v), min(K))
    )
    print()

    return z.squeeze(), (u.squeeze(), v.squeeze(), K.squeeze())


def psi(x):
    """
    Businger–Dyer relationship
    x =  z / mol
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
    Businger–Dyer relationship
    x =  z / mol
    """
    return np.where(
        x > 0.0, 1.0 + 5.0 * x, np.power(1.0 - 16.0 * x, -0.5, dtype=complex).real
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2)
    for mol in [-1000, -500, -400, -100, -30, 30, 100, 400, 500, 1000]:
        z, (u, v, K) = vertical_profiles(
            n=100, meas_height=5.0, wind=(3.0, 1.0), ustar=0.4, closure="MOST", mol=mol
        )
        axs[0].plot(u, z)
        axs[0].set_xlabel("u [m/s]")
        axs[0].set_ylabel("z [m]")
        axs[0].set_title("Profiles for MOST")
        axs[1].plot(K, z)
        axs[1].set_xlabel("K [m2 s-1]")
        axs[1].set_ylabel("z [m]")
    plt.show()

    fig, axs = plt.subplots(2)
    for tke in [0.2, 0.5, 1.0, 1.5, 2.0, 2.5]:
        z, (u, v, K) = vertical_profiles(
            n=100,
            meas_height=5.0,
            wind=(3.0, 1.0),
            ustar=0.4,
            closure="OAAHOC",
            tke=tke,
        )
        axs[0].plot(u, z)
        axs[0].set_xlabel("u [m/s]")
        axs[0].set_ylabel("z [m]")
        axs[0].set_title("Profiles for One-and-a-half-order closure")
        axs[1].plot(K, z)
        axs[1].set_xlabel("K [m2 s-1]")
        axs[1].set_ylabel("z [m]")
    plt.show()
