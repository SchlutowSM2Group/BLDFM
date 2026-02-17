"""
Error convergence test: compare numerical solutions against the analytic solution
at increasing resolution.

Low-level API version: calls vertical_profiles() and steady_state_transport_solver()
directly. For the config-driven version, see ../interface/analytic_convergence_test.py.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver
from bldfm import config

logger = get_logger("analytic_convergence_test")


if __name__ == "__main__":
    config.NUM_THREADS = 16

    nxy = 512, 256
    modes = 1024, 1024
    nz = 512
    domain = 200.0, 100.0
    src_loc = 10.0, 10.0
    halo = 1000.0
    meas_height = 10.0
    wind = 4.0, 2.0
    ustar = 0.6

    srf_flx = ideal_source(nxy, domain, src_loc=src_loc, shape="point")
    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")

    _, conc_ana, flx_ana = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo, analytic=True
    )

    modess = [
        (90, 90),
        (108, 108),
        (128, 128),
        (152, 152),
        (180, 180),
        (216, 216),
        (256, 256),
        (304, 304),
        (362, 362),
        (432, 432),
    ]
    nzs = [22, 27, 32, 38, 46, 54, 64, 76, 90, 108]

    conc_err = np.zeros(len(nzs))
    flx_err = np.zeros(len(nzs))

    for i, (modes_i, nz_i) in enumerate(zip(modess, nzs)):
        logger.info("modes: %s, nz: %d", modes_i, nz_i)
        z, profs = vertical_profiles(nz_i, meas_height, wind, ustar, closure="CONSTANT")
        _, conc, flx = steady_state_transport_solver(
            srf_flx, z, profs, domain, nz_i, modes=modes_i, halo=halo
        )
        conc_err[i] = np.mean((conc - conc_ana) ** 2) / np.mean(conc_ana**2)
        flx_err[i] = np.mean((flx - flx_ana) ** 2) / np.mean(flx_ana**2)

    # --- Curve fitting ---
    def expo(x, e0, r):
        return e0 * np.exp(-r / x)

    def cube(x, e0):
        return e0 * x**3

    # Effective grid spacing
    lx = domain[0] + 2.0 * halo
    ly = domain[1] + 2.0 * halo
    dx = 2.0 * lx / np.array(modess)[:, 0]
    dy = 2.0 * ly / np.array(modess)[:, 1]
    dz = 2.0 * meas_height / np.array(nzs)
    dxyz = np.cbrt(dx * dy * dz)

    popt_expo, _ = curve_fit(expo, dxyz, conc_err)
    popt_cube, _ = curve_fit(cube, dxyz, conc_err)

    # --- Plotting ---
    plt.plot(
        dxyz,
        expo(dxyz, *popt_expo),
        label=f"$\\exp(-{int(popt_expo[1])}" + "\\,\\mathrm{m}/h)$",
    )
    plt.plot(dxyz, cube(dxyz, *popt_cube), label="$h^3$")
    plt.plot(dxyz, conc_err, "o")
    plt.title("Error convergence for ANALY")
    plt.xlabel("$h$ [m]")
    plt.ylabel("Relative RMSE")
    plt.legend()
    plt.loglog()
    plt.savefig("plots/error_convergence_analytic.png", dpi=300)
