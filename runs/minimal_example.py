"""
Minimal example module showcasing basic usage of the BLDFM framework.
"""

import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver

logger = get_logger("minimal_example")

nxy = 512, 256
nz = 16
domain = 2000.0, 1000.0
meas_height = 10.0
wind = 4.0, 1.0
ustar = 0.4

srf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, ustar)

grid, conc, flx = steady_state_transport_solver(srf_flx, z, profs, domain, nz)

X, Y, Z = grid

if __name__ == "__main__":
    logger.info("Minimal example for neutrally stratified BL and default settings.")
    logger.info("")

    plt.figure()
    plt.pcolormesh(X, Y, conc)
    plt.title(f"Concentration at {meas_height} m")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.savefig("plots/concentration_at_meas_height.png")

    plt.figure()
    plt.pcolormesh(X, Y, flx)
    plt.title(f"Vertical kinematic flux at {meas_height} m")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.savefig("plots/kinematic_flux_at_meas_height.png")
