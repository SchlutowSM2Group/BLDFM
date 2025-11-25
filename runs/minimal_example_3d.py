"""
Minimal example module showcasing basic usage of the BLDFM framework.
"""

import numpy as np
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver
from bldfm import config

config.NUM_THREADS = 16

logger = get_logger("minimal_example_3d")

nxy = 512, 256
nz = 64
domain = 2000.0, 1000.0
meas_height = 10.0
wind = 6.0, 0.0
ustar = 0.5

srf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, ustar)

levels = np.arange(nz + 1)

grid, conc, flx = steady_state_transport_solver(srf_flx, z, profs, domain, levels)

Z, Y, X = grid

if __name__ == "__main__":
    logger.info("Minimal example for neutrally stratified BL and default settings.")
    logger.info("")

    plt.figure()
    plt.pcolormesh(X[:, 64, :], Z[:, 64, :], conc[:, 64, :])
    plt.title("Vertical slice of concentration")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.colorbar()
    plt.savefig("plots/concentration_vertical_slice.png")

    plt.figure()
    plt.pcolormesh(X[:, 64, :], Z[:, 64, :], flx[:, 64, :])
    plt.title("Vertical slice of kinematic flux")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.colorbar()
    plt.savefig("plots/flux_vertical_slice.png")
