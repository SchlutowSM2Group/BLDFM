"""
Example module showcasing BLDFM in parallelized mode.
"""

import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver
from bldfm import config

# Run BLDFM on 4 threads/CPUs
config.NUM_THREADS = 4

logger = get_logger("parallel_example")

# high resolution
nxy = 1024, 1024
nz = 512
domain = 2000.0, 1000.0
meas_height = 10.0
wind = 4.0, 1.0
z0 = 0.1

srf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, z0=z0)

grid, conc, flx = steady_state_transport_solver(srf_flx, z, profs, domain, nz)

if __name__ == "__main__":

    logger.info("Example for BLDFM in parallelized mode.")
    logger.info("")

    X, Y, Z = grid

    plt.figure()
    plt.pcolormesh(X, Y, conc)
    plt.title("Concentration at meas_height")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar()
    plt.savefig("plots/examples_low_level_concentration_at_meas_height.png")
