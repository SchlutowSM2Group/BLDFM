"""
Example module demonstrating the calculation of flux footprints.
"""

import matplotlib.pyplot as plt
import numpy as np

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver

logger = get_logger("Minimal footprint_example")

nx = 512
ny = 256
nz = 32
domain = 100.0, 700.0
meas_pt = 50.0, 0.0
meas_height = 10.0
wind = 0.0, -6.0
ustar = 0.5

area = np.zeros([ny, nx])

z, profs = vertical_profiles(nz, meas_height, wind, ustar)

grid, conc, flx = steady_state_transport_solver(
    area, z, profs, domain, nz, meas_pt=meas_pt, footprint=True
)

if __name__ == "__main__":

    X, Y, Z = grid

    plt.figure()
    plt.pcolormesh(X, Y, flx)
    plt.title("Flux footprint")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar()
    cbar.set_label("$m^{-2}$")
    plt.savefig("plots/flux_footprint.png")

# %%
