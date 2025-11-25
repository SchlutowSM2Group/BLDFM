"""
Minimal example module showcasing basic usage of the BLDFM framework.
"""

import numpy as np
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver
from bldfm.solver3d import steady_state_transport_solver_3d
from bldfm import config

config.NUM_THREADS = 16

logger = get_logger("minimal_example")

nxy = 512, 256
nz = 64
domain = 2000.0, 1000.0
meas_height = 10.0
wind = 4.0, 1.0
ustar = 0.4

srf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, z0=0.1)

levels = np.arange(nz+1)

_, _, conc_2d, flx_2d = steady_state_transport_solver(
    srf_flx, z, profs, domain, nz
)

grid, conc_3d, flx_3d = steady_state_transport_solver_3d(
    srf_flx, z, profs, domain, nz
)

Z, Y, X = grid

#plt.pcolormesh(X[:,64,:],Z[:,64,:],conc_3d[:,64,:])
plt.pcolormesh(X,Y,conc_3d)
plt.show()
exit()


if __name__ == "__main__":
    logger.info("Minimal example for neutrally stratified BL and default settings.")
    logger.info("")
 
    plt.figure()
    plt.imshow(conc_3d, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration at z0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/3d_concentration_at_z0.png")


    plt.figure()
    plt.imshow(conc_3d, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/3d_concentration_at_zm.png")

    plt.figure()
    plt.imshow(conc_2d, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/2d_concentration_at_zm.png")

    plt.figure()
    plt.imshow(srf_conc, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/2d_srf_conc.png")


    plt.figure()
    plt.imshow(conc_3d-conc_2d, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration diff at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/comparison_2d3d_concentration_at_zm.png")

    plt.figure()
    plt.imshow(flx_3d, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Vertical kinematic flux at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/kinematic_flux_at_zm_3d.png")

