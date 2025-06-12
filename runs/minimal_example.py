"""
Minimal example module showcasing basic usage of the BLDFM framework.
"""

import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver

logger = get_logger('minimal_example')

nxy = 512, 256
nz = 16
domain = 2000.0, 1000.0
meas_height = 10.0
wind = 4.0, 1.0
ustar = 0.4

srf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, ustar)

srf_conc, bg_conc, conc, flx = steady_state_transport_solver(
    srf_flx, z, profs, domain, nz
)

if __name__ == "__main__":
    logger.info("Minimal example for neutrally stratified BL and default settings.")
    logger.info("")
    plt.figure()
    plt.imshow(srf_conc, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration at z0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/concentration_at_z0.png")

    plt.figure()
    plt.imshow(conc, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/concentration_at_zm.png")

    plt.figure()
    plt.imshow(flx, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Vertical kinematic flux at zm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/kinematic_flux_at_zm.png")
