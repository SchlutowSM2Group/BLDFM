"""
Minimal example module showcasing basic usage of the BLDFM framework.
"""

import numpy as np
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver
from bldfm import config

config.NUM_THREADS = 8

logger = get_logger("3D plume")

nxy = 512, 256
modes = 512, 512
nz = 64
domain = 800.0, 100.0
src_loc = 200.0, 50.0
meas_height = 10.0
wind = 6.0, 0.0
ustar = 0.4

# define a point source at src_loc
srf_flx = ideal_source(nxy, domain, src_loc=src_loc, shape='point')

# compute profiles for neutral PBL
z, profs = vertical_profiles(nz, meas_height, wind, ustar)

# take every second level for output
levels = np.arange(0, nz + 1, 2)

# BLDFM in dispersion mode
grid, conc, flx = steady_state_transport_solver(srf_flx, z, profs, domain, levels, modes=modes)

X, Y, Z = grid

if __name__ == "__main__":
    logger.info("Plume")
    logger.info("")

    plt.figure()
    plt.pcolormesh(X[0, :, :], Y[0, :, :], conc[0, :, :])
    plt.title("Horizontal slice of concentration")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    #plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.savefig("plots/ptsrc_concentration_xy_slice_at_z0.png")

    plt.figure()
    plt.pcolormesh(X[0, :, :], Y[0, :, :], flx[0, :, :])
    plt.title("Horizontal slice of flux field")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    #plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.savefig("plots/ptsrc_flux_xy_slice_at_z0.png")

    plt.figure()
    plt.pcolormesh(X[-1, :, :], Y[-1, :, :], conc[-1, :, :])
    plt.title("Horizontal slice of concentration")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]") 
    #plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.savefig("plots/ptsrc_concentration_xy_slice_at_zmx.png")

    plt.figure()
    plt.pcolormesh(X[:, nxy[1]//2, :], Z[:, nxy[1]//2, :], conc[:, nxy[1]//2, :], vmax=0.01)
    plt.title("Vertical slice of concentration")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.colorbar()
    plt.savefig("plots/ptsrc_concentration_xz_slice.png")

    plt.figure()
    plt.pcolormesh(X[:, nxy[1]//2, :], Z[:, nxy[1]//2, :], flx[:, nxy[1]//2, :])
    plt.title("Vertical slice of flux field")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.colorbar()
    plt.savefig("plots/ptsrc_concentration_xz_slice.png")


    plt.figure()
    plt.pcolormesh(Y[:, :, nxy[0]*3//4], Z[:, :, nxy[0]*3//4], conc[:, :, nxy[0]*3//4])
    plt.title("Vertical slice of concentration")
    plt.xlabel("y [m]")
    plt.ylabel("z [m]")
    plt.colorbar()
    plt.savefig("plots/ptsrc_concentration_yz_slice.png")

