"""
Example module demonstrating the calculation of concentration and flux footprints.
"""

import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source
from bldfm.solver import steady_state_transport_solver

nxy = 512, 256
nz = 256
modes = 512, 512
domain = 200.0, 700.0
halo = 1000.0
meas_pt = 100.0, 0.0
meas_height = 10.0
wind = 0.0, -6.0
z0 = 0.1
mol = 10.0

surf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, mol=mol, z0=z0)

srf_flx, bg_conc, conc, flx = steady_state_transport_solver(
    surf_flx,
    z,
    profs,
    domain,
    nz,
    modes=modes,
    meas_pt=meas_pt,
    footprint=True,
    halo=halo,
)

if __name__ == "__main__":
    plt.figure()
    plt.imshow(conc, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration footprint")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar()
    plt.savefig("plots/concentration_footprint.png")

    plt.figure()
    plt.imshow(flx, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Flux footprint")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.colorbar()
    plt.savefig("plots/flux_footprint.png")

# %%
