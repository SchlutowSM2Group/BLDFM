import matplotlib.pyplot as plt

from src.pbl_model import vertical_profiles
from src.utils import ideal_source
from src.solver import steady_state_transport_solver

nxy = 512, 256
nz = 100
modes = 512, 512
domain = 1000.0, 700.0
fetch = 2000.0
meas_pt = 500.0, 0.0
meas_height = 6.0
wind = 0.0, -6.0
ustar = 0.5
z0 = 0.1
mol = -100.0

surf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, ustar, mol=mol, z0=z0)

srf_flx, bg_conc, conc, flx = steady_state_transport_solver(
    surf_flx,
    z,
    profs,
    domain,
    modes=modes,
    meas_pt=meas_pt,
    footprint=True,
    fetch=fetch,
)

if __name__ == "__main__":
    plt.figure()
    plt.imshow(conc, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Concentration footprint")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/concentration_footprint.png")

    plt.figure()
    plt.imshow(flx, origin="lower", extent=[0, domain[0], 0, domain[1]])
    plt.title("Flux footprint")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig("plots/flux_footprint.png")

# %%
