"""
Module for comparing BLDFM and Kormann-Meixner footprint models.
"""

import numpy as np
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source
from bldfm.solver import steady_state_transport_solver
from bldfm.ffm_kormann_meixner import estimateFootprint as FKM

nxy = 512, 512
nz = 128
modes = 1024, 1024
# modes       = 128, 128
domain = 200.0, 600.0
halo = 2000.0
meas_pt = 100.0, 0.0
meas_height = 10.0
wind = 0.0, -6.0
ustar = 0.63
z0 = 0.1
mol = -20.0

wd = np.arctan(wind[0] / wind[1]) * 180.0 / np.pi
umean = np.sqrt(wind[0] ** 2 + wind[1] ** 2)

############################################################
###  BLDFM
############################################################

surf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, z0=z0, mol=mol)

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

############################################################
### Korman and Meixner's footprint model
############################################################

grid_x, grid_y, grid_ffm = FKM(
    zm=meas_height,
    z0=z0,
    ws=umean,
    ustar=ustar,
    mo_len=mol,
    sigma_v=0.5 * ustar,
    grid_domain=[0, domain[0], 0, domain[1]],
    grid_res=domain[0] / nxy[0],
    mxy=meas_pt,
    wd=wd,
)


############################################################
### plotting
############################################################
if __name__ == "__main__":
    shrink = 0.5
    cmap = "turbo"

    fig, axs = plt.subplots(1, 2, figsize=[6, 8], sharey=True, layout="constrained")

    # BLDFM
    plot = axs[0].imshow(
        flx, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )

    axs[0].set_title("BLDFM")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")

    # FKM
    plot = axs[1].imshow(
        grid_ffm, origin="upper", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )

    axs[1].set_title("FKM")
    axs[1].set_xlabel("x [m]")
    cbar = fig.colorbar(plot, ax=axs, shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("$m^{-2}$")

    # plt.show()
    plt.savefig("plots/comparison_footprint_unstable.png", dpi=300)
