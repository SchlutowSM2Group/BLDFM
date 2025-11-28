"""
Module for comparing BLDFM and Kormann-Meixner footprint models.
"""

import numpy as np
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver
from bldfm.ffm_kormann_meixner import estimateFootprint as FKM
from bldfm import config

logger = get_logger("comparison_footprint_unstable")

nxy = 64, 1280
nz = 64
modes = 512, 512
# modes = 1024, 1024
# modes       = 128, 128
domain = 50.0, 1000.0
halo = 1000.0
meas_pt = 25.0, 20.0
meas_height = 10.0
wind = 0.0, -5.0
ustar = 0.250
z0 = 0.5
mol = 10.0
sigma_v = 0.04

xmx, ymx = domain
nx, ny = nxy
dx, dy = xmx / nx, ymx / ny
wd = np.arctan(wind[0] / wind[1]) * 180.0 / np.pi
umean = np.sqrt(wind[0] ** 2 + wind[1] ** 2)

############################################################
###  BLDFM
############################################################

surf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, z0=z0, mol=mol, closure="MOST")

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
###  BLDFM modified, no diffusion in flow direction
############################################################

surf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, z0=z0, mol=mol, closure="MOSTM")

srf_flx_m, bg_conc_m, conc_m, flx_m = steady_state_transport_solver(
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
    sigma_v=sigma_v,
    grid_domain=[0, xmx, 0, ymx],
    grid_res=dx,
    mxy=meas_pt,
    wd=wd,
)


############################################################
### plotting
############################################################
if __name__ == "__main__":
    shrink = 0.8
    cmap = "turbo"
    lvls = 6
    linewidths = 4.0

    fig, axs = plt.subplots(1, 3, figsize=[8, 8], sharey=True, layout="constrained")

    vmin = 1e-5
    vmax = np.max(flx)

    levels = np.linspace(vmin, vmax, lvls, endpoint=False)

    x = np.linspace(0, xmx, nx)
    y = np.linspace(0, ymx, ny)
    X, Y = np.meshgrid(x, y)

    c0 = axs[0].contour(
        X, Y, flx, levels, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=linewidths
    )

    axs[0].set_title("BLDFM")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")

    axs[1].contour(
        X, Y, flx_m, levels, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=linewidths
    )

    axs[1].set_title("BLDFM-SP")
    axs[1].set_xlabel("x [m]")

    plot = axs[2].contour(
        grid_x,
        grid_y,
        grid_ffm,
        levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=linewidths,
    )

    axs[2].set_title("KM01")
    axs[2].set_xlabel("x [m]")

    cbar = fig.colorbar(plot, ax=axs, shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("$m^{-2}$")

    axs[0].scatter(meas_pt[0], meas_pt[1], zorder=5, marker="*", color="red", s=300)
    axs[1].scatter(meas_pt[0], meas_pt[1], zorder=5, marker="*", color="red", s=300)
    axs[2].scatter(meas_pt[0], meas_pt[1], zorder=5, marker="*", color="red", s=300)

    plt.savefig("plots/comparison_footprint_stable.png", dpi=300)
