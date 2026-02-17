"""
Compare BLDFM and Kormann-Meixner footprint models under unstable conditions.

Low-level API version: calls vertical_profiles() and steady_state_transport_solver()
directly. For the config-driven version, see ../interface/comparison_footprint_unstable.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver
from bldfm.ffm_kormann_meixner import estimateFootprint as FKM

logger = get_logger("comparison_footprint_unstable")


if __name__ == "__main__":
    nxy = 256, 768
    nz = 64
    modes = 512, 512
    domain = 50.0, 150.0
    halo = 400.0
    meas_pt = 25.0, 20.0
    meas_height = 10.0
    wind = 0.0, -5.0
    ustar = 1.064
    z0 = 0.5
    mol = -10.0
    sigma_v = 1.0

    xmx, ymx = domain
    nx, ny = nxy
    dx, dy = xmx / nx, ymx / ny
    wd = np.arctan(wind[0] / wind[1]) * 180.0 / np.pi
    umean = np.sqrt(wind[0] ** 2 + wind[1] ** 2)

    # --- BLDFM with MOST closure ---
    surf_flx = ideal_source(nxy, domain)
    z, profs = vertical_profiles(nz, meas_height, wind, z0=z0, mol=mol, closure="MOST")
    grid, conc, flx = steady_state_transport_solver(
        surf_flx, z, profs, domain, nz,
        modes=modes, meas_pt=meas_pt, footprint=True, halo=halo,
    )

    # --- BLDFM with MOSTM closure (no along-wind diffusion) ---
    surf_flx = ideal_source(nxy, domain)
    z, profs = vertical_profiles(nz, meas_height, wind, z0=z0, mol=mol, closure="MOSTM")
    grid, conc_m, flx_m = steady_state_transport_solver(
        surf_flx, z, profs, domain, nz,
        modes=modes, meas_pt=meas_pt, footprint=True, halo=halo,
    )

    # --- Kormann-Meixner footprint model ---
    grid_x, grid_y, grid_ffm = FKM(
        zm=meas_height, z0=z0, ws=umean, ustar=ustar, mo_len=mol,
        sigma_v=sigma_v, grid_domain=[0, xmx, 0, ymx], grid_res=dx,
        mxy=meas_pt, wd=wd,
    )

    # --- Plotting ---
    shrink = 0.8
    cmap = "turbo"
    lvls = 6
    linewidths = 4.0

    fig, axs = plt.subplots(1, 3, figsize=[8, 8], sharey=True, layout="constrained")

    vmin = 2e-6
    vmax = np.max(flx)
    levels = np.linspace(vmin, vmax, lvls, endpoint=False)
    X, Y, Z = grid

    axs[0].contour(X, Y, flx, levels, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=linewidths)
    axs[0].set_title("BLDFM")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")

    axs[1].contour(X, Y, flx_m, levels, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=linewidths)
    axs[1].set_title("BLDFM-SP")
    axs[1].set_xlabel("x [m]")

    plot = axs[2].contour(
        grid_x, grid_y, grid_ffm, levels,
        cmap=cmap, vmin=vmin, vmax=vmax, linewidths=linewidths,
    )
    axs[2].set_title("KM01")
    axs[2].set_xlabel("x [m]")

    cbar = fig.colorbar(plot, ax=axs, shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("$m^{-2}$")

    for ax in axs:
        ax.scatter(meas_pt[0], meas_pt[1], zorder=5, marker="*", color="red", s=300)

    plt.savefig("plots/comparison_footprint_unstable.png", dpi=300)
