"""
Compare numerical and analytic solutions of concentration and flux.

Low-level API version: calls vertical_profiles() and steady_state_transport_solver()
directly. For the config-driven version, see ../interface/comparison_analytic.py.
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger
from bldfm.solver import steady_state_transport_solver

logger = get_logger("comparison_analytic")


if __name__ == "__main__":
    nxy = 512, 256
    modes = 512, 512
    nz = 256
    domain = 200.0, 100.0
    src_pt = 10.0, 10.0
    halo = 1000.0
    meas_height = 10.0
    wind = 4.0, 1.0
    ustar = 0.4

    srf_flx = ideal_source(nxy, domain, src_pt, shape="point")

    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")

    _, conc_ana, flx_ana = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo, analytic=True
    )

    tic = time.time()
    _, conc, flx = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo
    )
    toc = time.time()

    logger.info("Elapsed time for numerical solver %d s" % (toc - tic))

    diff_conc = (conc - conc_ana) / np.max(conc_ana)
    diff_flx = (flx - flx_ana) / np.max(flx_ana)

    # --- Plotting ---
    shrink = 0.7
    cmap = "turbo"

    fig, axs = plt.subplots(
        2, 2, figsize=[10, 6], sharex=True, sharey=True, layout="constrained"
    )

    plot = axs[0, 0].imshow(
        conc, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )
    axs[0, 0].set_title("Numerical concentration")
    axs[0, 0].set_ylabel("y [m]")
    axs[0, 0].xaxis.set_tick_params(labelbottom=False)
    cbar = fig.colorbar(plot, ax=axs[0, 0], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("a.u.")

    plot = axs[0, 1].imshow(
        flx, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )
    axs[0, 1].set_title("Numerical flux")
    axs[0, 1].xaxis.set_tick_params(labelbottom=False)
    axs[0, 1].yaxis.set_tick_params(labelleft=False)
    cbar = fig.colorbar(plot, ax=axs[0, 1], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("a.u. m/s")

    plot = axs[1, 0].imshow(
        diff_conc, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )
    axs[1, 0].set_title("Relative difference to analytic concentration")
    axs[1, 0].set_xlabel("x [m]")
    axs[1, 0].set_ylabel("y [m]")
    cbar = fig.colorbar(plot, ax=axs[1, 0], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    plot = axs[1, 1].imshow(
        diff_flx, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )
    axs[1, 1].set_title("Relative difference to analytic flux")
    axs[1, 1].set_xlabel("x [m]")
    axs[1, 1].yaxis.set_tick_params(labelleft=False)
    cbar = fig.colorbar(plot, ax=axs[1, 1], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    axs[0, 0].scatter(src_pt[0], src_pt[1], zorder=5, marker="*", color="red", s=100)
    axs[0, 1].scatter(src_pt[0], src_pt[1], zorder=5, marker="*", color="red", s=100)

    plt.savefig("plots/manuscript_low_level_comparison_analytic.png", dpi=300)
