"""
Run script for comparing numerical and analytic solutions of concentration and flux.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from src.pbl_model import vertical_profiles
from src.utils import point_source
from src.solver import steady_state_transport_solver

nxy = 512, 256
modes = 512, 512
nz = 512
# nxy         = 128, 64
# modes       = 128, 128
# nz          = 16
domain = 200.0, 100.0
src_pt = 10.0, 10.0
fetch = 1000.0
meas_height = 10.0
wind = 4.0, 1.0
ustar = 0.4

srf_flx = point_source(nxy, domain, src_pt)

z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")

srf_conc_ana, bg_conc_ana, conc_ana, flx_ana = steady_state_transport_solver(
    srf_flx, z, profs, domain, modes=modes, fetch=fetch, analytic=True
)

tic = time.time()
srf_conc, bg_conc, conc, flx = steady_state_transport_solver(
    srf_flx, z, profs, domain, modes=modes, fetch=fetch, ivp_method="TSEI3"
)
toc = time.time()

t = toc - tic

print("Elapsed time for numerical solver %d s" % t)

# diff_conc = (conc - conc_ana) / np.mean(conc_ana)
# diff_flx  = (flx - flx_ana) / np.mean(flx_ana)

diff_conc = (conc - conc_ana) / np.max(conc_ana)
diff_flx = (flx - flx_ana) / np.max(flx_ana)

if __name__ == "__main__":
    shrink = 0.7
    cmap = "turbo"

    fig, axs = plt.subplots(
        2, 2, figsize=[10, 6], sharex=True, sharey=True, layout="constrained"
    )

    plot = axs[0, 0].imshow(
        conc, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )

    axs[0, 0].plot(src_pt[0], src_pt[1], "ro")
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

    axs[0, 1].plot(src_pt[0], src_pt[1], "ro")
    axs[0, 1].set_title("Numerical flux")
    axs[0, 1].xaxis.set_tick_params(labelbottom=False)
    axs[0, 1].yaxis.set_tick_params(labelleft=False)
    cbar = fig.colorbar(plot, ax=axs[0, 1], shrink=shrink, location="bottom")
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label("a.u. m/s")

    cmap = "turbo"

    plot = axs[1, 0].imshow(
        diff_conc, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )

    # axs[1,0].plot(src_pt[0],src_pt[1],'ro')
    axs[1, 0].set_title("Relative difference to analytic concentration")
    axs[1, 0].set_xlabel("x [m]")
    axs[1, 0].set_ylabel("y [m]")
    cbar = fig.colorbar(plot, ax=axs[1, 0], shrink=shrink, location="bottom")
    # cbar.set_label('‰')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    plot = axs[1, 1].imshow(
        diff_flx, origin="lower", cmap=cmap, extent=[0, domain[0], 0, domain[1]]
    )

    # axs[1,1].plot(src_pt[0],src_pt[1],'ro')
    axs[1, 1].set_title("Relative difference to analytic flux")
    axs[1, 1].set_xlabel("x [m]")
    axs[1, 1].yaxis.set_tick_params(labelleft=False)
    cbar = fig.colorbar(plot, ax=axs[1, 1], shrink=shrink, location="bottom")
    # cbar.set_label('‰')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)

    # plt.show()
    plt.savefig("plots/comparison_analytic.png", dpi=300)
