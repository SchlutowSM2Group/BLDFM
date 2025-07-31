"""
Run script for comparing numerical and analytic solutions of concentration and flux.
"""

import numpy as np
import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import point_source, get_logger
from bldfm.solver import steady_state_transport_solver
from bldfm import config

config.NUM_THREADS = 16

logger = get_logger("numerical_convergence_test")

nxy = 512, 256
modes = 1024, 1024
nz = 512
domain = 200.0, 100.0
src_pt = 10.0, 10.0
halo = 1000.0
meas_height = 10.0
wind = 4.0, 2.0
ustar = 0.6

srf_flx = point_source(nxy, domain, src_pt)

z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")

_, _, conc_ana, flx_ana = steady_state_transport_solver(
    srf_flx, z, profs, domain, nz, modes=modes, halo=halo, analytic=True
)

modess = [
    (90, 90),
    (108, 108),
    (128, 128),
    (152, 152),
    (180, 180),
    (216, 216),
    (256, 256),
    (304, 304),
    (362, 362),
    (432, 432),
    #    (512, 512),
    #    (610, 610),
    #    (724, 724),
    #    (862, 862),
    #    (1024, 1024)
]
# nzs = [22, 27, 32, 38, 46, 54, 64, 76, 90, 108, 128, 152, 180, 215, 256]
# nzs = [22, 27, 32, 38, 46, 54, 64, 76, 90, 108, 128]
nzs = [22, 27, 32, 38, 46, 54, 64, 76, 90, 108]

# modess = [
#    (128, 128),
#    (180, 180),
#    (256, 256),
#    (362, 362),
#    (512, 512),
#    (724, 724),
#    (1024, 1024),
# ]
# nzs = [32, 45, 64, 90, 128, 180, 256]
#
conc_err = np.zeros(len(nzs))
flx_err = np.zeros(len(nzs))

for i, (modes, nz) in enumerate(zip(modess, nzs)):

    logger.info("modes: %s", modes)
    logger.info("nz: %d", nz)

    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")
    _, _, conc, flx = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo
    )

    conc_err[i] = np.mean((conc - conc_ana) ** 2) / np.mean(conc_ana**2)
    flx_err[i] = np.mean((flx - flx_ana) ** 2) / np.mean(flx_ana**2)

if __name__ == "__main__":

    # estimated grid lengths
    lx = domain[0] + 2.0 * halo
    ly = domain[1] + 2.0 * halo
    dx = 2.0 * lx / np.array(modess)[:, 0]
    dy = 2.0 * ly / np.array(modess)[:, 1]
    dz = 2.0 * meas_height / np.array(nzs)

    # effective grid size
    dxyz = np.cbrt(dx * dy * dz)

    plt.plot(dxyz, conc_err, "o")
    # plt.plot(dxyz, 1e-2 * dxyz**10, label="$\\mathcal{O}(h^{10})$")
    plt.title("Rate of numerical error convergence")
    plt.xlabel("$h$ [m]")
    plt.ylabel("Relative RMSE")
    # plt.legend()
    plt.loglog()
    plt.savefig("plots/error_convergence_analytic.png", dpi=300)
