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
    (128, 128),
    (180, 180),
    (256, 256),
    (362, 362),
    (512, 512),
    (724, 724),
    (1024, 1024),
]
nzs = [32, 45, 64, 90, 128, 180, 256]

conc_err = np.zeros(len(nzs))
flx_err = np.zeros(len(nzs))

for i, (modes, nz) in enumerate(zip(modess, nzs)):

    print("modes ", modes)
    print("nz ", nz)
    print()

    z, profs = vertical_profiles(nz, meas_height, wind, ustar, closure="CONSTANT")
    _, _, conc, flx = steady_state_transport_solver(
        srf_flx, z, profs, domain, nz, modes=modes, halo=halo
    )

    conc_err[i] = np.mean((conc - conc_ana) ** 2) / np.mean(conc_ana**2)
    flx_err[i] = np.mean((flx - flx_ana) ** 2) / np.mean(flx_ana**2)

if __name__ == "__main__":

    # estimated grid lengths
    dx = 2.0 * domain[0] / np.array(modess)[:, 0]
    dy = 2.0 * domain[1] / np.array(modess)[:, 1]
    dz = 2.0 * meas_height / np.array(nzs)

    # effective grid size
    dxyz = np.cbrt(dx * dy * dz)

    plt.plot(dxyz, conc_err, "o")
    plt.plot(dxyz, 1e-2 * dxyz**10, label="$\\mathcal{O}(h^{10})$")
    plt.title("Rate of numerical error convergence")
    plt.xlabel("$h$ [m]")
    plt.ylabel("Relative RMSE")
    plt.legend()
    plt.loglog()
    plt.savefig("plots/error_convergence.png", dpi=300)

