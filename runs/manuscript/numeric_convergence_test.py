"""
Error convergence test: low-res vs high-res numerical solutions.

Uses the config-driven interface with dataclasses.replace() to sweep over
(modes, nz) pairs, and the plotting library for the log-log convergence plot.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from scipy.optimize import curve_fit
from dataclasses import replace

from bldfm import initialize, run_bldfm_single
from bldfm.config_parser import (
    BLDFMConfig, DomainConfig, TowerConfig, MetConfig, SolverConfig,
    ParallelConfig,
)
from bldfm.plotting import plot_convergence
from bldfm.utils import get_logger
from bldfm import config as runtime_config

runtime_config.NUM_THREADS = 16

logger = get_logger("numeric_convergence_test")

# --- Parameters (manuscript values) ---
# Original wind = (4.0, 2.0); convert to speed/dir for config
wind_u, wind_v = 4.0, 2.0
wind_speed = np.sqrt(wind_u**2 + wind_v**2)
wind_dir = np.degrees(np.arctan2(wind_u, wind_v))

nxy = (512, 256)
domain_ext = (200.0, 100.0)
src_pt = (10.0, 10.0)
halo = 1000.0
meas_height = 10.0
z0 = 0.1

# --- Build base config (high-res numerical reference) ---
initialize()

tower = TowerConfig(name="tower", lat=0.0, lon=0.0, z_m=meas_height)

base_config = BLDFMConfig(
    domain=DomainConfig(
        nx=nxy[0], ny=nxy[1], xmax=domain_ext[0], ymax=domain_ext[1],
        nz=128, modes=(512, 512), halo=halo,
    ),
    towers=[tower],
    met=MetConfig(z0=z0, mol=1e9, wind_speed=wind_speed, wind_dir=wind_dir),
    solver=SolverConfig(
        closure="MOST",
        src_loc=src_pt, surface_flux_shape="point",
    ),
    parallel=ParallelConfig(num_threads=16),
)

# --- High-res reference solution ---
result_ref = run_bldfm_single(base_config, base_config.towers[0])
conc_ref = result_ref["conc"]
flx_ref = result_ref["flx"]

# --- Convergence sweep ---
modess = [
    (90, 90), (108, 108), (128, 128), (152, 152), (180, 180),
    (216, 216), (256, 256), (304, 304), (362, 362), (432, 432),
]
nzs = [22, 27, 32, 38, 46, 54, 64, 76, 90, 108]

conc_err = np.zeros(len(nzs))
flx_err = np.zeros(len(nzs))

for i, (modes, nz) in enumerate(zip(modess, nzs)):
    logger.info("modes: %s, nz: %d", modes, nz)

    sweep_config = replace(base_config,
        domain=replace(base_config.domain, nz=nz, modes=modes),
    )
    result = run_bldfm_single(sweep_config, sweep_config.towers[0])

    conc_err[i] = np.mean((result["conc"] - conc_ref) ** 2) / np.mean(conc_ref**2)
    flx_err[i] = np.mean((result["flx"] - flx_ref) ** 2) / np.mean(flx_ref**2)


def expo(x, e0, r):
    return e0 * np.exp(-r / x)


def cube(x, e0):
    return e0 * x**3


# --- Plotting ---
if __name__ == "__main__":
    lx = domain_ext[0] + 2.0 * halo
    ly = domain_ext[1] + 2.0 * halo
    dx = 2.0 * lx / np.array(modess)[:, 0]
    dy = 2.0 * ly / np.array(modess)[:, 1]
    dz = 2.0 * meas_height / np.array(nzs)
    dxyz = np.cbrt(dx * dy * dz)

    popt_expo, _ = curve_fit(expo, dxyz, conc_err)
    popt_cube, _ = curve_fit(cube, dxyz, conc_err)

    ax = plot_convergence(
        dxyz, conc_err,
        fits=[
            (lambda h, e0=popt_expo[0], r=popt_expo[1]: expo(h, e0, r),
             {}, f"$\\exp(-{int(popt_expo[1])}" + "\\,\\mathrm{m}/h)$"),
            (lambda h, e0=popt_cube[0]: cube(h, e0),
             {}, "$h^3$"),
        ],
        title="Error convergence for HI-RES",
    )
    ax.get_figure().savefig("plots/error_convergence_numeric.png", dpi=300)
