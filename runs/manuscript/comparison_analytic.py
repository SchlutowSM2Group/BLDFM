"""
Compare numerical and analytic solutions of concentration and flux.

Uses the config-driven interface with analytic=True and the plotting library
for the 2x2 panel comparison figure.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")

from dataclasses import replace

from bldfm import initialize, run_bldfm_single
from bldfm.config_parser import (
    BLDFMConfig, DomainConfig, TowerConfig, MetConfig, SolverConfig,
)
from bldfm.plotting import plot_field_comparison
from bldfm.utils import get_logger

logger = get_logger("comparison_analytic")

# --- Parameters (manuscript values) ---
# Original wind = (4.0, 1.0); convert to speed/dir for config
wind_u, wind_v = 4.0, 1.0
wind_speed = np.sqrt(wind_u**2 + wind_v**2)
wind_dir = np.degrees(np.arctan2(wind_u, wind_v))

nxy = (512, 256)
domain_ext = (200.0, 100.0)
src_pt = (10.0, 10.0)
meas_height = 10.0
ustar = 0.4

# --- Build config ---
initialize()

tower = TowerConfig(name="tower", lat=0.0, lon=0.0, z_m=meas_height)

config = BLDFMConfig(
    domain=DomainConfig(
        nx=nxy[0], ny=nxy[1], xmax=domain_ext[0], ymax=domain_ext[1],
        nz=256, modes=(512, 512), halo=1000.0,
    ),
    towers=[tower],
    met=MetConfig(ustar=ustar, mol=1e9, wind_speed=wind_speed, wind_dir=wind_dir),
    solver=SolverConfig(
        closure="CONSTANT", analytic=True,
        src_loc=src_pt, surface_flux_shape="point",
    ),
)

# --- Analytic reference ---
result_ana = run_bldfm_single(config, config.towers[0])

# --- Numerical solution ---
config_num = replace(config, solver=replace(config.solver, analytic=False))
tic = time.time()
result_num = run_bldfm_single(config_num, config_num.towers[0])
toc = time.time()

logger.info("Elapsed time for numerical solver %d s" % (toc - tic))

# --- Plotting ---
if __name__ == "__main__":
    fields = {
        "conc": result_num["conc"],
        "flx": result_num["flx"],
        "conc_ref": result_ana["conc"],
        "flx_ref": result_ana["flx"],
    }

    fig, axs = plot_field_comparison(fields, domain=domain_ext, src_pt=src_pt)
    fig.savefig("plots/comparison_analytic.png", dpi=300)
