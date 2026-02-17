"""
Compare BLDFM and Kormann-Meixner footprint models under stable conditions.

Uses the config-driven interface for BLDFM runs and the plotting library for
figure generation.  The KM01 external model is called directly.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from dataclasses import replace

from bldfm import initialize, run_bldfm_single, compute_wind_fields
from bldfm.config_parser import (
    BLDFMConfig, DomainConfig, TowerConfig, MetConfig, SolverConfig,
)
from bldfm.ffm_kormann_meixner import estimateFootprint as FKM
from bldfm.plotting import plot_footprint_comparison
from bldfm.utils import get_logger

logger = get_logger("comparison_footprint_stable")


if __name__ == "__main__":
    # --- Parameters (manuscript values) ---
    nxy = (64, 1280)
    domain_ext = (50.0, 1000.0)
    meas_pt = (25.0, 20.0)
    meas_height = 10.0
    ustar = 0.250
    z0 = 0.5
    mol = 10.0
    sigma_v = 0.04
    wind_speed = 5.0
    wind_dir = 180.0  # from south: (u, v) = (0, -5)

    # --- Build config ---
    initialize()

    tower = TowerConfig(name="tower", lat=0.0, lon=0.0, z_m=meas_height,
                        x=meas_pt[0], y=meas_pt[1])

    config = BLDFMConfig(
        domain=DomainConfig(
            nx=nxy[0], ny=nxy[1], xmax=domain_ext[0], ymax=domain_ext[1],
            nz=64, modes=(512, 512), halo=1000.0,
        ),
        towers=[tower],
        met=MetConfig(ustar=ustar, z0=z0, mol=mol,
                      wind_speed=wind_speed, wind_dir=wind_dir),
        solver=SolverConfig(closure="MOST", footprint=True),
    )

    # --- BLDFM with MOST closure ---
    result_most = run_bldfm_single(config, config.towers[0])

    # --- BLDFM with MOSTM closure (no along-wind diffusion) ---
    config_mostm = replace(config, solver=replace(config.solver, closure="MOSTM"))
    result_mostm = run_bldfm_single(config_mostm, config_mostm.towers[0])

    # --- KM01 (external model, direct call) ---
    xmx, ymx = domain_ext
    dx = xmx / nxy[0]
    u, v = compute_wind_fields(wind_speed, wind_dir)
    wd = np.arctan(u / v) * 180.0 / np.pi

    grid_x, grid_y, grid_ffm = FKM(
        zm=meas_height, z0=z0, ws=wind_speed, ustar=ustar, mo_len=mol,
        sigma_v=sigma_v, grid_domain=[0, xmx, 0, ymx], grid_res=dx,
        mxy=meas_pt, wd=wd,
    )

    # --- Plotting ---
    X, Y, _ = result_most["grid"]

    fig, axes = plot_footprint_comparison(
        fields=[result_most["flx"], result_mostm["flx"], grid_ffm],
        grids=[(X, Y), (X, Y), (grid_x, grid_y)],
        labels=["BLDFM", "BLDFM-SP", "KM01"],
        meas_pt=meas_pt,
    )

    fig.savefig("plots/comparison_footprint_stable.png", dpi=300)
