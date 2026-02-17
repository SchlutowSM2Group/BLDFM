"""
Source area contour gallery using the low-level API.

Demonstrates five different base functions for source area contours:
contribution, circular, upwind, crosswind, and sector.

Usage:
    python runs/low_level/source_area_example.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bldfm.pbl_model import vertical_profiles
from bldfm.solver import steady_state_transport_solver
from bldfm.plotting import plot_source_area_gallery

if __name__ == "__main__":
    nx, ny, nz = 512, 256, 32
    domain = 100.0, 700.0
    meas_pt = 50.0, 0.0
    meas_height = 10.0
    wind = 0.0, -6.0
    ustar = 0.5

    area = np.zeros([ny, nx])
    z, profs = vertical_profiles(nz, meas_height, wind, ustar)
    grid, conc, flx = steady_state_transport_solver(
        area, z, profs, domain, nz, meas_pt=meas_pt, footprint=True
    )

    fig, axes = plot_source_area_gallery(flx, grid, meas_pt, wind)
    fig.savefig("plots/source_area_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved plots/source_area_gallery.png")
