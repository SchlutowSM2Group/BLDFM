"""
Source area contour examples using the low-level API.

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
from bldfm.utils import (
    get_source_area,
    source_area_contribution,
    source_area_circular,
    source_area_upwind,
    source_area_crosswind,
    source_area_sector,
)
from bldfm.plotting import plot_source_area_contours, plot_source_area_gallery

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
    X, Y, Z = grid

    # --- Individual plots ---
    contour_types = [
        ("Contribution", source_area_contribution(flx)),
        ("Circular", source_area_circular(X, Y, meas_pt)),
        ("Upwind", source_area_upwind(X, Y, meas_pt, wind)),
        ("Crosswind", source_area_crosswind(X, Y, meas_pt, wind)),
        ("Sector", source_area_sector(X, Y, meas_pt, wind)),
    ]

    for name, g in contour_types:
        rescaled = get_source_area(flx, g)
        ax = plot_source_area_contours(flx, grid, rescaled, title=f"{name} contours")
        ax.figure.savefig(
            f"plots/source_area_{name.lower()}.png", dpi=150, bbox_inches="tight"
        )
        plt.close("all")
        print(f"Saved plots/source_area_{name.lower()}.png")

    # --- Gallery plot ---
    fig, axes = plot_source_area_gallery(flx, grid, meas_pt, wind)
    fig.savefig("plots/source_area_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved plots/source_area_gallery.png")
