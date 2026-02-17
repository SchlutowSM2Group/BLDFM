"""
3D point-source plume example using the config-driven workflow.

Computes a 3D concentration and flux field from a point source and
plots horizontal and vertical slices.

Usage:
    python examples/3d_plume.py

For the equivalent low-level API version, see runs/low_level/3d_plume.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm import initialize, load_config, run_bldfm_single
from bldfm.plotting import plot_vertical_slice
from bldfm import config as runtime_config

config_path = Path(__file__).parent / "configs" / "3d_plume.yaml"

if __name__ == "__main__":
    initialize()
    config = load_config(config_path)
    runtime_config.NUM_THREADS = config.parallel.num_threads

    result = run_bldfm_single(config, config.towers[0])

    X, Y, Z = result["grid"]
    conc = result["conc"]
    flx = result["flx"]

    # Horizontal slice at z=0 (ground level)
    fig, ax = plt.subplots()
    plot_vertical_slice(conc, result["grid"], "z", 0, ax=ax,
                        title="Concentration at z0 (horizontal)")
    fig.savefig("plots/ptsrc_concentration_xy_slice_at_z0.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Vertical slice through the plume centerline (xz plane)
    ny = conc.shape[1]
    fig, ax = plt.subplots()
    plot_vertical_slice(conc, result["grid"], "y", ny // 2, ax=ax,
                        title="Concentration (vertical xz slice)")
    fig.savefig("plots/ptsrc_concentration_xz_slice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plots/ptsrc_concentration_xy_slice_at_z0.png")
    print("Saved plots/ptsrc_concentration_xz_slice.png")
