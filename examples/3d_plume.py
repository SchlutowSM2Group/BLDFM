"""
3D point-source plume example using the config-driven workflow.

Computes a 3D concentration and flux field from a point source and
plots horizontal and vertical slices of both fields.

Usage:
    python examples/3d_plume.py

For the equivalent low-level API version, see examples/low_level/3d_plume.py.
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
    ny = conc.shape[1]
    nx = conc.shape[2]

    # Horizontal slice at z=0 (ground level): concentration and flux
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_vertical_slice(
        conc, result["grid"], "z", 0, ax=axes[0], title="Concentration at z0"
    )
    plot_vertical_slice(flx, result["grid"], "z", 0, ax=axes[1], title="Flux at z0")
    fig.savefig("plots/examples_ptsrc_xy_slice_at_z0.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Vertical xz slice through the plume centerline
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_vertical_slice(
        conc,
        result["grid"],
        "y",
        ny // 2,
        ax=axes[0],
        title="Concentration (xz slice)",
    )
    plot_vertical_slice(
        flx,
        result["grid"],
        "y",
        ny // 2,
        ax=axes[1],
        title="Flux (xz slice)",
    )
    fig.savefig("plots/examples_ptsrc_xz_slice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Vertical yz slice through the source location
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_vertical_slice(
        conc,
        result["grid"],
        "x",
        nx // 2,
        ax=axes[0],
        title="Concentration (yz slice)",
    )
    plot_vertical_slice(
        flx,
        result["grid"],
        "x",
        nx // 2,
        ax=axes[1],
        title="Flux (yz slice)",
    )
    fig.savefig("plots/examples_ptsrc_yz_slice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plots/examples_ptsrc_xy_slice_at_z0.png")
    print("Saved plots/examples_ptsrc_xz_slice.png")
    print("Saved plots/examples_ptsrc_yz_slice.png")
