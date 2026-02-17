"""
3D output example using the config-driven workflow.

Computes concentration and flux fields at all vertical levels and
plots vertical slices.

Usage:
    python examples/minimal_example_3d.py

For the equivalent low-level API version, see runs/low_level/minimal_example_3d.py.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm import initialize, load_config, run_bldfm_single
from bldfm.plotting import plot_vertical_slice
from bldfm import config as runtime_config

config_path = Path(__file__).parent / "configs" / "minimal_3d.yaml"

if __name__ == "__main__":
    initialize()
    config = load_config(config_path)
    runtime_config.NUM_THREADS = config.parallel.num_threads

    result = run_bldfm_single(config, config.towers[0])

    conc = result["conc"]
    flx = result["flx"]

    # Vertical slice of concentration at y midpoint
    ny = conc.shape[1]
    fig, ax = plt.subplots()
    plot_vertical_slice(conc, result["grid"], "y", ny // 2, ax=ax,
                        title="Vertical slice of concentration")
    fig.savefig("plots/concentration_vertical_slice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_vertical_slice(flx, result["grid"], "y", ny // 2, ax=ax,
                        title="Vertical slice of kinematic flux")
    fig.savefig("plots/flux_vertical_slice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plots/concentration_vertical_slice.png")
    print("Saved plots/flux_vertical_slice.png")
