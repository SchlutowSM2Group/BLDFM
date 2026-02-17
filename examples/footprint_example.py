"""
Footprint example using the config-driven workflow.

Computes a flux footprint and visualises it with percentile contours.

Usage:
    python examples/footprint_example.py

For the equivalent low-level API version, see examples/low_level/footprint_example.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm import initialize, load_config, run_bldfm_single
from bldfm.plotting import plot_footprint_field

config_path = Path(__file__).parent / "configs" / "footprint.yaml"

if __name__ == "__main__":
    initialize()
    config = load_config(config_path)

    result = run_bldfm_single(config, config.towers[0])

    fig, ax = plt.subplots()
    plot_footprint_field(
        result["flx"],
        result["grid"],
        ax=ax,
        contour_pcts=[0.5, 0.8],
        title="Flux footprint",
    )
    tx, ty = result["tower_xy"]
    ax.plot(
        tx,
        ty,
        "k^",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=5,
    )
    fig.savefig("plots/flux_footprint.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plots/flux_footprint.png")
