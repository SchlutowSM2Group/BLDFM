"""
Minimal BLDFM example using the config-driven workflow.

Computes concentration and flux fields for a neutral boundary layer and
saves plots to the plots/ directory.

Usage:
    python examples/minimal_example.py

For the equivalent low-level API version, see examples/low_level/minimal_example.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm import initialize, load_config, run_bldfm_single
from bldfm.plotting import plot_footprint_field

config_path = Path(__file__).parent / "configs" / "minimal.yaml"

if __name__ == "__main__":
    initialize()
    config = load_config(config_path)

    result = run_bldfm_single(config, config.towers[0])

    fig, ax = plt.subplots()
    plot_footprint_field(
        result["conc"],
        result["grid"],
        ax=ax,
        title=f"Concentration at {config.towers[0].z_m} m",
    )
    fig.savefig("plots/concentration_at_meas_height.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_footprint_field(
        result["flx"],
        result["grid"],
        ax=ax,
        title=f"Vertical kinematic flux at {config.towers[0].z_m} m",
    )
    fig.savefig("plots/kinematic_flux_at_meas_height.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plots/concentration_at_meas_height.png")
    print("Saved plots/kinematic_flux_at_meas_height.png")
