"""
Multi-tower example using the high-level interface.

Demonstrates running BLDFM for multiple towers across a short timeseries
using synthetic meteorological data and the config-driven workflow.

Usage:
    python examples/multitower_example.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm import initialize
from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_multitower
from bldfm.plotting import plot_footprint_field
from bldfm.utils import get_logger

logger = get_logger("multitower_example")

config = parse_config_dict(
    {
        "domain": {
            "nx": 64,
            "ny": 128,
            "xmax": 100.0,
            "ymax": 700.0,
            "nz": 16,
            "modes": [64, 128],
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "tower_A", "lat": 50.95, "lon": 11.5867, "z_m": 10.0},
            {"name": "tower_B", "lat": 50.9504, "lon": 11.5867, "z_m": 10.0},
        ],
        "met": {
            "ustar": [0.5, 0.4, 0.6],
            "mol": [-100.0, -200.0, -50.0],
            "wind_speed": [6.0, 5.0, 7.0],
            "wind_dir": [0.0, 15.0, 345.0],
        },
        "solver": {"closure": "MOST", "footprint": True},
    }
)

if __name__ == "__main__":
    initialize()
    logger.info(
        "Running multitower example: %d towers x %d timesteps",
        len(config.towers),
        config.met.n_timesteps,
    )

    results = run_bldfm_multitower(config)

    # Plot first timestep for each tower
    names = [t.name for t in config.towers]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]

    for i, name in enumerate(names):
        flx = results[name][0]["flx"]
        grid = results[name][0]["grid"]
        plot_footprint_field(flx, grid, ax=axes[i], title=name, contour_pcts=[0.5, 0.8])
        axes[i].set_aspect("auto")

    fig.suptitle("Multitower footprints (t=0)")
    fig.savefig(
        "plots/examples_multitower_footprints.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    logger.info("Saved plots/examples_multitower_footprints.png")
