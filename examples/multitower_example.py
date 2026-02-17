"""
Multi-tower example using the high-level interface.

Demonstrates running BLDFM for multiple towers across a short timeseries
using synthetic meteorological data and the config-driven workflow.
"""

import matplotlib.pyplot as plt

from bldfm import initialize
from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_multitower
from bldfm.synthetic import generate_synthetic_timeseries, generate_towers_grid
from bldfm.utils import get_logger

logger = get_logger("multitower_example")

# Generate synthetic inputs
towers = generate_towers_grid(n_towers=2, z_m=10.0, layout="transect", seed=42)
met = generate_synthetic_timeseries(n_timesteps=3, seed=42)

config = parse_config_dict({
    "domain": {
        "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
        "modes": [128, 64],
        "ref_lat": towers[0]["lat"], "ref_lon": towers[0]["lon"],
    },
    "towers": towers,
    "met": met,
    "solver": {"closure": "MOST", "footprint": True},
})

if __name__ == "__main__":
    initialize()
    logger.info("Running multitower example: %d towers x %d timesteps",
                len(config.towers), config.met.n_timesteps)

    results = run_bldfm_multitower(config)

    # Plot first timestep for each tower
    fig, axes = plt.subplots(1, len(config.towers), figsize=(6 * len(config.towers), 5))
    if len(config.towers) == 1:
        axes = [axes]

    for ax, tower in zip(axes, config.towers):
        result = results[tower.name][0]  # first timestep
        X, Y, Z = result["grid"]
        ax.pcolormesh(X, Y, result["flx"])
        ax.set_title(f"{tower.name} (t={result['timestamp']})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("plots/multitower_footprints.png")
    logger.info("Saved plots/multitower_footprints.png")
