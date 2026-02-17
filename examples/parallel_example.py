"""
Parallel execution example using the config-driven workflow.

Runs a high-resolution BLDFM solve with multi-threaded FFT.

Usage:
    python examples/parallel_example.py

    # Or via CLI:
    bldfm run examples/configs/parallel.yaml --plot

For the equivalent low-level API version, see examples/low_level/parallel_example.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm import initialize, load_config, run_bldfm_single
from bldfm.plotting import plot_footprint_field
from bldfm import config as runtime_config

config_path = Path(__file__).parent / "configs" / "parallel.yaml"

if __name__ == "__main__":
    initialize()
    config = load_config(config_path)

    # Apply runtime thread settings from config
    runtime_config.NUM_THREADS = config.parallel.num_threads

    result = run_bldfm_single(config, config.towers[0])

    fig, ax = plt.subplots()
    plot_footprint_field(
        result["conc"],
        result["grid"],
        ax=ax,
        title="Concentration at meas_height (parallel)",
    )
    fig.savefig("plots/concentration_at_meas_height.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved plots/concentration_at_meas_height.png")
