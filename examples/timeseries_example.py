"""
Single-tower timeseries example -- the recommended BLDFM workflow.

Demonstrates the full analysis pipeline for a flux footprint climatology:
load config -> run timeseries -> aggregate footprint -> save NetCDF -> plot.

This is the primary use case for BLDFM: one eddy-covariance tower with
half-hourly meteorological forcing producing a footprint climatology.

Usage:
    python examples/timeseries_example.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm import (
    initialize,
    load_config,
    run_bldfm_timeseries,
    save_footprints_to_netcdf,
)
from bldfm.plotting import (
    plot_footprint_field,
    plot_footprint_timeseries,
    extract_percentile_contour,
)

config_path = Path(__file__).parent / "configs" / "timeseries.yaml"

if __name__ == "__main__":
    initialize()

    # --- 1. Load configuration ---
    config = load_config(config_path)
    tower = config.towers[0]
    print(f"Tower: {tower.name} at ({tower.lat}, {tower.lon}), z_m={tower.z_m} m")
    print(f"Timesteps: {config.met.n_timesteps}")

    # --- 2. Run timeseries ---
    results = run_bldfm_timeseries(config, tower)
    grid = results[0]["grid"]

    # --- 3. Compute aggregated (time-averaged) footprint ---
    mean_flx = np.mean([r["flx"] for r in results], axis=0)

    # --- 4. Save to NetCDF ---
    # save_footprints_to_netcdf expects multitower format: {name: [results]}
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "timeseries_example.nc"
    save_footprints_to_netcdf({tower.name: results}, config, output_path)
    print(f"Saved {output_path}")

    # --- 5. Plot temporal evolution of footprint extent ---
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_footprint_timeseries(
        results,
        grid,
        pcts=[0.5, 0.8],
        ax=ax,
        title="Footprint area over time",
    )
    fig.savefig("plots/examples_timeseries_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved plots/examples_timeseries_evolution.png")

    # --- 6. Plot individual timestep footprints ---
    n = len(results)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.asarray(axes).ravel()
    for i, r in enumerate(results):
        plot_footprint_field(
            r["flx"],
            r["grid"],
            ax=axes_flat[i],
            contour_pcts=[0.5, 0.8],
            title=r.get("timestamp", f"t={i}"),
        )
        axes_flat[i].set_aspect("auto")
    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Footprint evolution (individual timesteps)")
    fig.savefig(
        "plots/examples_timeseries_footprints.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("Saved plots/examples_timeseries_footprints.png")

    # --- 7. Plot aggregated footprint ---
    fig, ax = plt.subplots(figsize=(5, 8))
    plot_footprint_field(
        mean_flx,
        grid,
        ax=ax,
        contour_pcts=[0.5, 0.7, 0.9],
        title="Time-averaged footprint",
    )
    ax.set_aspect("auto")
    for pct in [0.5, 0.7, 0.9]:
        _, area = extract_percentile_contour(mean_flx, grid, pct)
        print(f"  {int(pct * 100)}% source area: {area:.0f} m²")
    fig.savefig(
        "plots/examples_timeseries_aggregated.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print("Saved plots/examples_timeseries_aggregated.png")
