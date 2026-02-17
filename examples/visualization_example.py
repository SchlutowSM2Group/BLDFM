"""
Visualization showcase: optional-dependency plotting features.

Demonstrates four plotting features that require optional packages.
Each section is wrapped in try/except so the script runs regardless
of which optional dependencies are installed.

Optional packages: contextily, owslib, windrose, plotly

Usage:
    python examples/visualization_example.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bldfm import initialize, load_config, run_bldfm_single
from bldfm.synthetic import generate_synthetic_timeseries

config_path = Path(__file__).parent / "configs" / "visualization.yaml"

if __name__ == "__main__":
    initialize()
    config = load_config(config_path)
    result = run_bldfm_single(config, config.towers[0])

    # ---- 1. Map overlay (requires contextily) ----
    try:
        from bldfm import plot_footprint_on_map

        ax = plot_footprint_on_map(
            result["flx"],
            result["grid"],
            config,
            tower=config.towers[0],
            contour_pcts=[0.5, 0.8],
            title="Footprint on map tiles",
        )
        ax.figure.savefig("plots/viz_map_overlay.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print("Saved plots/viz_map_overlay.png")
    except ImportError as e:
        print(f"Skipping map overlay (install contextily): {e}")

    # ---- 2. Land cover overlay (requires owslib) ----
    try:
        from bldfm import plot_footprint_on_map

        ax = plot_footprint_on_map(
            result["flx"],
            result["grid"],
            config,
            tower=config.towers[0],
            contour_pcts=[0.5, 0.8],
            land_cover=True,
            title="Footprint on ESA WorldCover 2021",
        )
        ax.figure.savefig("plots/viz_land_cover.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print("Saved plots/viz_land_cover.png")
    except ImportError as e:
        print(f"Skipping land cover overlay (install owslib): {e}")

    # ---- 3. Wind rose (requires windrose) ----
    try:
        from bldfm import plot_wind_rose

        met_ts = generate_synthetic_timeseries(n_timesteps=100, seed=42)
        ax = plot_wind_rose(
            met_ts["wind_speed"],
            met_ts["wind_dir"],
            title="Synthetic wind rose",
        )
        ax.figure.savefig("plots/viz_wind_rose.png", dpi=150, bbox_inches="tight")
        plt.close(ax.figure)
        print("Saved plots/viz_wind_rose.png")
    except ImportError as e:
        print(f"Skipping wind rose (install windrose): {e}")

    # ---- 4. Interactive plot (requires plotly) ----
    try:
        from bldfm import plot_footprint_interactive

        fig = plot_footprint_interactive(
            result["flx"],
            result["grid"],
            title="Interactive footprint — open in browser",
            xlim=(0, config.domain.xmax),
            ylim=(0, config.domain.ymax),
        )
        fig.write_html("plots/viz_interactive.html")
        print("Saved plots/viz_interactive.html")
    except ImportError as e:
        print(f"Skipping interactive plot (install plotly): {e}")
