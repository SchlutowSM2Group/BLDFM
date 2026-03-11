"""
Command-line interface for BLDFM.

Usage:
    bldfm run config.yaml
    bldfm run config.yaml --dry-run
    bldfm run config.yaml --plot
"""

import argparse
import os
import sys

from . import initialize
from .config_parser import load_config
from .interface import run_bldfm_single
from abltk.logging import get_logger


def cmd_run(args):
    """Execute a BLDFM run from a YAML config file."""
    initialize()
    logger = get_logger("cli")

    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")
    logger.info(
        f"  Domain: {config.domain.nx}x{config.domain.ny}, nz={config.domain.nz}"
    )
    logger.info(f"  Towers: {len(config.towers)}")
    logger.info(f"  Timesteps: {config.met.n_timesteps}")

    if args.dry_run:
        logger.info("Dry run complete. Config is valid.")
        for tower in config.towers:
            logger.info(
                f"  Tower '{tower.name}': lat={tower.lat}, lon={tower.lon}, "
                f"z_m={tower.z_m}, x={tower.x:.1f}, y={tower.y:.1f}"
            )
        return

    # Apply runtime settings
    from . import config as runtime_config

    runtime_config.NUM_THREADS = config.parallel.num_threads
    runtime_config.MAX_WORKERS = config.parallel.max_workers
    runtime_config.USE_CACHE = config.parallel.use_cache

    # Run for each tower and timestep
    results = []
    for tower in config.towers:
        logger.info(f"Running tower '{tower.name}'...")
        for t in range(config.met.n_timesteps):
            result = run_bldfm_single(config, tower, met_index=t)
            results.append(result)
            logger.info(f"  t={result['timestamp']}: done")

    logger.info("All runs complete.")

    if args.plot:
        _save_plots(results, config, logger)


def _save_plots(results, config, logger):
    """Save concentration and flux/footprint PNGs for each result."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)
    is_footprint = config.solver.footprint

    for result in results:
        name = result["tower_name"]
        ts = result["timestamp"]
        conc = result["conc"]
        flx = result["flx"]
        grid = result["grid"]
        tx, ty = result["tower_xy"]
        is_3d = conc.ndim == 3

        if is_3d:
            _save_3d_plots(plt, conc, flx, grid, name, ts, logger)
        else:
            _save_2d_plots(plt, conc, flx, grid, tx, ty, name, ts, is_footprint, logger)


def _save_2d_plots(plt, conc, flx, grid, tx, ty, name, ts, is_footprint, logger):
    """Save 2D concentration and flux plots."""
    from abltk.plotting import plot_footprint_field

    # Concentration
    fname = f"plots/concentration_{name}_t{ts}.png"
    fig, ax = plt.subplots()
    plot_footprint_field(conc, grid, ax=ax, title=f"Concentration — {name} (t={ts})")
    ax.plot(
        tx,
        ty,
        "r*",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=5,
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {fname}")

    # Flux / footprint
    flx_label = "Footprint" if is_footprint else "Kinematic flux"
    fname = f"plots/flux_{name}_t{ts}.png"
    fig, ax = plt.subplots()
    plot_footprint_field(
        flx,
        grid,
        ax=ax,
        contour_pcts=[0.5, 0.8] if is_footprint else None,
        title=f"{flx_label} — {name} (t={ts})",
    )
    ax.plot(
        tx,
        ty,
        "r*",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=0.8,
        zorder=5,
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {fname}")


def _save_3d_plots(plt, conc, flx, grid, name, ts, logger):
    """Save horizontal and vertical slice plots for 3D results."""
    from abltk.plotting import plot_vertical_slice

    ny = conc.shape[1]
    nx = conc.shape[2]

    # Horizontal slice at z=0
    fname = f"plots/{name}_t{ts}_xy_slice_z0.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_vertical_slice(conc, grid, "z", 0, ax=axes[0], title="Concentration at z0")
    plot_vertical_slice(flx, grid, "z", 0, ax=axes[1], title="Flux at z0")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {fname}")

    # Vertical xz slice at y midpoint
    fname = f"plots/{name}_t{ts}_xz_slice.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_vertical_slice(
        conc, grid, "y", ny // 2, ax=axes[0], title="Concentration (xz slice)"
    )
    plot_vertical_slice(flx, grid, "y", ny // 2, ax=axes[1], title="Flux (xz slice)")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {fname}")

    # Vertical yz slice at x midpoint
    fname = f"plots/{name}_t{ts}_yz_slice.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_vertical_slice(
        conc, grid, "x", nx // 2, ax=axes[0], title="Concentration (yz slice)"
    )
    plot_vertical_slice(flx, grid, "x", nx // 2, ax=axes[1], title="Flux (yz slice)")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved {fname}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="bldfm",
        description="BLDFM - Boundary Layer Footprint Dispersion Model",
    )
    subparsers = parser.add_subparsers(dest="command")

    # 'run' subcommand
    run_parser = subparsers.add_parser("run", help="Run BLDFM from a YAML config file")
    run_parser.add_argument("config", help="Path to YAML configuration file")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running the solver",
    )
    run_parser.add_argument(
        "--plot", action="store_true", help="Save footprint plots to plots/"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
