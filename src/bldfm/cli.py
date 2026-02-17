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
from .utils import get_logger


def cmd_run(args):
    """Execute a BLDFM run from a YAML config file."""
    initialize()
    logger = get_logger("cli")

    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"  Domain: {config.domain.nx}x{config.domain.ny}, nz={config.domain.nz}")
    logger.info(f"  Towers: {len(config.towers)}")
    logger.info(f"  Timesteps: {config.met.n_timesteps}")

    if args.dry_run:
        logger.info("Dry run complete. Config is valid.")
        for tower in config.towers:
            logger.info(f"  Tower '{tower.name}': lat={tower.lat}, lon={tower.lon}, "
                        f"z_m={tower.z_m}, x={tower.x:.1f}, y={tower.y:.1f}")
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
        _save_plots(results, logger)


def _save_plots(results, logger):
    """Save a footprint PNG for each result."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotting import plot_footprint_field

    os.makedirs("plots", exist_ok=True)

    for result in results:
        name = result["tower_name"]
        ts = result["timestamp"]
        fname = f"plots/footprint_{name}_t{ts}.png"

        fig, ax = plt.subplots()
        plot_footprint_field(
            result["flx"], result["grid"],
            ax=ax,
            contour_pcts=[0.5, 0.8],
            title=f"{name} (t={ts})",
        )
        tx, ty = result["tower_xy"]
        ax.plot(tx, ty, "k^", markersize=10, markeredgecolor="white",
                markeredgewidth=1.5, zorder=5)
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
    run_parser.add_argument("--dry-run", action="store_true",
                            help="Validate config without running the solver")
    run_parser.add_argument("--plot", action="store_true",
                            help="Save footprint plots to plots/")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
