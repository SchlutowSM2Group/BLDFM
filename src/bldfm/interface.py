"""
High-level interface for running BLDFM simulations.

Provides convenience functions that wrap the manual workflow of
compute_wind_fields -> vertical_profiles -> steady_state_transport_solver
into single function calls driven by configuration objects.
"""

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from .config_parser import BLDFMConfig, TowerConfig
from .pbl_model import vertical_profiles
from .solver import steady_state_transport_solver
from .utils import compute_wind_fields, ideal_source, get_logger

logger = get_logger("interface")


def _make_cache(config):
    """Create a GreensFunctionCache if caching is enabled."""
    if config.parallel.use_cache and config.solver.footprint:
        from .cache import GreensFunctionCache

        return GreensFunctionCache()
    return None


def run_bldfm_single(
    config: BLDFMConfig,
    tower: TowerConfig,
    met_index: int = 0,
    surface_flux: np.ndarray = None,
    cache=None,
) -> dict:
    """Run a single BLDFM solve for one tower at one timestep.

    Encapsulates the 3-step workflow:
        1. compute_wind_fields(wind_speed, wind_dir) -> (u, v)
        2. vertical_profiles(nz, z_m, (u, v), ...) -> (z, profiles)
        3. steady_state_transport_solver(srf_flx, z, profiles, ...) -> (grid, conc, flx)

    Parameters
    ----------
    config : BLDFMConfig
        Full BLDFM configuration.
    tower : TowerConfig
        Tower to compute footprint/concentration for.
    met_index : int
        Index into the met timeseries (0 for single-timestep configs).
    surface_flux : ndarray, optional
        2D surface flux field. If None, generates an ideal source.
    cache : GreensFunctionCache, optional
        Cache instance for footprint reuse.

    Returns
    -------
    dict
        Result dictionary with keys:
        - grid: (X, Y, Z) coordinate arrays
        - conc: concentration field
        - flx: flux field
        - tower_name: name of the tower
        - timestamp: timestamp or index
        - params: dict of met parameters used
    """
    dom = config.domain
    sol = config.solver
    met_step = config.met.get_step(met_index)

    # Step 1: wind components
    u_wind, v_wind = compute_wind_fields(met_step["wind_speed"], met_step["wind_dir"])

    # Step 2: vertical profiles (z0 takes precedence over ustar)
    z0_val = met_step.get("z0")
    if z0_val is not None:
        z, profiles = vertical_profiles(
            n=dom.nz,
            meas_height=tower.z_m,
            wind=(u_wind, v_wind),
            z0=z0_val,
            mol=met_step["mol"],
            closure=sol.closure,
        )
    else:
        z, profiles = vertical_profiles(
            n=dom.nz,
            meas_height=tower.z_m,
            wind=(u_wind, v_wind),
            ustar=met_step["ustar"],
            mol=met_step["mol"],
            closure=sol.closure,
        )

    # Step 3: surface flux
    if surface_flux is None:
        nxy = (dom.nx, dom.ny)
        domain = (dom.xmax, dom.ymax)
        surface_flux = ideal_source(
            nxy,
            domain,
            src_loc=sol.src_loc,
            shape=sol.surface_flux_shape,
        )

    # Step 4: solve
    levels = dom.output_levels if dom.output_levels else dom.nz
    grid, conc, flx = steady_state_transport_solver(
        srf_flx=surface_flux,
        z=z,
        profiles=profiles,
        domain=(dom.xmax, dom.ymax),
        levels=levels,
        modes=dom.modes,
        meas_pt=(tower.x, tower.y),
        footprint=sol.footprint,
        analytic=sol.analytic,
        halo=dom.halo,
        precision=sol.precision,
        cache=cache,
    )

    return {
        "grid": grid,
        "conc": conc,
        "flx": flx,
        "tower_name": tower.name,
        "tower_xy": (tower.x, tower.y),
        "timestamp": met_step["timestamp"],
        "params": met_step,
    }


def run_bldfm_timeseries(
    config: BLDFMConfig,
    tower: TowerConfig,
    surface_flux: np.ndarray = None,
) -> list:
    """Run BLDFM for all timesteps in the met config for a single tower.

    Parameters
    ----------
    config : BLDFMConfig
        Full BLDFM configuration with timeseries met data.
    tower : TowerConfig
        Tower to compute footprint/concentration for.
    surface_flux : ndarray, optional
        2D surface flux field (reused for all timesteps).

    Returns
    -------
    list of dict
        One result dict per timestep (same format as run_bldfm_single).
    """
    n = config.met.n_timesteps
    logger.info("Running timeseries for tower '%s': %d timesteps", tower.name, n)

    cache = _make_cache(config)
    results = []
    for i in range(n):
        logger.debug("  timestep %d/%d", i + 1, n)
        result = run_bldfm_single(
            config, tower, met_index=i, surface_flux=surface_flux, cache=cache
        )
        results.append(result)

    return results


def run_bldfm_multitower(
    config: BLDFMConfig,
    surface_flux: np.ndarray = None,
) -> dict:
    """Run BLDFM for all towers and all timesteps.

    Parameters
    ----------
    config : BLDFMConfig
        Full BLDFM configuration with towers and met data.
    surface_flux : ndarray, optional
        2D surface flux field (reused for all towers/timesteps).

    Returns
    -------
    dict
        Mapping of tower_name -> list of result dicts (one per timestep).
    """
    logger.info(
        "Running multitower: %d towers x %d timesteps",
        len(config.towers),
        config.met.n_timesteps,
    )

    results = {}
    for tower in config.towers:
        results[tower.name] = run_bldfm_timeseries(
            config, tower, surface_flux=surface_flux
        )

    return results


# --- Worker function for parallel execution (must be top-level for pickling) ---


def _worker_single(args):
    """Worker function for parallel execution of a single (tower, timestep) pair."""
    config, tower, met_index = args
    # Reset inherited state from parent process to avoid fork-safety issues
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from bldfm import config as cfg

    cfg.NUM_THREADS = 1
    from .fft_manager import reset_fft_manager

    reset_fft_manager()
    return run_bldfm_single(config, tower, met_index=met_index)


def _worker_timeseries(args):
    """Worker function for parallel execution of a full timeseries for one tower."""
    config, tower = args
    # Reset inherited state from parent process to avoid fork-safety issues
    os.environ["NUMBA_NUM_THREADS"] = "1"
    from bldfm import config as cfg

    cfg.NUM_THREADS = 1
    from .fft_manager import reset_fft_manager

    reset_fft_manager()
    return tower.name, run_bldfm_timeseries(config, tower)


def run_bldfm_parallel(
    config: BLDFMConfig,
    max_workers: int = None,
    parallel_over: str = "towers",
    surface_flux: np.ndarray = None,
) -> dict:
    """Run BLDFM in parallel using ProcessPoolExecutor.

    Parameters
    ----------
    config : BLDFMConfig
        Full BLDFM configuration with towers and met data.
    max_workers : int, optional
        Number of worker processes. Defaults to config.parallel.max_workers.
    parallel_over : str
        Parallelization strategy:
        - "towers": distribute towers across workers, each runs full timeseries
        - "time": for each tower, distribute timesteps across workers
        - "both": flatten all (tower, timestep) pairs across workers
    surface_flux : ndarray, optional
        2D surface flux field (not passed to subprocesses to avoid large
        serialization; each worker generates its own ideal source).

    Returns
    -------
    dict
        Mapping of tower_name -> list of result dicts (one per timestep).
        Same format as run_bldfm_multitower.
    """
    if surface_flux is not None:
        logger.warning(
            "surface_flux is not passed to worker processes in parallel mode; "
            "each worker generates its own ideal source. The provided array "
            "will be ignored."
        )

    if max_workers is None:
        max_workers = config.parallel.max_workers

    n_towers = len(config.towers)
    n_time = config.met.n_timesteps

    logger.info(
        "Parallel run: %d towers x %d timesteps, %d workers, strategy=%s",
        n_towers,
        n_time,
        max_workers,
        parallel_over,
    )

    if parallel_over == "towers":
        tasks = [(config, tower) for tower in config.towers]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = pool.map(_worker_timeseries, tasks)
        results = {name: res for name, res in futures}

    elif parallel_over == "time":
        results = {}
        for tower in config.towers:
            tasks = [(config, tower, i) for i in range(n_time)]
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                step_results = list(pool.map(_worker_single, tasks))
            results[tower.name] = step_results

    elif parallel_over == "both":
        tasks = []
        for tower in config.towers:
            for i in range(n_time):
                tasks.append((config, tower, i))

        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            flat_results = list(pool.map(_worker_single, tasks))

        results = {}
        idx = 0
        for tower in config.towers:
            results[tower.name] = flat_results[idx : idx + n_time]
            idx += n_time

    else:
        raise ValueError(
            f"Unknown parallel_over={parallel_over!r}. "
            f"Choose 'towers', 'time', or 'both'."
        )

    return results
