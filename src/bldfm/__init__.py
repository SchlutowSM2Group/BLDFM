from pathlib import Path

from .solver import steady_state_transport_solver
from .utils import (
    compute_wind_fields,
    ideal_source,
    setup_logging,
    get_logger,
)

_initialized = False


def initialize(log_dir="logs", plot_dir="plots", **logging_kwargs):
    """Initialize BLDFM runtime: create output directories and configure logging.

    Call this explicitly before running simulations. The CLI and high-level
    interface functions call this automatically.
    """
    global _initialized
    if _initialized:
        return
    Path(log_dir).mkdir(exist_ok=True)
    Path(plot_dir).mkdir(exist_ok=True)
    setup_logging(log_dir=log_dir, **logging_kwargs)
    _initialized = True


from .config_parser import load_config, parse_config_dict, BLDFMConfig
from .interface import (
    run_bldfm_single, run_bldfm_timeseries, run_bldfm_multitower, run_bldfm_parallel,
)
from .synthetic import generate_synthetic_timeseries, generate_towers_grid
from .cache import GreensFunctionCache
from .io import save_footprints_to_netcdf, load_footprints_from_netcdf
from .plotting import (
    plot_footprint_field,
    plot_footprint_on_map,
    plot_wind_rose,
    extract_percentile_contour,
    plot_footprint_timeseries,
    plot_footprint_interactive,
    plot_footprint_comparison,
    plot_field_comparison,
    plot_convergence,
    plot_vertical_profiles,
    plot_vertical_slice,
)

__all__ = [
    "initialize",
    "steady_state_transport_solver",
    "compute_wind_fields",
    "ideal_source",
    "setup_logging",
    "get_logger",
    "load_config",
    "parse_config_dict",
    "BLDFMConfig",
    "run_bldfm_single",
    "run_bldfm_timeseries",
    "run_bldfm_multitower",
    "run_bldfm_parallel",
    "GreensFunctionCache",
    "save_footprints_to_netcdf",
    "load_footprints_from_netcdf",
    "generate_synthetic_timeseries",
    "generate_towers_grid",
    "plot_footprint_field",
    "plot_footprint_on_map",
    "plot_wind_rose",
    "extract_percentile_contour",
    "plot_footprint_timeseries",
    "plot_footprint_interactive",
    "plot_footprint_comparison",
    "plot_field_comparison",
    "plot_convergence",
    "plot_vertical_profiles",
    "plot_vertical_slice",
]
