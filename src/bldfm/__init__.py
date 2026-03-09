from pathlib import Path

from .solver import steady_state_transport_solver
from .utils import compute_wind_fields, ideal_source

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
    from abltk.logging import setup_logging

    setup_logging(log_dir=log_dir, **logging_kwargs)
    _initialized = True


from .config_parser import load_config, parse_config_dict, BLDFMConfig
from .interface import (
    run_bldfm_single,
    run_bldfm_timeseries,
    run_bldfm_multitower,
    run_bldfm_parallel,
)
from .synthetic import generate_synthetic_timeseries, generate_towers_grid
from .cache import GreensFunctionCache

# ---------------------------------------------------------------------------
# Lazy deprecation for migrated symbols
# ---------------------------------------------------------------------------
import warnings as _warnings

_MIGRATED = {
    # logging (Phase 1)
    "setup_logging": ("abltk.logging", "setup_logging"),
    "get_logger": ("abltk.logging", "get_logger"),
    # source area (Phase 3)
    "get_source_area": ("abltk.plotting.source_area", "get_source_area"),
    "source_area_contribution": (
        "abltk.plotting.source_area",
        "source_area_contribution",
    ),
    "source_area_circular": ("abltk.plotting.source_area", "source_area_circular"),
    "source_area_upwind": ("abltk.plotting.source_area", "source_area_upwind"),
    "source_area_crosswind": ("abltk.plotting.source_area", "source_area_crosswind"),
    "source_area_sector": ("abltk.plotting.source_area", "source_area_sector"),
    # I/O (Phase 2)
    "save_footprints_to_netcdf": ("abltk.io.netcdf", "save_footprints_to_netcdf"),
    "load_footprints_from_netcdf": ("abltk.io.netcdf", "load_footprints_from_netcdf"),
    # plotting (Phase 3)
    "plot_footprint_field": ("abltk.plotting", "plot_footprint_field"),
    "plot_footprint_on_map": ("abltk.plotting", "plot_footprint_on_map"),
    "plot_wind_rose": ("abltk.plotting", "plot_wind_rose"),
    "extract_percentile_contour": ("abltk.plotting", "extract_percentile_contour"),
    "plot_footprint_timeseries": ("abltk.plotting", "plot_footprint_timeseries"),
    "plot_footprint_interactive": ("abltk.plotting", "plot_footprint_interactive"),
    "plot_footprint_comparison": ("abltk.plotting", "plot_footprint_comparison"),
    "plot_field_comparison": ("abltk.plotting", "plot_field_comparison"),
    "plot_convergence": ("abltk.plotting", "plot_convergence"),
    "plot_vertical_profiles": ("abltk.plotting", "plot_vertical_profiles"),
    "plot_vertical_slice": ("abltk.plotting", "plot_vertical_slice"),
    "plot_source_area_contours": ("abltk.plotting", "plot_source_area_contours"),
    "plot_source_area_gallery": ("abltk.plotting", "plot_source_area_gallery"),
}


def __getattr__(name):
    if name in _MIGRATED:
        mod_path, attr = _MIGRATED[name]
        _warnings.warn(
            f"bldfm.{name} is deprecated, use {mod_path}.{attr}",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'bldfm' has no attribute {name!r}")


__all__ = [
    "initialize",
    "steady_state_transport_solver",
    "compute_wind_fields",
    "ideal_source",
    "load_config",
    "parse_config_dict",
    "BLDFMConfig",
    "run_bldfm_single",
    "run_bldfm_timeseries",
    "run_bldfm_multitower",
    "run_bldfm_parallel",
    "GreensFunctionCache",
    "generate_synthetic_timeseries",
    "generate_towers_grid",
    # Migrated (still in __all__ for discoverability, loaded lazily)
    "setup_logging",
    "get_logger",
    "get_source_area",
    "source_area_contribution",
    "source_area_circular",
    "source_area_upwind",
    "source_area_crosswind",
    "source_area_sector",
    "save_footprints_to_netcdf",
    "load_footprints_from_netcdf",
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
    "plot_source_area_contours",
    "plot_source_area_gallery",
]
