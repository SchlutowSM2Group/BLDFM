"""
Plotting package for BLDFM footprint visualisation.

.. deprecated::
    All plotting functions have moved to ``abltk.plotting``.
    Import from there directly. These wrappers will be removed in a future release.
"""

import warnings


def _make_wrapper(name, new_module, new_name=None):
    """Create a deprecation wrapper that delegates to abltk.plotting."""
    actual_name = new_name or name

    def wrapper(*args, **kwargs):
        warnings.warn(
            f"bldfm.plotting.{name} is deprecated, "
            f"use abltk.plotting.{new_module}.{actual_name}",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        mod = importlib.import_module(f"abltk.plotting.{new_module}")
        return getattr(mod, actual_name)(*args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    return wrapper


extract_percentile_contour = _make_wrapper("extract_percentile_contour", "footprint")
plot_footprint_field = _make_wrapper("plot_footprint_field", "footprint")
plot_footprint_comparison = _make_wrapper("plot_footprint_comparison", "comparison")
plot_field_comparison = _make_wrapper("plot_field_comparison", "comparison")
plot_convergence = _make_wrapper("plot_convergence", "diagnostics_plot")
plot_vertical_profiles = _make_wrapper("plot_vertical_profiles", "diagnostics_plot")
plot_vertical_slice = _make_wrapper("plot_vertical_slice", "diagnostics_plot")
plot_footprint_timeseries = _make_wrapper(
    "plot_footprint_timeseries", "timeseries_footprint"
)
plot_footprint_interactive = _make_wrapper("plot_footprint_interactive", "interactive")
plot_wind_rose = _make_wrapper("plot_wind_rose", "meteorology")
plot_source_area_contours = _make_wrapper("plot_source_area_contours", "contours")
plot_source_area_gallery = _make_wrapper("plot_source_area_gallery", "contours")


# Special case: plot_footprint_on_map needs config unpacking
def plot_footprint_on_map(flx, grid, config, **kwargs):
    """Deprecated wrapper that unpacks config for the new signature."""
    warnings.warn(
        "bldfm.plotting.plot_footprint_on_map is deprecated, "
        "use abltk.plotting.footprint.plot_footprint_on_map",
        DeprecationWarning,
        stacklevel=2,
    )
    from abltk.plotting.footprint import plot_footprint_on_map as _new

    return _new(
        flx,
        grid,
        ref_lat=config.domain.ref_lat,
        ref_lon=config.domain.ref_lon,
        towers=config.towers,
        **kwargs,
    )


# Re-export for monkeypatch compatibility
from abltk.plotting.geo import fetch_land_cover as _fetch_land_cover


__all__ = [
    "extract_percentile_contour",
    "plot_footprint_field",
    "plot_footprint_on_map",
    "plot_footprint_comparison",
    "plot_field_comparison",
    "plot_convergence",
    "plot_vertical_profiles",
    "plot_vertical_slice",
    "plot_footprint_timeseries",
    "plot_footprint_interactive",
    "plot_wind_rose",
    "plot_source_area_contours",
    "plot_source_area_gallery",
]
