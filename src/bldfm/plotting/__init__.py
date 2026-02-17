"""
Plotting package for BLDFM footprint visualisation.

All matplotlib-based functions accept an optional ``ax`` parameter for
composability.  Optional dependencies (contextily, windrose, plotly) are
imported lazily and raise helpful messages when missing.
"""

from .footprint import (
    extract_percentile_contour,
    plot_footprint_field,
    plot_footprint_on_map,
)
from .comparison import (
    plot_footprint_comparison,
    plot_field_comparison,
)
from .diagnostics import (
    plot_convergence,
    plot_vertical_profiles,
    plot_vertical_slice,
)
from .timeseries import plot_footprint_timeseries
from .interactive import plot_footprint_interactive
from .meteorology import plot_wind_rose
from .contours import (
    plot_source_area_contours,
    plot_source_area_gallery,
)

# Re-export for monkeypatch compatibility (tests patch "bldfm.plotting._fetch_land_cover")
from ._geo import fetch_land_cover as _fetch_land_cover

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
