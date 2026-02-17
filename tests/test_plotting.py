"""Tests for the plotting module."""

import numpy as np
import pytest

pytestmark = pytest.mark.plotting
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_timeseries
from bldfm.plotting import (
    extract_percentile_contour,
    plot_footprint_field,
    plot_footprint_timeseries,
)
from bldfm.synthetic import generate_synthetic_timeseries, generate_towers_grid


# --- extract_percentile_contour ---

def test_percentile_contour_properties(footprint_result_session):
    """Test that percentile contours return valid floats with monotonic area."""
    result, _ = footprint_result_session

    # Returns floats
    level_50, area_50 = extract_percentile_contour(result["flx"], result["grid"], 0.5)
    level_80, area_80 = extract_percentile_contour(result["flx"], result["grid"], 0.8)
    assert isinstance(level_50, float)
    assert isinstance(area_50, float)
    assert level_50 > 0
    assert area_50 > 0

    # Higher percentile -> more area
    assert area_80 > area_50

    # Higher percentile -> lower threshold level
    assert level_80 < level_50


# --- plot_footprint_field ---

def test_plot_footprint_field_variants(footprint_result_session):
    """Test footprint field plot: basic, with contours, and on custom axes."""
    result, _ = footprint_result_session

    # Basic
    ax = plot_footprint_field(result["flx"], result["grid"])
    assert ax is not None
    plt.close("all")

    # With contours
    ax = plot_footprint_field(result["flx"], result["grid"], contour_pcts=[0.5, 0.8])
    assert ax is not None
    plt.close("all")

    # Custom axes
    fig, ax = plt.subplots()
    returned_ax = plot_footprint_field(result["flx"], result["grid"], ax=ax, title="Test")
    assert returned_ax is ax
    plt.close("all")


# --- plot_footprint_timeseries ---

def test_plot_footprint_timeseries():
    towers = generate_towers_grid(n_towers=1, z_m=10.0, seed=42)
    met = generate_synthetic_timeseries(n_timesteps=3, seed=42)
    config = parse_config_dict({
        "domain": {
            "nx": 64, "ny": 64, "xmax": 200.0, "ymax": 200.0, "nz": 8,
            "modes": [64, 64],
            "ref_lat": towers[0]["lat"], "ref_lon": towers[0]["lon"],
        },
        "towers": towers[:1],
        "met": met,
        "solver": {"closure": "MOST", "footprint": True},
    })
    results = run_bldfm_timeseries(config, config.towers[0])
    grid = results[0]["grid"]

    ax = plot_footprint_timeseries(results, grid, pcts=[0.5, 0.8])
    assert ax is not None
    plt.close("all")


# --- plot_footprint_on_map (skip if contextily not installed) ---

def test_plot_footprint_on_map_import_error(footprint_result_session):
    """Should raise ImportError with helpful message if contextily missing."""
    from bldfm.plotting import plot_footprint_on_map
    result, config = footprint_result_session
    try:
        import contextily
        pytest.skip("contextily is installed")
    except ImportError:
        with pytest.raises(ImportError, match="contextily"):
            plot_footprint_on_map(result["flx"], result["grid"], config)


# --- plot_wind_rose (skip if windrose not installed) ---

def test_plot_wind_rose_import_error():
    """Should raise ImportError with helpful message if windrose missing."""
    from bldfm.plotting import plot_wind_rose
    try:
        import windrose
        pytest.skip("windrose is installed")
    except ImportError:
        with pytest.raises(ImportError, match="windrose"):
            plot_wind_rose([1, 2, 3], [90, 180, 270])


# --- plot_footprint_interactive ---

def test_plot_footprint_interactive(footprint_result_session):
    from bldfm.plotting import plot_footprint_interactive
    result, _ = footprint_result_session
    try:
        import plotly
    except ImportError:
        pytest.skip("plotly not installed")

    fig = plot_footprint_interactive(result["flx"], result["grid"],
                                     title="Test interactive")
    assert fig is not None
    assert hasattr(fig, "to_html")
    plt.close("all")
