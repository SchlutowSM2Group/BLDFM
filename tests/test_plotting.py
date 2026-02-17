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
    plot_footprint_comparison,
    plot_field_comparison,
    plot_convergence,
    plot_vertical_profiles,
    plot_vertical_slice,
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


# --- plot_footprint_on_map: land_cover ---

def test_plot_footprint_on_map_land_cover_import_error(footprint_result_session):
    """Should raise ImportError with helpful message if owslib missing."""
    from bldfm.plotting import plot_footprint_on_map
    result, config = footprint_result_session
    try:
        import owslib
        pytest.skip("owslib is installed")
    except ImportError:
        with pytest.raises(ImportError, match="owslib"):
            plot_footprint_on_map(result["flx"], result["grid"], config,
                                  land_cover=True)


def test_plot_footprint_on_map_land_cover_mock(footprint_result_session, monkeypatch):
    """Test land cover overlay with mocked WMS response."""
    from bldfm.plotting import plot_footprint_on_map
    result, config = footprint_result_session

    fake_img = np.random.rand(64, 64, 4).astype(np.float32)

    def mock_fetch(bbox, size=(512, 512)):
        extent = (bbox[0], bbox[2], bbox[1], bbox[3])
        return fake_img, extent

    monkeypatch.setattr("bldfm.plotting._fetch_land_cover", mock_fetch)

    ax = plot_footprint_on_map(result["flx"], result["grid"], config,
                                land_cover=True, title="Land cover test")
    assert ax is not None
    # Verify legend is present
    legend = ax.get_legend()
    assert legend is not None
    plt.close("all")


# --- plot_footprint_comparison ---

def test_plot_footprint_comparison(footprint_result_session):
    """Test multi-panel comparison plot."""
    result, _ = footprint_result_session
    flx = result["flx"]
    X, Y, _ = result["grid"]
    grid_2d = (X, Y)

    fig, axes = plot_footprint_comparison(
        fields=[flx, flx * 0.5],
        grids=[grid_2d, grid_2d],
        labels=["Model A", "Model B"],
        meas_pt=(100.0, 100.0),
    )
    assert fig is not None
    assert len(axes) == 2
    plt.close("all")


# --- plot_field_comparison ---

def test_plot_field_comparison():
    """Test 2x2 field comparison with synthetic data."""
    rng = np.random.default_rng(42)
    shape = (32, 64)
    fields = {
        "conc": rng.random(shape),
        "flx": rng.random(shape),
        "conc_ref": rng.random(shape),
        "flx_ref": rng.random(shape),
    }

    fig, axs = plot_field_comparison(fields, domain=(200, 100), src_pt=(10, 10))
    assert fig is not None
    assert axs.shape == (2, 2)
    plt.close("all")


# --- plot_convergence ---

def test_plot_convergence():
    """Test log-log convergence plot with and without fits."""
    h = np.array([10.0, 5.0, 2.5, 1.25])
    err = np.array([1e-2, 2.5e-3, 6.25e-4, 1.56e-4])

    # Without fits
    ax = plot_convergence(h, err, title="Convergence test")
    assert ax is not None
    plt.close("all")

    # With fits
    ax = plot_convergence(h, err, fits=[
        (lambda x: 1e-4 * x**2, {}, "$O(h^2)$"),
    ])
    assert ax is not None
    plt.close("all")


# --- plot_vertical_profiles ---

def test_plot_vertical_profiles():
    """Test vertical profile plot using real PBL profiles."""
    from bldfm.pbl_model import vertical_profiles

    z1, profs1 = vertical_profiles(32, 10.0, wind=(0, -5), z0=0.5, mol=-10.0)
    z2, profs2 = vertical_profiles(32, 10.0, wind=(0, -5), z0=0.5, mol=100.0)

    fig, axes = plot_vertical_profiles(
        [z1, z2], [profs1, profs2],
        labels=["L = -10 m", "L = +100 m"],
        meas_height=10.0,
    )
    assert fig is not None
    assert len(axes) == 2
    plt.close("all")


# --- plot_vertical_slice ---

def test_plot_vertical_slice():
    """Test 2D slice from a 3D field for all axes."""
    nz, ny, nx = 4, 8, 16
    field = np.random.rand(nz, ny, nx)
    X, Y, Z = np.meshgrid(
        np.linspace(0, 100, nx),
        np.linspace(0, 50, ny),
        np.linspace(0, 20, nz),
        indexing="ij",
    )
    # Transpose to (nz, ny, nx) ordering for grid arrays
    X = X.transpose(2, 1, 0)
    Y = Y.transpose(2, 1, 0)
    Z = Z.transpose(2, 1, 0)
    grid = (X, Y, Z)

    for axis in ("x", "y", "z"):
        ax = plot_vertical_slice(field, grid, slice_axis=axis, slice_index=0)
        assert ax is not None
        plt.close("all")

    # Invalid axis raises ValueError
    with pytest.raises(ValueError, match="slice_axis"):
        plot_vertical_slice(field, grid, slice_axis="w", slice_index=0)
