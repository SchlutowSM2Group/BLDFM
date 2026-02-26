"""Tests for the plotting module."""

import numpy as np
import pytest

pytestmark = pytest.mark.plotting
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

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
    ax.figure.savefig(
        "plots/test_plot_footprint_field_variants_basic.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    # With contours
    ax = plot_footprint_field(result["flx"], result["grid"], contour_pcts=[0.5, 0.8])
    assert ax is not None
    ax.figure.savefig(
        "plots/test_plot_footprint_field_variants_contours.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")

    # Custom axes
    fig, ax = plt.subplots()
    returned_ax = plot_footprint_field(
        result["flx"], result["grid"], ax=ax, title="Test"
    )
    assert returned_ax is ax
    fig.savefig(
        "plots/test_plot_footprint_field_variants_custom_ax.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")


# --- plot_footprint_timeseries ---


def test_plot_footprint_timeseries(timeseries_results_session):
    """Test temporal evolution plot using the shared timeseries fixture."""
    results = timeseries_results_session
    grid = results[0]["grid"]

    ax = plot_footprint_timeseries(
        results, grid, pcts=[0.5, 0.8], title="Footprint timeseries (test)"
    )
    assert ax is not None
    ax.figure.savefig(
        "plots/test_plot_footprint_timeseries.png", dpi=150, bbox_inches="tight"
    )
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


def test_plot_footprint_on_map_happy_path(footprint_result_session):
    """Test map plot with contextily tiles (requires network + contextily)."""
    from bldfm.plotting import plot_footprint_on_map

    try:
        import contextily
    except ImportError:
        pytest.skip("contextily not installed")

    result, config = footprint_result_session
    try:
        ax = plot_footprint_on_map(
            result["flx"],
            result["grid"],
            config,
            contour_pcts=[0.5, 0.8],
            title="Map plot (test)",
        )
    except Exception as exc:
        # Network or PROJ/CRS errors should not fail the test suite
        skip_types = ("URLError", "ConnectionError", "CRSError")
        if any(t in type(exc).__name__ for t in skip_types):
            pytest.skip(f"External dependency error: {exc}")
        raise
    assert ax is not None
    ax.figure.savefig(
        "plots/test_plot_footprint_on_map_happy_path.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")


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


def test_plot_wind_rose_happy_path():
    """Test wind rose plot with synthetic data (requires windrose)."""
    from bldfm.plotting import plot_wind_rose

    try:
        import windrose
    except ImportError:
        pytest.skip("windrose not installed")

    rng = np.random.default_rng(42)
    ws = rng.uniform(1, 8, size=100)
    wd = rng.uniform(0, 360, size=100)

    ax = plot_wind_rose(ws, wd, title="Wind rose (test)")
    assert ax is not None
    ax.figure.savefig(
        "plots/test_plot_wind_rose_happy_path.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")


# --- plot_footprint_interactive ---


def test_plot_footprint_interactive(footprint_result_session):
    from bldfm.plotting import plot_footprint_interactive

    result, _ = footprint_result_session
    try:
        import plotly
    except ImportError:
        pytest.skip("plotly not installed")

    fig = plot_footprint_interactive(
        result["flx"], result["grid"], title="Test interactive"
    )
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
            plot_footprint_on_map(
                result["flx"], result["grid"], config, land_cover=True
            )


def test_plot_footprint_on_map_land_cover(footprint_result_session):
    """Test land cover overlay using the real ESA Terrascope WMS."""
    from bldfm.plotting import plot_footprint_on_map

    result, config = footprint_result_session

    try:
        ax = plot_footprint_on_map(
            result["flx"],
            result["grid"],
            config,
            land_cover=True,
            title="Land cover test",
        )
    except Exception as exc:
        pytest.skip(f"WMS service unavailable: {exc}")

    assert ax is not None
    # Verify legend is present
    legend = ax.get_legend()
    assert legend is not None
    ax.figure.savefig(
        "plots/test_plot_footprint_on_map_land_cover.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")


# --- plot_footprint_comparison ---


def test_plot_footprint_comparison(
    source_area_result_session, footprint_result_session
):
    """Test multi-panel comparison: source area footprint vs concentration."""
    r = source_area_result_session

    # Visual output: side-by-side plot_footprint_field (pcolormesh + contours)
    fig, axes = plt.subplots(1, 2, figsize=(8, 10))
    plot_footprint_field(
        r["flx"],
        r["grid"],
        ax=axes[0],
        contour_pcts=[0.25, 0.5, 0.75],
        title="Footprint",
    )
    plot_footprint_field(
        r["conc"],
        r["grid"],
        ax=axes[1],
        contour_pcts=[0.25, 0.5, 0.75],
        title="Concentration",
    )
    fig.savefig(
        "plots/test_plot_footprint_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")

    # Also exercise the plot_footprint_comparison contour function
    result, _ = footprint_result_session
    X, Y, _ = result["grid"]
    fig2, axes2 = plot_footprint_comparison(
        fields=[result["flx"], result["flx"] * 0.5],
        grids=[(X, Y), (X, Y)],
        labels=["A", "B"],
    )
    assert fig2 is not None
    assert len(axes2) == 2
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
    fig.savefig(
        "plots/test_plot_smoke_field_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")


# --- plot_convergence ---


def test_plot_convergence():
    """Test log-log convergence plot with and without fits."""
    h = np.array([10.0, 5.0, 2.5, 1.25])
    err = np.array([1e-2, 2.5e-3, 6.25e-4, 1.56e-4])

    # Without fits
    ax = plot_convergence(h, err, title="Convergence test")
    assert ax is not None
    ax.figure.savefig(
        "plots/test_plot_smoke_convergence_basic.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")

    # With fits
    ax = plot_convergence(
        h,
        err,
        fits=[
            (lambda x: 1e-4 * x**2, {}, "$O(h^2)$"),
        ],
    )
    assert ax is not None
    ax.figure.savefig(
        "plots/test_plot_smoke_convergence_fits.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")


# --- plot_vertical_profiles ---


def test_plot_vertical_profiles():
    """Test vertical profile plot using real PBL profiles."""
    from bldfm.pbl_model import vertical_profiles

    z1, profs1 = vertical_profiles(32, 10.0, wind=(0, -5), z0=0.5, mol=-10.0)
    z2, profs2 = vertical_profiles(32, 10.0, wind=(0, -5), z0=0.5, mol=100.0)

    fig, axes = plot_vertical_profiles(
        [z1, z2],
        [profs1, profs2],
        labels=["L = -10 m", "L = +100 m"],
        meas_height=10.0,
    )
    assert fig is not None
    assert len(axes) == 2
    fig.savefig("plots/test_plot_vertical_profiles.png", dpi=150, bbox_inches="tight")
    plt.close("all")


# --- plot_vertical_slice ---


def test_plot_vertical_slice(plume_3d_result_session):
    """Test 2D slice from a real 3D plume: concentration and flux side by side."""
    r = plume_3d_result_session
    conc = r["conc"]
    flx = r["flx"]
    grid = r["grid"]
    nlvls, ny, nx = conc.shape

    for axis, idx in [("y", ny // 2), ("x", nx // 2), ("z", 0)]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, field, label in [
            (axes[0], conc, "Concentration"),
            (axes[1], flx, "Flux"),
        ]:
            plot_vertical_slice(field, grid, slice_axis=axis, slice_index=idx, ax=ax)
            ax.set_title(f"{label} ({axis}-slice, idx={idx})")
        fig.savefig(
            f"plots/test_plot_vertical_slice_{axis}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close("all")

    # Invalid axis raises ValueError
    with pytest.raises(ValueError, match="slice_axis"):
        plot_vertical_slice(conc, grid, slice_axis="w", slice_index=0)


# --- get_source_area ---


def test_get_source_area_basic():
    """Test that get_source_area returns correct shape and value range."""
    from bldfm.utils import get_source_area

    rng = np.random.default_rng(42)
    f = rng.random((32, 64))
    f = f / f.sum()  # normalize to unit sum

    rescaled = get_source_area(f, f)
    assert rescaled.shape == f.shape
    assert rescaled.min() >= 0.0
    assert rescaled.max() < 1.0


def test_get_source_area_monotone():
    """Test that higher g values map to lower rescaled values."""
    from bldfm.utils import get_source_area

    f = np.array([[0.1, 0.2], [0.3, 0.4]])
    rescaled = get_source_area(f, f)

    # Highest-f cell (0.4) should get rescaled=0 (nothing above it)
    assert rescaled[1, 1] == 0.0
    # Second-highest (0.3) should get cumsum of just the top cell
    assert np.isclose(rescaled[1, 0], 0.4)


def test_source_area_base_functions_shapes(source_area_result_session):
    """Test that all 5 base function constructors return correct shapes."""
    from bldfm.utils import (
        source_area_contribution,
        source_area_circular,
        source_area_upwind,
        source_area_crosswind,
        source_area_sector,
    )

    r = source_area_result_session
    flx = r["flx"]
    X, Y, Z = r["grid"]
    meas_pt = r["meas_pt"]
    wind = r["wind"]

    for name, g in [
        ("contribution", source_area_contribution(flx)),
        ("circular", source_area_circular(X, Y, meas_pt)),
        ("upwind", source_area_upwind(X, Y, meas_pt, wind)),
        ("crosswind", source_area_crosswind(X, Y, meas_pt, wind)),
        ("sector", source_area_sector(X, Y, meas_pt, wind)),
    ]:
        assert g.shape == flx.shape, f"{name} shape mismatch"


# --- plot_source_area_contours ---


def test_plot_source_area_contours(source_area_result_session):
    """Test source area contour plotting returns axes."""
    from bldfm.utils import get_source_area
    from bldfm.plotting import plot_source_area_contours

    r = source_area_result_session
    rescaled = get_source_area(r["flx"], r["flx"])
    ax = plot_source_area_contours(r["flx"], r["grid"], rescaled, title="Test contours")
    assert ax is not None
    ax.figure.savefig(
        "plots/test_plot_source_area_contours.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")


def test_plot_source_area_contours_custom_ax(source_area_result_session):
    """Test source area contour plotting on provided axes."""
    from bldfm.utils import get_source_area
    from bldfm.plotting import plot_source_area_contours

    r = source_area_result_session
    fig, ax = plt.subplots()
    rescaled = get_source_area(r["flx"], r["flx"])
    returned_ax = plot_source_area_contours(r["flx"], r["grid"], rescaled, ax=ax)
    assert returned_ax is ax
    fig.savefig(
        "plots/test_plot_source_area_contours_custom_ax.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close("all")


# --- plot_source_area_gallery ---


def test_plot_source_area_gallery(source_area_result_session):
    """Test gallery plot creates 2x3 grid with 5 visible panels.

    Uses an elongated domain (100x700m) matching the high-res example
    in runs/low_level/source_area_example.py but at 128x64 resolution.
    """
    from bldfm.plotting import plot_source_area_gallery

    r = source_area_result_session
    fig, axes = plot_source_area_gallery(
        r["flx"],
        r["grid"],
        meas_pt=r["meas_pt"],
        wind=r["wind"],
    )
    assert fig is not None
    assert axes.shape == (2, 3)
    assert not axes[1, 2].get_visible()
    fig.savefig("plots/test_plot_source_area_gallery.png", dpi=150, bbox_inches="tight")
    plt.close("all")


# --- _maybe_slice_level ---


def test_maybe_slice_level_2d_passthrough():
    """2D field and grid pass through unchanged."""
    from bldfm.plotting._common import _maybe_slice_level

    field = np.random.rand(32, 64)
    X, Y = np.meshgrid(np.arange(64), np.arange(32))
    Z = np.zeros_like(X)
    grid = (X, Y, Z)
    out_field, out_grid = _maybe_slice_level(field, grid, level=0)
    assert out_field is field
    assert out_grid is grid


def test_maybe_slice_level_3d_slicing():
    """3D field is sliced correctly at the given level."""
    from bldfm.plotting._common import _maybe_slice_level

    nz, ny, nx = 4, 8, 16
    field = np.random.rand(nz, ny, nx)
    Z, Y, X = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    grid = (X, Y, Z)

    for lvl in range(nz):
        out_field, out_grid = _maybe_slice_level(field, grid, level=lvl)
        assert out_field.shape == (ny, nx)
        np.testing.assert_array_equal(out_field, field[lvl])
        assert out_grid[0].shape == (ny, nx)


# --- 3D input handling in plotting functions ---


def test_plot_footprint_field_3d_input(plume_3d_result_session):
    """plot_footprint_field auto-slices 3D input at requested level."""
    r = plume_3d_result_session
    flx = r["flx"]
    grid = r["grid"]

    ax = plot_footprint_field(flx, grid, level=0, title="3D footprint (level=0)")
    assert ax is not None
    ax.set_aspect("auto")
    ax.figure.savefig(
        "plots/test_plot_footprint_field_3d.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")


def test_extract_percentile_contour_3d_input(plume_3d_result_session):
    """extract_percentile_contour auto-slices 3D input."""
    r = plume_3d_result_session
    flx = r["flx"]
    grid = r["grid"]
    X, Y, Z = grid

    level_val, area = extract_percentile_contour(flx, grid, pct=0.8, level=0)
    assert isinstance(level_val, float)
    assert level_val > 0
    assert area > 0

    # Result should match slicing manually
    level_manual, area_manual = extract_percentile_contour(
        flx[0], (X[0], Y[0], Z[0]), pct=0.8
    )
    assert level_val == level_manual
    assert area == area_manual

    print(
        f"\nPLOTTING percentile_contour_3d: "
        f"80% level={level_val:.4e} area={area:.1f}m²"
    )
