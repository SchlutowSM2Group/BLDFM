"""Tests for the high-level interface module."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_single, run_bldfm_multitower
from bldfm.synthetic import generate_synthetic_timeseries, generate_towers_grid


def test_single_run_structure(single_run_result, simple_config_session):
    """Test output structure, shapes, types, and metadata from a single run."""
    result = single_run_result
    config = simple_config_session

    # Dict structure
    assert isinstance(result, dict)
    for key in ("grid", "conc", "flx", "tower_name", "timestamp", "params"):
        assert key in result

    # Output shapes
    assert result["conc"].shape == (config.domain.ny, config.domain.nx)
    assert result["flx"].shape == result["conc"].shape

    # Tower name
    assert result["tower_name"] == "test_tower"


def test_single_run_with_synthetic_data():
    """Integration test: synthetic data -> config -> run."""
    towers = generate_towers_grid(n_towers=1, z_m=5.0, seed=42)
    met = generate_synthetic_timeseries(n_timesteps=2, seed=42)

    config = parse_config_dict(
        {
            "domain": {
                "nx": 64,
                "ny": 64,
                "xmax": 200.0,
                "ymax": 200.0,
                "nz": 8,
                "modes": [64, 64],
                "ref_lat": towers[0]["lat"],
                "ref_lon": towers[0]["lon"],
            },
            "towers": towers[:1],
            "met": met,
            "solver": {"closure": "MOST", "footprint": True},
        }
    )

    result = run_bldfm_single(config, config.towers[0], met_index=0)
    assert result["flx"] is not None
    assert np.isfinite(result["flx"]).all()


def test_timeseries_structure(timeseries_results_session, timeseries_config_session):
    """Test timeseries output: list structure, keys, unique timestamps, finite values."""
    results = timeseries_results_session

    # List structure
    assert isinstance(results, list)
    assert len(results) == 3

    # Keys present in each step
    for r in results:
        for key in ("grid", "conc", "flx", "tower_name", "timestamp"):
            assert key in r

    # Unique timestamps
    timestamps = [r["timestamp"] for r in results]
    assert len(set(timestamps)) == 3

    # Finite outputs
    for r in results:
        assert np.isfinite(r["flx"]).all()
        assert np.isfinite(r["conc"]).all()


def test_multitower_structure(multitower_results_session, timeseries_config_session):
    """Test multitower output: dict keyed by tower names, all timesteps, different footprints."""
    results, config = multitower_results_session

    # Dict with correct keys
    assert isinstance(results, dict)
    assert len(results) == 2
    expected_names = {t.name for t in config.towers}
    assert set(results.keys()) == expected_names

    # Each tower has all timesteps with correct tower_name
    for tower_name, tower_results in results.items():
        assert len(tower_results) == 3
        for r in tower_results:
            assert r["tower_name"] == tower_name

    # Different tower positions produce different footprints
    names = list(results.keys())
    flx_0 = results[names[0]][0]["flx"]
    flx_1 = results[names[1]][0]["flx"]
    assert not np.allclose(flx_0, flx_1)


def test_timeseries_met_params_vary(
    timeseries_results_session, timeseries_config_session
):
    """Test that time-varying ustar, mol, wind_dir, wind_speed propagate to each result."""
    results = timeseries_results_session

    # Each result carries its met params
    for r in results:
        assert "params" in r
        for key in ("ustar", "mol", "wind_speed", "wind_dir"):
            assert key in r["params"], f"Missing '{key}' in params"

    # Met params should vary across timesteps (synthetic timeseries with seed=42)
    for key in ("ustar", "mol", "wind_speed", "wind_dir"):
        values = [r["params"][key] for r in results]
        assert (
            len(set(values)) > 1
        ), f"Expected varying '{key}' across timesteps, got constant {values[0]}"


def test_timeseries_footprints_evolve(
    timeseries_results_session, timeseries_config_session
):
    """Test that changing met conditions produce different footprints at each timestep."""
    results = timeseries_results_session
    config = timeseries_config_session
    expected_shape = (config.domain.ny, config.domain.nx)

    for i, r in enumerate(results):
        # Correct shape
        assert r["flx"].shape == expected_shape, f"Step {i}: wrong flx shape"

        # Grid is a 3-tuple (X, Y, Z)
        assert isinstance(r["grid"], tuple) and len(r["grid"]) == 3

        # Non-trivial footprint (has positive values)
        assert r["flx"].max() > 0, f"Step {i}: footprint is all non-positive"

    # At least one pair of timesteps should produce different footprints
    differs = any(
        not np.allclose(results[0]["flx"], results[i]["flx"])
        for i in range(1, len(results))
    )
    assert (
        differs
    ), "All timestep footprints are identical despite varying met conditions"


def test_timeseries_aggregated_footprint(timeseries_results_session):
    """Test aggregated mean footprint with 50% and 70% flux contribution contours."""
    from bldfm.plotting import extract_percentile_contour

    results = timeseries_results_session
    grid = results[0]["grid"]

    # Compute time-averaged footprint
    mean_flx = np.mean([r["flx"] for r in results], axis=0)
    assert mean_flx.shape == results[0]["flx"].shape
    assert np.isfinite(mean_flx).all()

    # Extract percentile contours at 50% and 70%
    level_50, area_50 = extract_percentile_contour(mean_flx, grid, pct=0.5)
    level_70, area_70 = extract_percentile_contour(mean_flx, grid, pct=0.7)

    # Valid positive values
    assert level_50 > 0 and area_50 > 0
    assert level_70 > 0 and area_70 > 0

    # 70% contour encloses at least as much area as 50%
    assert area_70 >= area_50

    # 50% contour level is at least as high (more concentrated core)
    assert level_50 >= level_70
