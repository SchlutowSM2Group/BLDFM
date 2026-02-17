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
