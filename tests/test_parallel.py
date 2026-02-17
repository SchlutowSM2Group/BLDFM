"""Tests for parallel execution."""

import numpy as np
import pytest

pytestmark = pytest.mark.parallel

from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_multitower, run_bldfm_parallel
from bldfm.synthetic import generate_synthetic_timeseries, generate_towers_grid


@pytest.fixture
def parallel_config():
    """Config with 2 towers x 2 timesteps for parallel tests."""
    towers = generate_towers_grid(n_towers=2, z_m=10.0, layout="transect", seed=42)
    met = generate_synthetic_timeseries(n_timesteps=2, seed=42)
    return parse_config_dict({
        "domain": {
            "nx": 64, "ny": 64, "xmax": 200.0, "ymax": 200.0, "nz": 8,
            "modes": [64, 64],
            "ref_lat": towers[0]["lat"], "ref_lon": towers[0]["lon"],
        },
        "towers": towers,
        "met": met,
        "solver": {"closure": "MOST", "footprint": True},
        "parallel": {"max_workers": 2},
    })


@pytest.mark.parametrize("strategy", ["towers", "time", "both"])
def test_parallel_matches_serial(parallel_config, strategy):
    serial = run_bldfm_multitower(parallel_config)
    parallel = run_bldfm_parallel(parallel_config, max_workers=2, parallel_over=strategy)

    for name in serial:
        assert name in parallel
        for t in range(len(serial[name])):
            np.testing.assert_allclose(
                serial[name][t]["flx"], parallel[name][t]["flx"], rtol=1e-5
            )


def test_parallel_invalid_strategy(parallel_config):
    with pytest.raises(ValueError, match="Unknown parallel_over"):
        run_bldfm_parallel(parallel_config, parallel_over="invalid")


def test_parallel_result_structure(parallel_config):
    result = run_bldfm_parallel(parallel_config, max_workers=2, parallel_over="towers")
    assert isinstance(result, dict)
    assert len(result) == 2
    for name, tower_results in result.items():
        assert isinstance(tower_results, list)
        assert len(tower_results) == 2
        for r in tower_results:
            assert "flx" in r
            assert "conc" in r
            assert "tower_name" in r
