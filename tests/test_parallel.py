"""Tests for parallel execution."""

import os
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.parallel

from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_multitower, run_bldfm_parallel
from bldfm.synthetic import generate_synthetic_timeseries, generate_towers_grid


class _TrackedPool(ProcessPoolExecutor):
    """Wrapper that records pool usage for diagnostics."""

    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_workers_requested = kwargs.get(
            "max_workers", args[0] if args else None
        )
        self._map_calls = []
        _TrackedPool.instances.append(self)

    def map(self, fn, *iterables, **kwargs):
        items = [list(it) for it in iterables]
        n_tasks = len(items[0]) if items else 0
        self._map_calls.append({"fn": fn.__name__, "n_tasks": n_tasks})
        return super().map(fn, *[iter(i) for i in items], **kwargs)


@pytest.fixture(autouse=True)
def track_pool():
    """Patch ProcessPoolExecutor to print diagnostics after each test."""
    _TrackedPool.instances.clear()
    with patch("bldfm.interface.ProcessPoolExecutor", _TrackedPool):
        yield
    if _TrackedPool.instances:
        for i, pool in enumerate(_TrackedPool.instances):
            for call in pool._map_calls:
                print(
                    f"  Pool #{i + 1}: max_workers={pool._max_workers_requested}, "
                    f"fn={call['fn']}, tasks={call['n_tasks']}, "
                    f"pid={os.getpid()}"
                )


@pytest.fixture
def parallel_config():
    """Config with 2 towers x 2 timesteps for parallel tests."""
    towers = generate_towers_grid(n_towers=2, z_m=10.0, layout="transect", seed=42)
    met = generate_synthetic_timeseries(n_timesteps=2, seed=42)
    return parse_config_dict(
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
            "towers": towers,
            "met": met,
            "solver": {"closure": "MOST", "footprint": True},
            "parallel": {"max_workers": 2},
        }
    )


@pytest.mark.parametrize("strategy", ["towers", "time", "both"])
def test_parallel_matches_serial(parallel_config, strategy):
    serial = run_bldfm_multitower(parallel_config)
    parallel = run_bldfm_parallel(
        parallel_config, max_workers=2, parallel_over=strategy
    )

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
