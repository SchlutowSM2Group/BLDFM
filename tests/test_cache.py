"""Tests for Green's function caching."""

import tempfile
import numpy as np
import pytest

pytestmark = pytest.mark.unit

from bldfm.cache import GreensFunctionCache
from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_single


@pytest.fixture
def cache(tmp_path):
    return GreensFunctionCache(cache_dir=tmp_path / "cache")


@pytest.fixture
def solver_inputs():
    """Minimal inputs matching what the cache hashes."""
    z = np.linspace(0.1, 10.0, 8)
    profiles = (
        np.ones(8),
        np.zeros(8),  # u, v
        np.ones(8) * 0.5,
        np.ones(8) * 0.5,
        np.ones(8) * 0.5,  # Kx, Ky, Kz
    )
    domain = (200.0, 200.0)
    modes = (64, 64)
    meas_pt = (0.0, 0.0)
    halo = None
    precision = "single"
    return z, profiles, domain, modes, meas_pt, halo, precision


def test_cache_miss(cache, solver_inputs):
    result = cache.get(*solver_inputs)
    assert result is None


def test_cache_put_and_hit(cache, solver_inputs):
    grid = (np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 4)))
    conc = np.random.rand(4, 4)
    flx = np.random.rand(4, 4)

    cache.put(*solver_inputs, grid, conc, flx)

    result = cache.get(*solver_inputs)
    assert result is not None
    cached_grid, cached_conc, cached_flx = result
    np.testing.assert_array_equal(cached_conc, conc)
    np.testing.assert_array_equal(cached_flx, flx)


def test_cache_different_inputs_miss(cache, solver_inputs):
    grid = (np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 4)))
    conc = np.random.rand(4, 4)
    flx = np.random.rand(4, 4)

    cache.put(*solver_inputs, grid, conc, flx)

    # Change meas_pt
    z, profiles, domain, modes, _, halo, precision = solver_inputs
    result = cache.get(z, profiles, domain, modes, (10.0, 10.0), halo, precision)
    assert result is None


def test_cache_clear(cache, solver_inputs):
    grid = (np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 4)))
    cache.put(*solver_inputs, grid, np.ones((4, 4)), np.ones((4, 4)))

    cache.clear()
    assert cache.get(*solver_inputs) is None


def test_cache_integration():
    """Test caching through the full solver with footprint=True."""
    config = parse_config_dict(
        {
            "domain": {
                "nx": 64,
                "ny": 64,
                "xmax": 200.0,
                "ymax": 200.0,
                "nz": 8,
                "modes": [64, 64],
                "ref_lat": 50.95,
                "ref_lon": 11.586,
            },
            "towers": [
                {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
            ],
            "met": {"ustar": 0.4, "mol": -100.0, "wind_speed": 5.0, "wind_dir": 270.0},
            "solver": {"closure": "MOST", "footprint": True},
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = GreensFunctionCache(cache_dir=tmpdir)

        # First run: cache miss
        r1 = run_bldfm_single(config, config.towers[0], cache=cache)

        # Second run: cache hit (same inputs)
        r2 = run_bldfm_single(config, config.towers[0], cache=cache)

        np.testing.assert_array_equal(r1["flx"], r2["flx"])
        np.testing.assert_array_equal(r1["conc"], r2["conc"])
