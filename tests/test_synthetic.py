"""Tests for synthetic data generators."""

import pytest

pytestmark = pytest.mark.unit
from bldfm.synthetic import generate_synthetic_timeseries, generate_towers_grid


# --- generate_synthetic_timeseries ---

class TestSyntheticTimeseries:

    def test_value_ranges(self):
        ts = generate_synthetic_timeseries(
            n_timesteps=100,
            ustar_range=(0.1, 0.8),
            wind_speed_range=(1.0, 8.0),
            seed=42,
        )
        assert all(0.1 <= u <= 0.8 for u in ts["ustar"])
        assert all(1.0 <= ws <= 8.0 for ws in ts["wind_speed"])
        assert all(0.0 <= wd < 360.0 for wd in ts["wind_dir"])

    def test_reproducibility(self):
        ts1 = generate_synthetic_timeseries(n_timesteps=10, seed=123)
        ts2 = generate_synthetic_timeseries(n_timesteps=10, seed=123)
        assert ts1["ustar"] == ts2["ustar"]
        assert ts1["mol"] == ts2["mol"]

    def test_timestamps_format(self):
        ts = generate_synthetic_timeseries(
            n_timesteps=3,
            start_time="2024-06-15T12:00",
            dt_minutes=30,
            seed=42,
        )
        assert ts["timestamps"][0] == "2024-06-15T12:00:00"
        assert ts["timestamps"][1] == "2024-06-15T12:30:00"
        assert ts["timestamps"][2] == "2024-06-15T13:00:00"

    def test_compatible_with_met_config(self):
        """Output dict should be valid input to MetConfig."""
        from bldfm.config_parser import MetConfig

        ts = generate_synthetic_timeseries(n_timesteps=10, seed=42)
        met = MetConfig(**ts)
        met.validate()
        assert met.n_timesteps == 10


# --- generate_towers_grid ---

class TestTowersGrid:

    def test_tower_output_structure(self):
        """Test tower count, dict keys, and name uniqueness."""
        towers = generate_towers_grid(n_towers=4, layout="grid")
        assert len(towers) == 4
        for t in towers:
            assert "name" in t
            assert "lat" in t
            assert "lon" in t
            assert "z_m" in t
        names = [t["name"] for t in towers]
        assert len(set(names)) == len(names)

    def test_invalid_layout_raises(self):
        with pytest.raises(ValueError, match="Unknown layout"):
            generate_towers_grid(layout="hexagonal")

    def test_compatible_with_tower_config(self):
        """Output dicts should be valid input to TowerConfig."""
        from bldfm.config_parser import TowerConfig

        towers = generate_towers_grid(n_towers=2, seed=42)
        for t in towers:
            tc = TowerConfig(**t)
            assert tc.z_m > 0

    def test_reproducibility(self):
        t1 = generate_towers_grid(n_towers=3, layout="random", seed=99)
        t2 = generate_towers_grid(n_towers=3, layout="random", seed=99)
        assert t1 == t2
