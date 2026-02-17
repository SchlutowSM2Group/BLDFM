"""Tests for config_parser module."""

import pytest

pytestmark = pytest.mark.unit
import tempfile
from pathlib import Path

from bldfm.config_parser import (
    BLDFMConfig,
    TowerConfig,
    DomainConfig,
    MetConfig,
    SolverConfig,
    OutputConfig,
    ParallelConfig,
    load_config,
    parse_config_dict,
    latlon_to_xy,
)


# --- latlon_to_xy ---


@pytest.mark.parametrize(
    "lat, lon, ref_lat, ref_lon, exp_x, exp_y, tol",
    [
        (50.0, 11.0, 50.0, 11.0, 0.0, 0.0, 0.01),  # same point
        (51.0, 11.0, 50.0, 11.0, 0.0, 111_194.9, 500),  # 1 deg north ~ 111 km
        (50.0, 12.0, 50.0, 11.0, 71_696.0, 0.0, 500),  # 1 deg east at 50N ~ 71.7 km
    ],
)
def test_latlon_to_xy(lat, lon, ref_lat, ref_lon, exp_x, exp_y, tol):
    x, y = latlon_to_xy(lat, lon, ref_lat, ref_lon)
    assert abs(x - exp_x) < tol
    assert abs(y - exp_y) < tol


# --- MetConfig validation ---


def test_met_config_scalar():
    met = MetConfig(ustar=0.4, mol=-100.0, wind_speed=5.0, wind_dir=270.0)
    met.validate()
    assert met.n_timesteps == 1


def test_met_config_timeseries():
    met = MetConfig(
        ustar=[0.3, 0.4, 0.5],
        mol=[-100.0, -200.0, 500.0],
        wind_speed=[3.0, 5.0, 6.0],
        wind_dir=[270.0, 180.0, 90.0],
    )
    met.validate()
    assert met.n_timesteps == 3


def test_met_config_mixed_lengths_raises():
    met = MetConfig(
        ustar=[0.3, 0.4],
        mol=[-100.0, -200.0, 500.0],
    )
    with pytest.raises(ValueError, match="same length"):
        met.validate()


def test_met_config_get_step_scalar():
    met = MetConfig(ustar=0.4, mol=-100.0, wind_speed=5.0, wind_dir=270.0)
    step = met.get_step(0)
    assert step["ustar"] == 0.4
    assert step["wind_dir"] == 270.0


def test_met_config_get_step_timeseries():
    met = MetConfig(
        ustar=[0.3, 0.4],
        mol=[-100.0, -200.0],
        wind_speed=5.0,  # scalar (same for all steps)
        wind_dir=[270.0, 180.0],
    )
    step = met.get_step(1)
    assert step["ustar"] == 0.4
    assert step["mol"] == -200.0
    assert step["wind_speed"] == 5.0
    assert step["wind_dir"] == 180.0


# --- TowerConfig ---


def test_tower_compute_local_xy():
    tower = TowerConfig(name="A", lat=50.9505, lon=11.5865, z_m=10.0)
    tower.compute_local_xy(50.9500, 11.5860)
    assert abs(tower.x) < 100  # small offset
    assert abs(tower.y) < 100


# --- parse_config_dict ---


def test_parse_minimal_config():
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
    }
    config = parse_config_dict(raw)
    assert isinstance(config, BLDFMConfig)
    assert config.domain.nx == 256
    assert len(config.towers) == 1
    assert config.towers[0].name == "A"
    assert config.solver.closure == "MOST"  # default
    assert config.output.format == "netcdf"  # default


def test_parse_config_missing_domain():
    with pytest.raises(ValueError, match="domain"):
        parse_config_dict({"towers": [], "met": {"ustar": 0.4}})


def test_parse_config_missing_towers():
    raw = {
        "domain": {"nx": 256, "ny": 128, "xmax": 1000.0, "ymax": 500.0, "nz": 32},
        "met": {"ustar": 0.4},
    }
    with pytest.raises(ValueError, match="towers"):
        parse_config_dict(raw)


# --- load_config (YAML file) ---


def test_load_config_from_yaml():
    yaml_content = """\
domain:
  nx: 256
  ny: 128
  xmax: 1000.0
  ymax: 500.0
  nz: 32
  ref_lat: 50.95
  ref_lon: 11.586
towers:
  - name: A
    lat: 50.9505
    lon: 11.5865
    z_m: 10.0
met:
  ustar: 0.4
  mol: -100.0
  wind_speed: 5.0
  wind_dir: 270.0
solver:
  closure: MOST
  footprint: true
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)

    assert config.domain.nz == 32
    assert config.towers[0].lat == 50.9505
    assert config.solver.footprint is True
    # Tower local coords should be computed
    assert config.towers[0].x != 0.0 or config.towers[0].y != 0.0

    Path(f.name).unlink()


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


# --- New config fields (v1.0 expansion) ---


def test_domain_output_levels():
    """Test that output_levels in domain config round-trips correctly."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
            "output_levels": [0, 2, 4, 8],
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
    }
    config = parse_config_dict(raw)
    assert config.domain.output_levels == [0, 2, 4, 8]


def test_domain_output_levels_default():
    """Test that output_levels defaults to None when not specified."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
    }
    config = parse_config_dict(raw)
    assert config.domain.output_levels is None


def test_met_z0():
    """Test that z0 in met config appears in get_step() result."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4, "z0": 0.1},
    }
    config = parse_config_dict(raw)
    step = config.met.get_step(0)
    assert "z0" in step
    assert step["z0"] == 0.1


def test_met_z0_default():
    """Test that z0 is not in get_step() when not set."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
    }
    config = parse_config_dict(raw)
    step = config.met.get_step(0)
    assert "z0" not in step


def test_solver_analytic():
    """Test that analytic field in solver config round-trips correctly."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
        "solver": {"analytic": True},
    }
    config = parse_config_dict(raw)
    assert config.solver.analytic is True


def test_solver_src_loc():
    """Test that src_loc in solver config is converted to tuple."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
        "solver": {"src_loc": [200.0, 50.0]},
    }
    config = parse_config_dict(raw)
    assert config.solver.src_loc == (200.0, 50.0)
    assert isinstance(config.solver.src_loc, tuple)


def test_solver_defaults_new_fields():
    """Test that new solver fields default correctly when not specified."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
    }
    config = parse_config_dict(raw)
    assert config.solver.analytic is False
    assert config.solver.src_loc is None


def test_domain_full_output():
    """Test that full_output in domain config round-trips correctly."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
            "full_output": True,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
    }
    config = parse_config_dict(raw)
    assert config.domain.full_output is True


def test_domain_full_output_default():
    """Test that full_output defaults to False when not specified."""
    raw = {
        "domain": {
            "nx": 256,
            "ny": 128,
            "xmax": 1000.0,
            "ymax": 500.0,
            "nz": 32,
            "ref_lat": 50.95,
            "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4},
    }
    config = parse_config_dict(raw)
    assert config.domain.full_output is False
