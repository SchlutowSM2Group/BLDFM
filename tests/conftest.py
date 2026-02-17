import os
import pytest

from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_single, run_bldfm_timeseries, run_bldfm_multitower
from bldfm.synthetic import generate_synthetic_timeseries, generate_towers_grid


@pytest.fixture(scope="session", autouse=True)
def ensure_plots_dir():
    os.makedirs("plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Session-scoped fixtures to avoid redundant solver runs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def simple_config_session():
    """Minimal single-tower, single-timestep config."""
    return parse_config_dict({
        "domain": {
            "nx": 128, "ny": 64, "xmax": 500.0, "ymax": 250.0, "nz": 16,
            "modes": [128, 64],
            "ref_lat": 50.95, "ref_lon": 11.586,
        },
        "towers": [
            {"name": "test_tower", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4, "mol": -100.0, "wind_speed": 5.0, "wind_dir": 270.0},
        "solver": {"closure": "MOST", "footprint": True},
    })


@pytest.fixture(scope="session")
def single_run_result(simple_config_session):
    """Run solver once for all single-run structure tests."""
    return run_bldfm_single(simple_config_session, simple_config_session.towers[0])


@pytest.fixture(scope="session")
def timeseries_config_session():
    """Config with 3-step timeseries met data and 2 towers."""
    towers = generate_towers_grid(n_towers=2, z_m=10.0, layout="transect", seed=42)
    met = generate_synthetic_timeseries(n_timesteps=3, seed=42)
    return parse_config_dict({
        "domain": {
            "nx": 64, "ny": 64, "xmax": 200.0, "ymax": 200.0, "nz": 8,
            "modes": [64, 64],
            "ref_lat": towers[0]["lat"], "ref_lon": towers[0]["lon"],
        },
        "towers": towers,
        "met": met,
        "solver": {"closure": "MOST", "footprint": True},
    })


@pytest.fixture(scope="session")
def timeseries_results_session(timeseries_config_session):
    """Run timeseries once for first tower."""
    return run_bldfm_timeseries(
        timeseries_config_session, timeseries_config_session.towers[0]
    )


@pytest.fixture(scope="session")
def multitower_results_session(timeseries_config_session):
    """Run multitower once, shared by interface and I/O tests."""
    return run_bldfm_multitower(timeseries_config_session), timeseries_config_session


@pytest.fixture(scope="session")
def footprint_result_session():
    """Run a single footprint solve for plotting tests."""
    config = parse_config_dict({
        "domain": {
            "nx": 64, "ny": 64, "xmax": 200.0, "ymax": 200.0, "nz": 8,
            "modes": [64, 64],
            "ref_lat": 50.95, "ref_lon": 11.586,
        },
        "towers": [
            {"name": "A", "lat": 50.9505, "lon": 11.5865, "z_m": 10.0},
        ],
        "met": {"ustar": 0.4, "mol": -100.0, "wind_speed": 5.0, "wind_dir": 270.0},
        "solver": {"closure": "MOST", "footprint": True},
    })
    return run_bldfm_single(config, config.towers[0]), config
