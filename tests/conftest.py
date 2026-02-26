import os

import numpy as np
import pytest

from bldfm.config_parser import parse_config_dict
from bldfm.interface import run_bldfm_single, run_bldfm_timeseries, run_bldfm_multitower
from bldfm.pbl_model import vertical_profiles
from bldfm.solver import steady_state_transport_solver
from bldfm.utils import ideal_source


@pytest.fixture(scope="session", autouse=True)
def ensure_plots_dir():
    os.makedirs("plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Session-scoped fixtures to avoid redundant solver runs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def simple_config_session():
    """Minimal single-tower, single-timestep config on an elongated domain.

    Domain is elongated in y (100x700 m) to match the northerly wind
    (wind_dir=0), giving the footprint room to develop upwind.
    Tower placed at ~(50, 0) in grid coords to match source_area_result_session.
    """
    return parse_config_dict(
        {
            "domain": {
                "nx": 64,
                "ny": 128,
                "xmax": 100.0,
                "ymax": 700.0,
                "nz": 16,
                "modes": [64, 128],
                "ref_lat": 50.95,
                "ref_lon": 11.586,
            },
            "towers": [
                {"name": "test_tower", "lat": 50.95, "lon": 11.5867, "z_m": 10.0},
            ],
            "met": {"ustar": 0.5, "mol": -100.0, "wind_speed": 6.0, "wind_dir": 0.0},
            "solver": {"closure": "MOST", "footprint": True},
        }
    )


@pytest.fixture(scope="session")
def single_run_result(simple_config_session):
    """Run solver once for all single-run structure tests."""
    return run_bldfm_single(simple_config_session, simple_config_session.towers[0])


@pytest.fixture(scope="session")
def timeseries_config_session():
    """Config with 3-step timeseries met data and 1 tower (primary use case).

    Met conditions are explicit variants of simple_config_session
    (ustar=0.5, mol=-100, wind_speed=6, wind_dir=0) so that each
    timestep produces a recognisable footprint plume.
    """
    return parse_config_dict(
        {
            "domain": {
                "nx": 64,
                "ny": 128,
                "xmax": 100.0,
                "ymax": 700.0,
                "nz": 16,
                "modes": [64, 128],
                "ref_lat": 50.95,
                "ref_lon": 11.586,
            },
            "towers": [
                {"name": "tower_A", "lat": 50.95, "lon": 11.5867, "z_m": 10.0},
            ],
            "met": {
                "ustar": [0.5, 0.4, 0.6],
                "mol": [-100.0, -200.0, -50.0],
                "wind_speed": [6.0, 5.0, 7.0],
                "wind_dir": [0.0, 15.0, 345.0],
            },
            "solver": {"closure": "MOST", "footprint": True},
        }
    )


@pytest.fixture(scope="session")
def multitower_config_session():
    """Config with 3-step timeseries met data and 2 towers.

    Same domain and met as timeseries_config_session but with a second tower.
    """
    return parse_config_dict(
        {
            "domain": {
                "nx": 64,
                "ny": 128,
                "xmax": 100.0,
                "ymax": 700.0,
                "nz": 16,
                "modes": [64, 128],
                "ref_lat": 50.95,
                "ref_lon": 11.586,
            },
            "towers": [
                {"name": "tower_A", "lat": 50.95, "lon": 11.5867, "z_m": 10.0},
                {"name": "tower_B", "lat": 50.9504, "lon": 11.5867, "z_m": 10.0},
            ],
            "met": {
                "ustar": [0.5, 0.4, 0.6],
                "mol": [-100.0, -200.0, -50.0],
                "wind_speed": [6.0, 5.0, 7.0],
                "wind_dir": [0.0, 15.0, 345.0],
            },
            "solver": {"closure": "MOST", "footprint": True},
        }
    )


@pytest.fixture(scope="session")
def timeseries_results_session(timeseries_config_session):
    """Run timeseries once for the single tower."""
    return run_bldfm_timeseries(
        timeseries_config_session, timeseries_config_session.towers[0]
    )


@pytest.fixture(scope="session")
def multitower_results_session(multitower_config_session):
    """Run multitower once, shared by interface and I/O tests."""
    return run_bldfm_multitower(multitower_config_session), multitower_config_session


@pytest.fixture(scope="session")
def footprint_result_session(source_area_result_session):
    """Repackage source_area_result_session for plotting tests that need (result, config).

    Shares the low-level solver result (no duplicate run) and provides a
    config object for tests that need geo-referencing (e.g. map plots).
    """
    r = source_area_result_session
    result = {
        "grid": r["grid"],
        "flx": r["flx"],
        "conc": r["conc"],
        "tower_name": "A",
        "tower_xy": r["meas_pt"],
        "timestamp": None,
        "params": {"ustar": 0.5, "mol": -100.0, "wind_speed": 6.0, "wind_dir": 0.0},
    }
    config = parse_config_dict(
        {
            "domain": {
                "nx": 64,
                "ny": 128,
                "xmax": 100.0,
                "ymax": 700.0,
                "nz": 16,
                "modes": [64, 128],
                "ref_lat": 50.95,
                "ref_lon": 11.586,
            },
            "towers": [
                {"name": "A", "lat": 50.95, "lon": 11.5867, "z_m": 10.0},
            ],
            "met": {"ustar": 0.5, "mol": -100.0, "wind_speed": 6.0, "wind_dir": 0.0},
            "solver": {"closure": "MOST", "footprint": True},
        }
    )
    return result, config


@pytest.fixture(scope="session")
def source_area_result_session():
    """Low-level footprint solve on an elongated domain for source area tests.

    Matches the configuration from runs/low_level/source_area_example.py
    but at reduced resolution (64x128 instead of 512x256).  The grid is
    elongated in y to match the 100x700m domain.
    """
    nx, ny, nz = 64, 128, 16
    domain = (100.0, 700.0)
    modes = (64, 128)
    meas_pt = (50.0, 0.0)
    meas_height = 10.0
    wind = (0.0, -6.0)
    ustar = 0.5

    srf_flx = np.zeros([ny, nx])
    z, profs = vertical_profiles(nz, meas_height, wind, ustar)
    grid, conc, flx = steady_state_transport_solver(
        srf_flx,
        z,
        profs,
        domain,
        nz,
        modes=modes,
        meas_pt=meas_pt,
        footprint=True,
    )
    return {
        "grid": grid,
        "conc": conc,
        "flx": flx,
        "meas_pt": meas_pt,
        "wind": wind,
    }


@pytest.fixture(scope="session")
def plume_3d_result_session():
    """Low-level 3D concentration solve (dispersion mode).

    Reduced-resolution version of examples/low_level/minimal_example_3d.py.
    Output on every 2nd vertical level for manageable size.
    """
    nx, ny, nz = 64, 32, 16
    domain = (800.0, 100.0)
    modes = (64, 32)
    meas_pt = (400.0, 50.0)
    meas_height = 10.0
    wind = (6.0, 0.0)
    ustar = 0.4

    srf_flx = ideal_source((nx, ny), domain)
    z, profs = vertical_profiles(nz, meas_height, wind, ustar)
    levels = np.arange(0, nz + 1, 2)  # [0, 2, 4, ..., 16]

    grid, conc, flx = steady_state_transport_solver(
        srf_flx,
        z,
        profs,
        domain,
        levels,
        modes=modes,
        meas_pt=meas_pt,
        footprint=False,
    )
    return {"grid": grid, "conc": conc, "flx": flx, "levels": levels}
