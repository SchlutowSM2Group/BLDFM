"""Tests for NetCDF I/O."""

import tempfile
import numpy as np
import pytest

pytestmark = pytest.mark.integration

from bldfm.io import save_footprints_to_netcdf, load_footprints_from_netcdf


def test_netcdf_roundtrip_and_values(multitower_results_session):
    """Test save/load roundtrip and verify values match original."""
    results, config = multitower_results_session
    filepath = "output/test_multitower_timeseries.nc"
    save_footprints_to_netcdf(results, config, filepath)
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = load_footprints_from_netcdf(filepath)

        # Structure
        assert "footprint" in ds
        assert "concentration" in ds
        assert ds.sizes["time"] == 3
        assert ds.sizes["tower"] == 2
        assert ds.sizes["x"] == 64
        assert ds.sizes["y"] == 64

        # Values match
        tower_names = list(results.keys())
        for ti, name in enumerate(tower_names):
            for t in range(3):
                original_flx = results[name][t]["flx"]
                loaded_flx = ds["footprint"].values[t, ti]
                np.testing.assert_allclose(loaded_flx, original_flx, rtol=1e-6)
        ds.close()


def test_netcdf_metadata(multitower_results_session):
    """Test CF attributes, met variables, and tower metadata."""
    results, config = multitower_results_session
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_output.nc"
        save_footprints_to_netcdf(results, config, filepath)
        ds = load_footprints_from_netcdf(filepath)

        # CF attributes
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert ds.attrs["closure"] == "MOST"
        assert "long_name" in ds["footprint"].attrs
        assert "units" in ds["footprint"].attrs

        # Met variables
        for var in ("ustar", "mol", "wind_speed", "wind_dir"):
            assert var in ds
        assert len(ds["ustar"]) == 3

        # Tower metadata
        for var in ("tower_lat", "tower_lon", "tower_z"):
            assert var in ds
        assert len(ds["tower_lat"]) == 2
        ds.close()


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_footprints_from_netcdf("/nonexistent/path.nc")


def test_netcdf_met_values_match(multitower_results_session):
    """Test that saved met values match the original result params exactly."""
    results, config = multitower_results_session
    first_tower = list(results.keys())[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_met.nc"
        save_footprints_to_netcdf(results, config, filepath)
        ds = load_footprints_from_netcdf(filepath)

        for t in range(3):
            original_params = results[first_tower][t]["params"]
            for var in ("ustar", "mol", "wind_speed", "wind_dir"):
                np.testing.assert_allclose(
                    ds[var].values[t],
                    original_params[var],
                    rtol=1e-6,
                    err_msg=f"Mismatch for {var} at timestep {t}",
                )
        ds.close()


def test_netcdf_tower_metadata_match(multitower_results_session):
    """Test that tower lat/lon/z and names match config exactly."""
    results, config = multitower_results_session

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_towers.nc"
        save_footprints_to_netcdf(results, config, filepath)
        ds = load_footprints_from_netcdf(filepath)

        # Tower names match
        assert list(ds["tower"].values) == [t.name for t in config.towers]

        # Coordinates match
        for i, tower in enumerate(config.towers):
            np.testing.assert_allclose(ds["tower_lat"].values[i], tower.lat)
            np.testing.assert_allclose(ds["tower_lon"].values[i], tower.lon)
            np.testing.assert_allclose(ds["tower_z"].values[i], tower.z_m)
        ds.close()


def test_netcdf_multitower_select_by_name(multitower_results_session):
    """Test that individual tower data can be selected by name and matches."""
    results, config = multitower_results_session

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_select.nc"
        save_footprints_to_netcdf(results, config, filepath)
        ds = load_footprints_from_netcdf(filepath)

        for tower_name in results:
            tower_ds = ds.sel(tower=tower_name)

            # Correct dimensions (time, y, x) — tower dim squeezed out
            assert tower_ds["footprint"].dims == ("time", "y", "x")
            assert tower_ds.sizes["time"] == 3

            # Values match original
            for t in range(3):
                np.testing.assert_allclose(
                    tower_ds["footprint"].values[t],
                    results[tower_name][t]["flx"],
                    rtol=1e-6,
                )
        ds.close()


def test_netcdf_timeseries_timestamps(multitower_results_session):
    """Test that timestamps are stored correctly and data varies across time."""
    results, config = multitower_results_session
    first_tower = list(results.keys())[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_time.nc"
        save_footprints_to_netcdf(results, config, filepath)
        ds = load_footprints_from_netcdf(filepath)

        # Timestamps match original results
        expected_ts = [str(r["timestamp"]) for r in results[first_tower]]
        loaded_ts = list(ds["time"].values)
        assert loaded_ts == expected_ts

        # Met params vary across time (not all identical)
        for var in ("ustar", "mol", "wind_speed", "wind_dir"):
            values = ds[var].values
            assert len(set(values)) > 1, f"{var} is constant across timesteps"
        ds.close()


def test_netcdf_single_tower_timeseries(
    timeseries_results_session, timeseries_config_session
):
    """Test save/load roundtrip for a single tower timeseries."""
    from dataclasses import replace

    results = timeseries_results_session
    config = timeseries_config_session

    # Wrap single-tower timeseries into multitower format
    tower_name = results[0]["tower_name"]
    multitower = {tower_name: results}

    # Config must match results: keep only the tower present in results
    matching_tower = [t for t in config.towers if t.name == tower_name]
    single_tower_config = replace(config, towers=matching_tower)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_single_tower.nc"
        save_footprints_to_netcdf(multitower, single_tower_config, filepath)
        ds = load_footprints_from_netcdf(filepath)

        assert ds.sizes["tower"] == 1
        assert ds.sizes["time"] == 3
        assert list(ds["tower"].values) == [tower_name]

        # Values roundtrip
        for t in range(3):
            np.testing.assert_allclose(
                ds["footprint"].values[t, 0],
                results[t]["flx"],
                rtol=1e-6,
            )
        ds.close()


def test_netcdf_global_attrs(multitower_results_session):
    """Test that global attributes capture domain and solver config."""
    results, config = multitower_results_session

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_attrs.nc"
        save_footprints_to_netcdf(results, config, filepath)
        ds = load_footprints_from_netcdf(filepath)

        assert ds.attrs["title"] == "BLDFM footprint output"
        assert ds.attrs["source"] == "BLDFM v1.0"
        assert ds.attrs["domain_xmax"] == config.domain.xmax
        assert ds.attrs["domain_ymax"] == config.domain.ymax

        # Coordinate arrays span the domain
        assert ds["x"].values[0] < ds["x"].values[-1]
        assert ds["y"].values[0] < ds["y"].values[-1]
        assert "units" in ds["x"].attrs
        assert "units" in ds["y"].attrs
        ds.close()


def test_netcdf_3d_roundtrip():
    """Test save/load roundtrip with 3D output fields (z dimension)."""
    from bldfm.config_parser import parse_config_dict

    nz_out, ny, nx = 4, 16, 32
    n_time, n_towers = 2, 1

    # Build mock 3D results
    rng = np.random.default_rng(42)
    Z, Y, X = np.meshgrid(
        np.arange(nz_out, dtype=float),
        np.linspace(0, 100, ny),
        np.linspace(0, 200, nx),
        indexing="ij",
    )
    grid = (X, Y, Z)

    results = {
        "tower_A": [
            {
                "grid": grid,
                "flx": rng.random((nz_out, ny, nx)),
                "conc": rng.random((nz_out, ny, nx)),
                "tower_name": "tower_A",
                "tower_xy": (0.0, 0.0),
                "timestamp": f"t{t}",
                "params": {
                    "ustar": 0.4,
                    "mol": -100.0,
                    "wind_speed": 5.0,
                    "wind_dir": 270.0,
                },
            }
            for t in range(n_time)
        ]
    }

    config = parse_config_dict(
        {
            "domain": {
                "nx": nx,
                "ny": ny,
                "xmax": 200.0,
                "ymax": 100.0,
                "nz": 8,
                "ref_lat": 0.0,
                "ref_lon": 0.0,
            },
            "towers": [{"name": "tower_A", "lat": 0.0, "lon": 0.0, "z_m": 10.0}],
            "met": {"ustar": 0.4},
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_3d.nc"
        save_footprints_to_netcdf(results, config, filepath)
        ds = load_footprints_from_netcdf(filepath)

        # z dimension exists
        assert "z" in ds.sizes
        assert ds.sizes["z"] == nz_out
        assert ds.sizes["time"] == n_time
        assert ds.sizes["tower"] == n_towers

        # Values roundtrip
        for t in range(n_time):
            np.testing.assert_allclose(
                ds["footprint"].values[t, 0],
                results["tower_A"][t]["flx"],
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                ds["concentration"].values[t, 0],
                results["tower_A"][t]["conc"],
                rtol=1e-6,
            )

        # z coordinate values are correct
        np.testing.assert_allclose(ds["z"].values, Z[:, 0, 0], rtol=1e-6)
        ds.close()
