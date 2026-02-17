"""Tests for NetCDF I/O."""

import tempfile
import numpy as np
import pytest

pytestmark = pytest.mark.integration

from bldfm.io import save_footprints_to_netcdf, load_footprints_from_netcdf


def test_netcdf_roundtrip_and_values(multitower_results_session):
    """Test save/load roundtrip and verify values match original."""
    results, config = multitower_results_session
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = f"{tmpdir}/test_output.nc"
        save_footprints_to_netcdf(results, config, filepath)
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
