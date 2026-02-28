"""
NetCDF I/O for BLDFM footprint results.

Saves and loads multi-tower, multi-timestep footprint fields as CF-1.8
compliant xarray Datasets with zlib compression.
"""

import numpy as np
import xarray as xr
from pathlib import Path

from .utils import get_logger

logger = get_logger("io")


def save_footprints_to_netcdf(results, config, filepath):
    """Save multitower results to a CF-compliant NetCDF file.

    Parameters
    ----------
    results : dict
        Output of run_bldfm_multitower: {tower_name: [result_dict, ...]}.
    config : BLDFMConfig
        Configuration used for the run.
    filepath : str or Path
        Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    tower_names = list(results.keys())
    n_towers = len(tower_names)

    # Get dimensions from first result
    first_result = results[tower_names[0]][0]
    X, Y, Z_coord = first_result["grid"]
    is_3d = first_result["flx"].ndim == 3
    n_time = len(results[tower_names[0]])

    if is_3d:
        nz_out, ny, nx = first_result["flx"].shape
    else:
        ny, nx = first_result["flx"].shape

    # Extract coordinate arrays
    if is_3d:
        x = X[0, 0, :]
        y = Y[0, :, 0]
        z = Z_coord[:, 0, 0]
    else:
        x = X[0, :] if X.ndim == 2 else X
        y = Y[:, 0] if Y.ndim == 2 else Y

    # Build timestamps
    timestamps = []
    for r in results[tower_names[0]]:
        ts = r["timestamp"]
        timestamps.append(str(ts))

    # Collect data arrays
    if is_3d:
        flx_data = np.zeros((n_time, n_towers, nz_out, ny, nx))
        conc_data = np.zeros((n_time, n_towers, nz_out, ny, nx))
        dims = ["time", "tower", "z", "y", "x"]
    else:
        flx_data = np.zeros((n_time, n_towers, ny, nx))
        conc_data = np.zeros((n_time, n_towers, ny, nx))
        dims = ["time", "tower", "y", "x"]

    ustar_data = np.zeros((n_time,))
    mol_data = np.zeros((n_time,))
    wind_speed_data = np.zeros((n_time,))
    wind_dir_data = np.zeros((n_time,))

    for ti, tower_name in enumerate(tower_names):
        for t, r in enumerate(results[tower_name]):
            flx_data[t, ti] = r["flx"]
            conc_data[t, ti] = r["conc"]
            if ti == 0:  # met params are the same for all towers
                ustar_data[t] = r["params"]["ustar"]
                mol_data[t] = r["params"]["mol"]
                wind_speed_data[t] = r["params"]["wind_speed"]
                wind_dir_data[t] = r["params"]["wind_dir"]

    # Tower metadata
    tower_lats = [t.lat for t in config.towers]
    tower_lons = [t.lon for t in config.towers]
    tower_z = [t.z_m for t in config.towers]

    coords = {
        "x": ("x", x, {"long_name": "easting", "units": "m"}),
        "y": ("y", y, {"long_name": "northing", "units": "m"}),
        "time": ("time", timestamps),
        "tower": ("tower", tower_names),
    }
    if is_3d:
        coords["z"] = ("z", z, {"long_name": "height", "units": "m"})

    ds = xr.Dataset(
        {
            "footprint": (
                dims,
                flx_data,
                {
                    "long_name": "flux footprint",
                    "units": "m^-2",
                },
            ),
            "concentration": (
                dims,
                conc_data,
                {
                    "long_name": "concentration field",
                    "units": "scalar_unit",
                },
            ),
            "ustar": (
                ["time"],
                ustar_data,
                {"long_name": "friction velocity", "units": "m s^-1"},
            ),
            "mol": (
                ["time"],
                mol_data,
                {"long_name": "Monin-Obukhov length", "units": "m"},
            ),
            "wind_speed": (
                ["time"],
                wind_speed_data,
                {"long_name": "wind speed", "units": "m s^-1"},
            ),
            "wind_dir": (
                ["time"],
                wind_dir_data,
                {"long_name": "wind direction", "units": "degrees"},
            ),
            "tower_lat": (
                ["tower"],
                tower_lats,
                {"long_name": "tower latitude", "units": "degrees_north"},
            ),
            "tower_lon": (
                ["tower"],
                tower_lons,
                {"long_name": "tower longitude", "units": "degrees_east"},
            ),
            "tower_z": (
                ["tower"],
                tower_z,
                {"long_name": "measurement height", "units": "m"},
            ),
        },
        coords=coords,
        attrs={
            "Conventions": "CF-1.8",
            "title": "BLDFM footprint output",
            "source": "BLDFM v1.0",
            "closure": config.solver.closure,
            "domain_xmax": config.domain.xmax,
            "domain_ymax": config.domain.ymax,
        },
    )

    encoding = {
        "footprint": {"zlib": True, "complevel": 4},
        "concentration": {"zlib": True, "complevel": 4},
    }

    ds.to_netcdf(filepath, encoding=encoding)
    logger.info("Saved footprints to %s", filepath)


def load_footprints_from_netcdf(filepath):
    """Load footprint results from a NetCDF file.

    Parameters
    ----------
    filepath : str or Path
        Path to the NetCDF file.

    Returns
    -------
    xr.Dataset
        Dataset with footprint, concentration, and metadata.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    ds = xr.open_dataset(filepath)
    logger.info(
        "Loaded footprints: %d times x %d towers",
        ds.sizes["time"],
        ds.sizes["tower"],
    )
    return ds
