"""
Synthetic data generators for testing and demonstration.

Provides functions to generate realistic-looking meteorological timeseries
and tower grid configurations without requiring real observational data.
"""

import numpy as np
from typing import List, Optional, Tuple


def generate_synthetic_timeseries(
    n_timesteps: int = 48,
    dt_minutes: int = 30,
    start_time: str = "2024-01-01T00:00",
    ustar_range: Tuple[float, float] = (0.1, 0.8),
    mol_range: Tuple[float, float] = (-500.0, 500.0),
    wind_speed_range: Tuple[float, float] = (1.0, 8.0),
    wind_dir_mean: float = 270.0,
    wind_dir_std: float = 30.0,
    seed: Optional[int] = None,
) -> dict:
    """Generate a synthetic meteorological timeseries with diurnal cycle.

    Produces ustar, Monin-Obukhov length, wind speed, and wind direction
    arrays that follow a realistic diurnal pattern (unstable daytime,
    stable nighttime) with added noise.

    Parameters
    ----------
    n_timesteps : int
        Number of time steps to generate.
    dt_minutes : int
        Time step interval in minutes.
    start_time : str
        ISO-format start time string.
    ustar_range : tuple of float
        (min, max) friction velocity [m/s].
    mol_range : tuple of float
        (min_negative, max_positive) Monin-Obukhov length [m].
        Negative = unstable, positive = stable.
    wind_speed_range : tuple of float
        (min, max) wind speed [m/s].
    wind_dir_mean : float
        Mean wind direction [degrees].
    wind_dir_std : float
        Standard deviation of wind direction [degrees].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys matching MetConfig schema:
        ustar, mol, wind_speed, wind_dir, timestamps (all lists).
    """
    rng = np.random.default_rng(seed)

    # Time array (hours from start)
    t_hours = np.arange(n_timesteps) * dt_minutes / 60.0

    # Diurnal phase: 0 at midnight, pi at noon
    phase = 2.0 * np.pi * (t_hours % 24.0) / 24.0

    # ustar: peaks during daytime (convective mixing)
    ustar_min, ustar_max = ustar_range
    ustar_mean = 0.5 * (ustar_min + ustar_max)
    ustar_amp = 0.5 * (ustar_max - ustar_min)
    ustar = ustar_mean + ustar_amp * np.sin(phase - np.pi / 2)
    ustar += rng.normal(0, 0.05 * ustar_amp, n_timesteps)
    ustar = np.clip(ustar, ustar_min, ustar_max)

    # Monin-Obukhov length: negative (unstable) during day, positive (stable) at night
    mol_neg, mol_pos = mol_range
    # Daytime: unstable (negative L, small magnitude = strong instability at noon)
    # Nighttime: stable (positive L, small magnitude = strong stability at midnight)
    mol_magnitude = 50.0 + 450.0 * np.abs(np.cos(phase - np.pi / 2))
    mol_sign = np.where(np.sin(phase - np.pi / 2) > 0, -1.0, 1.0)
    mol = mol_sign * mol_magnitude
    mol += rng.normal(0, 20.0, n_timesteps)
    mol = np.clip(mol, mol_neg, mol_pos)

    # Wind speed: slightly higher during daytime
    ws_min, ws_max = wind_speed_range
    ws_mean = 0.5 * (ws_min + ws_max)
    ws_amp = 0.3 * (ws_max - ws_min)
    wind_speed = ws_mean + ws_amp * np.sin(phase - np.pi / 2)
    wind_speed += rng.normal(0, 0.5, n_timesteps)
    wind_speed = np.clip(wind_speed, ws_min, ws_max)

    # Wind direction: random walk around mean
    wind_dir = wind_dir_mean + rng.normal(0, wind_dir_std, n_timesteps)
    wind_dir = wind_dir % 360.0

    # Generate timestamps
    timestamps = []
    from datetime import datetime, timedelta

    t0 = datetime.fromisoformat(start_time)
    for i in range(n_timesteps):
        timestamps.append((t0 + timedelta(minutes=i * dt_minutes)).isoformat())

    return {
        "ustar": ustar.tolist(),
        "mol": mol.tolist(),
        "wind_speed": wind_speed.tolist(),
        "wind_dir": wind_dir.tolist(),
        "timestamps": timestamps,
    }


def generate_towers_grid(
    n_towers: int = 4,
    center_lat: float = 50.9500,
    center_lon: float = 11.5860,
    spacing_m: float = 500.0,
    z_m: float = 10.0,
    layout: str = "grid",
    seed: Optional[int] = None,
) -> List[dict]:
    """Generate tower configurations in various spatial layouts.

    Parameters
    ----------
    n_towers : int
        Number of towers to generate.
    center_lat, center_lon : float
        Center point of the tower array [decimal degrees].
    spacing_m : float
        Approximate spacing between towers [meters].
    z_m : float
        Measurement height [meters] (same for all towers).
    layout : str
        Spatial layout: "grid", "random", or "transect".
    seed : int, optional
        Random seed for reproducibility (used by "random" layout).

    Returns
    -------
    list of dict
        Tower configurations matching TowerConfig schema.
    """
    rng = np.random.default_rng(seed)

    # Approximate degrees per meter at center latitude
    deg_per_m_lat = 1.0 / 111_320.0
    deg_per_m_lon = 1.0 / (111_320.0 * np.cos(np.radians(center_lat)))

    if layout == "grid":
        side = int(np.ceil(np.sqrt(n_towers)))
        offsets = []
        for i in range(side):
            for j in range(side):
                if len(offsets) >= n_towers:
                    break
                dx = (j - (side - 1) / 2) * spacing_m
                dy = (i - (side - 1) / 2) * spacing_m
                offsets.append((dx, dy))
    elif layout == "transect":
        offsets = [
            ((i - (n_towers - 1) / 2) * spacing_m, 0.0) for i in range(n_towers)
        ]
    elif layout == "random":
        extent = spacing_m * np.sqrt(n_towers)
        offsets = [
            (rng.uniform(-extent / 2, extent / 2), rng.uniform(-extent / 2, extent / 2))
            for _ in range(n_towers)
        ]
    else:
        raise ValueError(f"Unknown layout: {layout}. Use 'grid', 'transect', or 'random'.")

    towers = []
    for i, (dx, dy) in enumerate(offsets):
        lat = center_lat + dy * deg_per_m_lat
        lon = center_lon + dx * deg_per_m_lon
        towers.append({
            "name": f"tower_{chr(65 + i)}" if i < 26 else f"tower_{i}",
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "z_m": z_m,
        })

    return towers
