"""
YAML configuration parser and dataclass schema for BLDFM.

Defines the configuration hierarchy:
    BLDFMConfig
    ├── DomainConfig
    ├── List[TowerConfig]
    ├── MetConfig
    ├── SolverConfig
    ├── OutputConfig
    └── ParallelConfig
"""

import math
import yaml
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union


# --- Coordinate utilities ---

# Earth radius in meters (WGS-84 mean)
_EARTH_RADIUS = 6_371_000.0


def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    """Convert lat/lon to local x/y (meters) relative to a reference point.

    Uses an equirectangular (flat-Earth) approximation, which is accurate
    to <0.1% for domains up to ~100 km at mid-latitudes.

    Parameters
    ----------
    lat, lon : float
        Point coordinates in decimal degrees.
    ref_lat, ref_lon : float
        Reference origin coordinates in decimal degrees.

    Returns
    -------
    x, y : float
        Easting and northing in meters relative to (ref_lat, ref_lon).
    """
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    ref_lat_r = math.radians(ref_lat)
    ref_lon_r = math.radians(ref_lon)

    x = _EARTH_RADIUS * (lon_r - ref_lon_r) * math.cos(ref_lat_r)
    y = _EARTH_RADIUS * (lat_r - ref_lat_r)
    return x, y


# --- Dataclasses ---


@dataclass
class TowerConfig:
    """Configuration for a single measurement tower."""

    name: str
    lat: float
    lon: float
    z_m: float

    # Local coordinates (computed from ref origin)
    x: float = 0.0
    y: float = 0.0

    def compute_local_xy(self, ref_lat: float, ref_lon: float):
        """Compute local x/y from lat/lon and reference origin."""
        self.x, self.y = latlon_to_xy(self.lat, self.lon, ref_lat, ref_lon)


@dataclass
class DomainConfig:
    """Configuration for the computational domain."""

    nx: int
    ny: int
    xmax: float
    ymax: float
    nz: int
    modes: Tuple[int, int] = (512, 512)
    halo: Optional[float] = None
    ref_lat: Optional[float] = None
    ref_lon: Optional[float] = None
    output_levels: Optional[List[int]] = None


@dataclass
class MetConfig:
    """Meteorological forcing data (scalar for single-time, list for timeseries)."""

    ustar: Optional[Union[float, List[float]]] = None
    mol: Union[float, List[float]] = 1e9
    wind_speed: Union[float, List[float]] = 5.0
    wind_dir: Union[float, List[float]] = 270.0
    z0: Optional[float] = None
    timestamps: Optional[List[str]] = None

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps in the timeseries."""
        if isinstance(self.ustar, list):
            return len(self.ustar)
        if isinstance(self.wind_speed, list):
            return len(self.wind_speed)
        return 1

    def get_step(self, i: int) -> dict:
        """Get met parameters for a single timestep.

        Returns
        -------
        dict with scalar values for ustar, mol, wind_speed, wind_dir, timestamp.
        """
        def _get(val, idx):
            if val is None:
                return None
            return val[idx] if isinstance(val, list) else val

        result = {
            "ustar": _get(self.ustar, i),
            "mol": _get(self.mol, i),
            "wind_speed": _get(self.wind_speed, i),
            "wind_dir": _get(self.wind_dir, i),
        }
        if self.z0 is not None:
            result["z0"] = self.z0
        if self.timestamps is not None:
            result["timestamp"] = self.timestamps[i]
        else:
            result["timestamp"] = i
        return result

    def validate(self):
        """Validate that at least one of ustar/z0 is provided, and list lengths match."""
        if self.ustar is None and self.z0 is None:
            raise ValueError("MetConfig requires at least one of 'ustar' or 'z0'")

        list_fields = {}
        for name in ("ustar", "mol", "wind_speed", "wind_dir"):
            val = getattr(self, name)
            if isinstance(val, list):
                list_fields[name] = len(val)

        if not list_fields:
            return  # all scalars, fine

        lengths = set(list_fields.values())
        if len(lengths) > 1:
            raise ValueError(
                f"Met timeseries arrays must all have the same length. "
                f"Got: {list_fields}"
            )

        n = lengths.pop()
        if self.timestamps is not None and len(self.timestamps) != n:
            raise ValueError(
                f"timestamps length ({len(self.timestamps)}) does not match "
                f"met array length ({n})"
            )


@dataclass
class SolverConfig:
    """Solver configuration."""

    closure: str = "MOST"
    precision: str = "single"
    footprint: bool = False
    surface_flux_shape: str = "diamond"
    analytic: bool = False
    src_loc: Optional[Tuple[float, float]] = None


@dataclass
class OutputConfig:
    """Output configuration."""

    format: str = "netcdf"
    directory: str = "./output"


@dataclass
class ParallelConfig:
    """Parallelism configuration."""

    num_threads: int = 1
    max_workers: int = 1
    use_cache: bool = False


@dataclass
class BLDFMConfig:
    """Top-level BLDFM configuration."""

    domain: DomainConfig
    towers: List[TowerConfig]
    met: MetConfig
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    def __post_init__(self):
        """Compute local tower coordinates from reference origin."""
        if self.domain.ref_lat is not None and self.domain.ref_lon is not None:
            for tower in self.towers:
                tower.compute_local_xy(self.domain.ref_lat, self.domain.ref_lon)

        self.met.validate()


# --- YAML parsing ---


def _parse_tower(d: dict) -> TowerConfig:
    return TowerConfig(
        name=d["name"],
        lat=d["lat"],
        lon=d["lon"],
        z_m=d["z_m"],
    )


def _parse_domain(d: dict) -> DomainConfig:
    modes = d.get("modes", [512, 512])
    output_levels = d.get("output_levels")
    return DomainConfig(
        nx=d["nx"],
        ny=d["ny"],
        xmax=float(d["xmax"]),
        ymax=float(d["ymax"]),
        nz=d["nz"],
        modes=tuple(modes),
        halo=d.get("halo"),
        ref_lat=d.get("ref_lat"),
        ref_lon=d.get("ref_lon"),
        output_levels=output_levels,
    )


def _parse_met(d: dict) -> MetConfig:
    return MetConfig(
        ustar=d.get("ustar"),
        mol=d.get("mol", 1e9),
        wind_speed=d.get("wind_speed", 5.0),
        wind_dir=d.get("wind_dir", 270.0),
        z0=d.get("z0"),
        timestamps=d.get("timestamps"),
    )


def _parse_solver(d: dict) -> SolverConfig:
    if d is None:
        return SolverConfig()
    src_loc = d.get("src_loc")
    if src_loc is not None:
        src_loc = tuple(src_loc)
    return SolverConfig(
        closure=d.get("closure", "MOST"),
        precision=d.get("precision", "single"),
        footprint=d.get("footprint", False),
        surface_flux_shape=d.get("surface_flux_shape", "diamond"),
        analytic=d.get("analytic", False),
        src_loc=src_loc,
    )


def _parse_output(d: dict) -> OutputConfig:
    if d is None:
        return OutputConfig()
    return OutputConfig(
        format=d.get("format", "netcdf"),
        directory=d.get("directory", "./output"),
    )


def _parse_parallel(d: dict) -> ParallelConfig:
    if d is None:
        return ParallelConfig()
    return ParallelConfig(
        num_threads=d.get("num_threads", 1),
        max_workers=d.get("max_workers", 1),
        use_cache=d.get("use_cache", False),
    )


def load_config(path: Union[str, Path]) -> BLDFMConfig:
    """Load a BLDFM configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    BLDFMConfig
        Parsed and validated configuration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return parse_config_dict(raw)


def parse_config_dict(raw: dict) -> BLDFMConfig:
    """Parse a BLDFM configuration from a dictionary.

    Parameters
    ----------
    raw : dict
        Raw configuration dictionary (e.g. from YAML).

    Returns
    -------
    BLDFMConfig
        Parsed and validated configuration.
    """
    if "domain" not in raw:
        raise ValueError("Config must include 'domain' section")
    if "towers" not in raw:
        raise ValueError("Config must include 'towers' section")
    if "met" not in raw:
        raise ValueError("Config must include 'met' section")

    domain = _parse_domain(raw["domain"])
    towers = [_parse_tower(t) for t in raw["towers"]]
    met = _parse_met(raw["met"])
    solver = _parse_solver(raw.get("solver"))
    output = _parse_output(raw.get("output"))
    parallel = _parse_parallel(raw.get("parallel"))

    return BLDFMConfig(
        domain=domain,
        towers=towers,
        met=met,
        solver=solver,
        output=output,
        parallel=parallel,
    )
