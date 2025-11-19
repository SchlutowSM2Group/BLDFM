from pathlib import Path

from .solver import steady_state_transport_solver
from .solver3d import steady_state_transport_solver_3d
from .utils import (
    compute_wind_fields,
    point_source,
    ideal_source,
    setup_logging,
    get_logger,
)

Path("logs").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

setup_logging()

__all__ = [
    "steady_state_transport_solver",
    "steady_state_transport_solver_3d",
    "compute_wind_fields",
    "point_source",
    "ideal_source",
    "setup_logging",
    "get_logger",
]
