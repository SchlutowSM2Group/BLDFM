"""
Example module demonstrating point measurements.
"""

import matplotlib.pyplot as plt

from bldfm.pbl_model import vertical_profiles
from bldfm.utils import ideal_source, get_logger, point_measurement
from bldfm.solver import steady_state_transport_solver

logger = get_logger("point_measurement_example")

nxy = 256, 256
nz = 64
domain = 500.0, 500.0
meas_pt = 300.0, 300.0
meas_height = 10.0
wind = 4.0, 4.0
z0 = 0.1

# point source in the middle of the domain at (x, y) = (250m, 250m)
surf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(nz, meas_height, wind, z0=z0)

# compute flux footprint, note that surf_flx is a dummy input for shape
_, _, _, flx = steady_state_transport_solver(
    surf_flx,
    z,
    profs,
    domain,
    nz,
    meas_pt=meas_pt,
    footprint=True,
)

# point measurement by convolution of surface flux with footprint
flx_meas_convolve = point_measurement(surf_flx, flx)

# direct computation of flux field from surface flux
srf_conc, bg_conc, conc, flx = steady_state_transport_solver(
    surf_flx,
    z,
    profs,
    domain,
    nz,
    meas_pt=meas_pt,
)

# measurement by evaluating flux from flow field
flx_meas_direct = flx[nxy[1] // 2, nxy[0] // 2]

if __name__ == "__main__":

    print(
        "Flux measurement at (x, y) = (300m, 300m) by convolution of surface flux with footprint:",
        flx_meas_convolve,
    )
    print(
        "Flux measurement at (x, y) = (300m, 300m) by direct computation of the flux' flow field:",
        flx_meas_convolve,
    )
# %%
