import numpy as np
import matplotlib.pyplot as plt

from ..src.most import vertical_profiles
from ..src.utils import ideal_source
from ..src.solver import steady_state_transport_solver

nxy         = 512, 256
nz          = 10
modes       = 512, 512
domain      = 2000.0, 1000.0
fetch       = 2000.0
meas_pt     = 1500.0, 700.0
meas_height = 6.0
wind        = 4.0, 2.0
ustar       = 0.5

q0 = ideal_source(nxy, domain)

z, profs = vertical_profiles(
        nz, 
        meas_height, 
        wind,
        ustar )

p0, pm00, pm, qm = steady_state_transport_solver(
        q0, 
        z, 
        profs, 
        domain,
        modes = modes,
        meas_pt = meas_pt,
        footprint = True,
        fetch = fetch
        )

plt.imshow(pm,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Concentration footprint")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
plt.imshow(qm,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Flux footprint")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()



