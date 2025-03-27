import numpy as np
import matplotlib.pyplot as plt

from ..src.most import vertical_profiles
from ..src.utils import ideal_source
from ..src.solver import steady_state_transport_solver
from ..src.calc_footprint_FFP import FFP

nxy         = 512, 256
nz          = 100
modes       = 2048, 2048
domain      = 1000.0, 700.0
fetch       = 4000.0
meas_pt     = 500.0, 0.0
meas_height = 6.0
wind        = 0.0, -6.0
ustar       = 0.5
z0          = 0.1
mol         = -100.0

surf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(
        nz, 
        meas_height, 
        wind,
        ustar,
        mol = mol,
        z0 = z0)

p0, p00, p, q = steady_state_transport_solver(
        surf_flx, 
        z, 
        profs, 
        domain,
        modes = modes,
        meas_pt = meas_pt,
        footprint = True,
        fetch = fetch
        )

plt.imshow(p,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Concentration footprint")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
plt.imshow(q,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("BLDFM Flux footprint")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()

############################################################

umean = np.sqrt(wind[0]**2+wind[1]**2)

FFP_res = FFP(
        zm = meas_height, 
        z0 = z0, 
        umean = umean,
        h=2000., 
        ol = mol, 
        sigmav=0.6, 
        ustar = 0.437, 
        wind_dir = 0.0,
        nx = 1000,
        rs= [20., 40., 60., 80.],
        fig = False)

plt.imshow(FFP_res["f_2d"],origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("FFP Flux footprint")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()


