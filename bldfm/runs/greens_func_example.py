import numpy as np
import matplotlib.pyplot as plt

from ..src.pbl_model import vertical_profiles
from ..src.utils import ideal_source
from ..src.utils import point_measurement
from ..src.solver import steady_state_transport_solver

nxy         = 512, 256
nz          = 10
modes       = 512, 512
domain      = 2000.0, 1000.0
fetch       = 2000.0
meas_pt     = 300.0, 400.0
meas_height = 6.0
wind        = -2.0, 2.0
ustar       = 0.5
srf_bg_conc = 1.0

srf_flx = ideal_source(nxy, domain)

z, profs = vertical_profiles(
        nz, 
        meas_height, 
        wind,
        ustar )

# Direct computation
srf_conc, bg_conc, conc, flx = steady_state_transport_solver(
        srf_flx, 
        z, 
        profs, 
        domain,
        modes = modes,
        srf_bg_conc = srf_bg_conc,
        meas_pt = meas_pt,
        footprint = False,
        fetch = fetch
        )

plt.imshow(conc,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Concentration at zm")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(domain[0]/2,domain[1]/2,'ro') 
plt.colorbar()
plt.show()
plt.imshow(flx,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Vertical kinematic flux at zm")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(domain[0]/2,domain[1]/2,'ro') 
plt.colorbar()
plt.show()
print('Direct method')
print('conc at meas_pt  = ',conc[nxy[1]//2,nxy[0]//2])
print('flx  at meas_pt  = ',flx[nxy[1]//2,nxy[0]//2])
print()

# Computation with Green's function
srf_conc, bg_conc, conc, flx = steady_state_transport_solver(
        srf_flx, 
        z, 
        profs, 
        domain,
        modes = modes,
        meas_pt = meas_pt,
        footprint = True,
        fetch = fetch
        )

conc = srf_bg_conc + point_measurement(srf_flx, conc)
flx  = point_measurement(srf_flx, flx)

print('Convolution with Green function')
print('conc at meas_pt = ',conc)
print('flx at meas_pt = ',flx)
print()



