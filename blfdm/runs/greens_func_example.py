import numpy as np
import matplotlib.pyplot as plt

from ..src.most import vertical_profiles
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
surf_bg     = 1.0

q0 = ideal_source(nxy, domain)

z, profs = vertical_profiles(
        nz, 
        meas_height, 
        wind,
        ustar )

# Direct computation
p0, p00, p, q = steady_state_transport_solver(
        q0, 
        z, 
        profs, 
        domain,
        modes = modes,
        surf_bg = surf_bg,
        meas_pt = meas_pt,
        footprint = False,
        fetch = fetch
        )

plt.imshow(p,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Concentration at zm")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(domain[0]/2,domain[1]/2,'ro') 
plt.colorbar()
plt.show()
plt.imshow(q,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Vertical kinematic flux at zm")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(domain[0]/2,domain[1]/2,'ro') 
plt.colorbar()
plt.show()
print('Direct method')
print('p at meas_pt  = ',p[nxy[1]//2,nxy[0]//2])
print('q at meas_pt  = ',q[nxy[1]//2,nxy[0]//2])
print()

# Computation with Green's function
p0, p00, p, q = steady_state_transport_solver(
        q0, 
        z, 
        profs, 
        domain,
        modes = modes,
        meas_pt = meas_pt,
        footprint = True,
        fetch = fetch
        )

pm = surf_bg + point_measurement(q0,p)
qm = point_measurement(q0,q)

print('Convolution with Green function')
print('p at meas_pt = ',pm)
print('q at meas_pt = ',qm)
print()



