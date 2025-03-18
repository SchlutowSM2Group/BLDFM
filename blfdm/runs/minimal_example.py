import numpy as np
import matplotlib.pyplot as plt

from ..src.most import vertical_profiles
from ..src.utils import ideal_source
from ..src.solver import steady_state_transport_solver

nxy         = 512, 256
nz          = 10
domain      = 2000.0, 1000.0
meas_height = 6.0
wind        = 2.0, 0.5
ustar       = 0.5

q0 = ideal_source(nxy, domain)

z, profs = vertical_profiles(
        nz, 
        meas_height, 
        wind,
        ustar )

p0, pm00, pm, qm = steady_state_transport_solver(q0, z, profs, domain)

print('Minimal example for neutrally stratified BL and default settings.')
print()
plt.imshow(p0,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Concentration at z0")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
plt.imshow(pm,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Concentration at zm")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
plt.imshow(qm,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.title("Vertical kinematic flux at zm")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()



