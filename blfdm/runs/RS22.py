import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from ..src.most import vertical_profiles
from ..src.utils import point_source
from ..src.utils import compute_wind_fields
from ..src.solver import steady_state_transport_solver

nxy         = 512, 512
nz          = 10
modes       = 512, 512
domain      = 200.0, 200.0
fetch       = 400.0
meas_pt     = 100.0, 100.0
meas_height = 5.0

# df = pd.read_csv(file_path, na_values='<missing>')

# Read data file
file_path = "./data/Rey-Sanchez_et_al_2022/Aug_day_2019228_to_2019238_20Hz_L1_15min.mat"
df = scipy.io.loadmat(file_path)

time  = df['data']['time'][0][0] 
qm    = df['data']['wm'][0][0]   # CH4 flux (after all corrections) [nmol CH4 m-2 s-1']
wd    = df['data']['WD'][0][0]   # wind direction [degrees clockwise from true north]
ubar  = df['data']['ubar'][0][0] # average wind speed
ustar = df['data']['ustar'][0][0] 
mol   = df['data']['L'][0][0]


# Metadata
dist = 20.0  # [m] distance of release pipe from tower
azi  = 253.0 # [deg] azimuth of release pipe
vfr  = 1.75  # [L min-1] volumetric flow rate at release pipe

Dx = dist * np.cos((90-azi) / 180.0 * np.pi)
Dy = dist * np.sin((90-azi) / 180.0 * np.pi)

src_pt = meas_pt[0] + Dx, meas_pt[1] + Dy

surf_flx = point_source(nxy, domain, src_pt)

# plt.imshow(surf_flx, origin="lower", extent=[0,domain[0],0,domain[1]])
# plt.plot(meas_pt[0], meas_pt[1],"ro")
# plt.plot(src_pt[0], src_pt[1],"go")
# plt.show()

 
for n in range(time.shape[0]):
 
    print("n    =",n)
    print("time =",time[n][0])
 
    wind = compute_wind_fields(ubar[n][0], wd[n][0])
 
    z, profs = vertical_profiles(
         nz, 
         meas_height, 
         wind,
         ustar[n][0],
         mol[n][0])
 
    p0, p00, p, q = steady_state_transport_solver(
         surf_flx,
         z, 
         profs, 
         domain, 
         modes   = modes,
         meas_pt = meas_pt,
         fetch   = fetch)
 
    plt.imshow(q, origin="lower", extent=[0,domain[0],0,domain[1]])
    plt.plot(meas_pt[0], meas_pt[1],"ro")
    plt.plot(src_pt[0], src_pt[1],"go")
    # plt.show()

    qm = q[nxy[1]//2,nxy[0]//2]
    print("qm =",qm)
    print()

plt.show()
