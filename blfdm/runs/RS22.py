import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from ..src.most import vertical_profiles
from ..src.utils import point_source
from ..src.utils import compute_wind_fields
from ..src.solver import steady_state_transport_solver

nxy         = 512, 256
nz          = 10
modes       = 512, 512
domain      = 100.0, 50.0
fetch       = 400.0
meas_pt     = 50.0, 20.0
meas_height = 3.9
start_doy   = 228
end_doy     = 230
start_hour  = 6.0
end_hour    = 21.0
ustar_min   = 0.2

dist = 20.0  # [m] distance of release pipe from tower
azi  = 253.0 # [deg] azimuth of release pipe measured from tower from North
vfr  = 1.75  # [L min-1] volumetric flow rate at release pipe

# df = pd.read_csv(file_path, na_values='<missing>')

# Read data file
file_path = "./data/Rey-Sanchez_et_al_2022/Aug_day_2019228_to_2019238_20Hz_L1_15min.mat"

data = scipy.io.loadmat(file_path, simplify_cells=True)

df = data['data']

doy   = df['DOY']
time  = df['time'] 
ustar = df['ustar'] 

# Filter
msk = (doy   >= start_doy) & \
      (doy   <= end_doy) & \
      (time  >= start_hour) & \
      (time  <= end_hour) & \
      (ustar >= ustar_min)

doy   = df['DOY'][msk].copy()
time  = df['time'][msk].copy()
flxo  = df['wm'][msk]   # CH4 flux (after all corrections) [nmol CH4 m-2 s-1']
wd    = df['WD'][msk]   # wind direction [degrees clockwise from true north]
ubar  = df['ubar'][msk] # average wind speed
ustar = df['ustar'][msk].copy() 
mol   = df['L'][msk]

nt = time.shape[0]

plt.plot(wd,'.')
plt.show()

plt.plot(flxo,'.')
plt.show()


Dx = dist * np.cos((90-azi) / 180.0 * np.pi)
Dy = dist * np.sin((90-azi) / 180.0 * np.pi)

src_pt = meas_pt[0] + Dx, meas_pt[1] + Dy

#surf_flx = point_source(nxy, domain, src_pt)

ix = int(src_pt[0] / domain[0] * nxy[0])  
iy = int(src_pt[1] / domain[1] * nxy[1])  
surf_flx = np.zeros(nxy)
surf_flx[iy,ix] = 1.0

plt.imshow(surf_flx, origin="lower", extent=[0,domain[0],0,domain[1]])
plt.plot(meas_pt[0], meas_pt[1],"ro")
# plt.plot(src_pt[0], src_pt[1],"go")
plt.colorbar()
plt.show()

flx_mean = np.zeros(nxy)
flxm = np.zeros(nt)

for n in range(92,93):
 
    print("n    =",n)
    print("DOY  =",doy[n])
    print("time =",time[n])
    print()
 
    wind = compute_wind_fields(ubar[n], wd[n]-180)
 
    z, profs = vertical_profiles(
         nz, 
         meas_height, 
         wind,
         ustar[n],
         mol[n],
         # z0 = 0.002,
         z0_min = 0.001,
         z0_max = 0.100)
 
    conc0, conc00, conc, flx = steady_state_transport_solver(
         surf_flx,
         z, 
         profs, 
         domain, 
         modes   = modes,
         meas_pt = meas_pt,
         fetch   = fetch)

    plt.imshow(flx, origin="lower", extent=[0,domain[0],0,domain[1]])
    plt.plot(meas_pt[0], meas_pt[1],"ro")
    plt.plot(src_pt[0], src_pt[1],"go")
    plt.colorbar()
    plt.show()

    plt.imshow(conc, origin="lower", extent=[0,domain[0],0,domain[1]])
    plt.plot(meas_pt[0], meas_pt[1],"ro")
    plt.plot(src_pt[0], src_pt[1],"go")
    plt.colorbar()
    plt.show()

    # taking the measurement
    flxm[n] = flx[nxy[1]//2, nxy[0]//2]
    flx_mean = flx_mean + flx
    print("flxm =",flxm[n])
    print()

plt.plot(flxo,flxm,'.')
plt.loglog()
plt.show()

# plt.imshow(qmean, origin="lower", extent=[0,domain[0],0,domain[1]])
# plt.plot(meas_pt[0], meas_pt[1],"ro")
# plt.plot(src_pt[0], src_pt[1],"go")
# plt.show()
