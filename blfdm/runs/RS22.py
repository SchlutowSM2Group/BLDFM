import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from ..src.most import vertical_profiles
from ..src.utils import point_source
from ..src.utils import point_measurement
from ..src.utils import compute_wind_fields
from ..src.solver import steady_state_transport_solver

nxy         = 256, 128
nz          = 30
modes       = 1024, 1024
domain      = 100.0, 50.0
fetch       = 600.0

start_doy   = 228 # 228
end_doy     = 230
start_hour  = 6.0
end_hour    = 21.0

ustar_min   = 0.2
flxo_min    = 50.0 # [nmol m-2 s-1]

meas_pt     = 50.0, 25.0
meas_height = 3.9
dist        = 20.0  # [m] distance of release pipe from tower
azi         = 253.0 # [deg] azimuth of release pipe measured from tower from North
vfr         = 1.75  # [L min-1] volumetric flow rate at release pipe

# Convert methane flux to mol s-1
dCH4 = 0.664 # density of methane for 18 deg C under normal pressure [kg m-3] 
mCH4 = 16.04 # molecular mass of methane [g mol-1]

vfr  = vfr / 60 * 0.001 # [m3 s-1]
mCH4 = mCH4 * 1e-3      # [kg mol-1] 
mfr  = vfr * dCH4       # mass flow rate [kg s-1]
nfr  = mfr / mCH4       # particle flow rate [mol s-1]

print("particle flow rate = %.5f [mol s-1]"% nfr)

# df = pd.read_csv(file_path, na_values='<missing>')

# Read data file
file_path = "./data/Rey-Sanchez_et_al_2022/Aug_day_2019228_to_2019238_20Hz_L1_15min.mat"
#file_path = "./data/Rey-Sanchez_et_al_2022/Aug_night_2019228_to_2019247_20Hz_L1_15min.mat"

data = scipy.io.loadmat(file_path, simplify_cells=True)

df = data['data']

doy   = df['DOY']
time  = df['time'] 
ustar = df['ustar'] 
flxo  = df['wm']  # CH4 flux (after all corrections) [nmol CH4 m-2 s-1']

# Filter
msk = (doy   >= start_doy) & \
      (doy   <= end_doy) & \
      (time  >= start_hour) & \
      (time  <= end_hour) & \
      (ustar >= ustar_min) & \
      (flxo  >= flxo_min)

doy   = df['DOY'][msk].copy()
time  = df['time'][msk].copy()
flxo  = df['wm'][msk].copy()   # CH4 flux (after all corrections) [nmol CH4 m-2 s-1']
wd    = df['WD'][msk]   # wind direction [degrees clockwise from true north]
ubar  = df['ubar'][msk] # average wind speed
ustar = df['ustar'][msk].copy() 
mol   = df['L'][msk]
alp   = df['alpha'][msk] # vertical rotation angle in wind coordinate rotation [radians]

alp = alp / np.pi * 180.0 # [deg]

nt = time.shape[0]

plt.plot(doy+time/24,wd,'.')
plt.title('Wind direction')
plt.show()

plt.plot(doy+time/24,flxo,'.')
plt.title('Observed flux')
plt.show()


Dx = dist * np.cos((90-azi) / 180.0 * np.pi)
Dy = dist * np.sin((90-azi) / 180.0 * np.pi)

src_pt = meas_pt[0] + Dx, meas_pt[1] + Dy

surf_flx = point_source(nxy, domain, src_pt) * nfr

#ix = int(src_pt[0] / domain[0] * nxy[0])  
#iy = int(src_pt[1] / domain[1] * nxy[1])  
#surf_flx = np.zeros(nxy)
#surf_flx[iy,ix] = 1.0

plt.imshow(surf_flx, origin="lower", extent=[0,domain[0],0,domain[1]])
plt.plot(meas_pt[0], meas_pt[1],"ro")
plt.plot(src_pt[0], src_pt[1],"go")
plt.colorbar()
plt.show()

flx_footprint_mean  = np.zeros((nxy[1],nxy[0]))
conc_footprint_mean = np.zeros((nxy[1],nxy[0]))
flxm = np.zeros(nt)

for n in range(nt): #range(92,93):
 
    print("n    =",n)
    print("DOY  =",doy[n])
    print("time = %.3f h"% time[n])
    print("alp  = %.3f deg"% alp[n])
    print()
 
    wind = compute_wind_fields(ubar[n], wd[n]-180)
 
    z, profs = vertical_profiles(
         nz, 
         meas_height, 
         wind,
         ustar[n],
         mol[n],
         z0 = 0.1,
         z0_min = 0.05,
         z0_max = 0.15)
 
    conc0, conc00, conc_footprint, flx_footprint = steady_state_transport_solver(
         surf_flx,
         z, 
         profs, 
         domain, 
         modes   = modes,
         meas_pt = meas_pt,
         fetch   = fetch,
         footprint = True)

    # taking the measurement
    flxm[n] = point_measurement(flx_footprint, surf_flx) * 1e9

    #plt.imshow(flx, origin="lower", extent=[0,domain[0],0,domain[1]])
    #plt.plot(domain[0]/2, domain[1]/2,"ro")
    #plt.colorbar()
    #plt.show()

    #plt.imshow(conc, origin="lower", extent=[0,domain[0],0,domain[1]])
    #plt.plot(domain[0]/2, domain[1]/2,"ro")
    #plt.colorbar()
    #plt.show()

    # taking the measurement
    # flxm[n] = flx[nxy[1]//2, nxy[0]//2] * 1e9 # [nmol m-2 s-1]
    flx_footprint_mean = flx_footprint_mean + flx_footprint
    conc_footprint_mean = conc_footprint_mean + conc_footprint
    print("flxm = %.1f nmol m-2 s-1"% flxm[n])
    print()

plt.plot(flxo,flxm,'.')
plt.plot(flxo,flxo)
plt.loglog()
plt.show()

# normalization
flx_footprint = flx_footprint / nt
conc_footprint = conc_footprint / nt

plt.imshow(flx_footprint_mean, origin="lower", extent=[0,domain[0],0,domain[1]])
plt.title("Average flux footprint")
plt.plot(meas_pt[0], meas_pt[1],"ro")
plt.plot(src_pt[0], src_pt[1],"go")
plt.show()

plt.imshow(conc_footprint_mean, origin="lower", extent=[0,domain[0],0,domain[1]])
plt.title("Average concentration footprint")
plt.plot(meas_pt[0], meas_pt[1],"ro")
plt.plot(src_pt[0], src_pt[1],"go")
plt.show()
