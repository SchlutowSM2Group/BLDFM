import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

from ..src.most import vertical_profiles
from ..src.utils import point_source
from ..src.solver import steady_state_transport_solver

nxy    = 512, 256
domain = 1000.0, 500.0

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

srcpt = 100.0, 100.0

q0 = point_source(nxy, domain, srcpt)

plt.imshow(q0,origin="lower",extent=[0,domain[0],0,domain[1]])
plt.show()


