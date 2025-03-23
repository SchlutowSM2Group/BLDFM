import numpy as np
import pandas as pd

from ..src.most import vertical_profiles
from ..src.solver import steady_state_transport_solver
import scipy.io

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




