import numpy as np
import pandas as pd

from ..src.most import vertical_profiles
from ..src.solver import steady_state_transport_solver

file_path = "./data/footprint model experiment data/03-24 basic data base from matlab.csv"
df = pd.read_csv(file_path, na_values='<missing>')

print(df.var)

