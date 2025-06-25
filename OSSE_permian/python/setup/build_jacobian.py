## Run from inversion directory in IMI
root_dir = '/nobackupnfs1/hnesser/BC_sensitivity/Permian_BC_test'

import glob
import pickle
from copy import deepcopy as dc
import xarray as xr
import numpy as np

def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)

## Get state vector
sv = xr.open_dataset(f'{root_dir}/StateVector.nc')['StateVector']

## Now construct the Jacobian
# First get the lats and lons that exclude the buffer grid cells
lat_min = float(sv.lat.min().values)
lat_max = float(sv.lat.max().values)
lon_min = float(sv.lon.min().values)
lon_max = float(sv.lon.max().values)

# Adjust to account for GEOS-Chem buffer grid cells
degx = 2 * 0.3125
degy = 2 * 0.25

xlim = [lon_min + degx, lon_max - degx]
ylim = [lat_min + degy, lat_max - degy]

# Define preliminary error statistics
# values from Chen et al., 2023, https://doi.org/10.5194/egusphere-2022-1504
# So = observational error specified in config file
# p = average number of observations contained within each superobservation
r_retrieval = 0.55
s_transport = 4.5
s_super = lambda s0, p: np.sqrt(s0**2 * (((1 - r_retrieval) / p) + r_retrieval) + s_transport**2)
s_super0_1 = s_super(15, 1)

# Get a list of the Jacobian files
files = glob.glob(f'{root_dir}/inversion/data_converted/*.pkl')
files.sort()

# Initialize vectors for the observations and prior model output,
# and initialize a matrix for the Jacobian
y = np.array([])
Fxa = np.array([])
lon = np.array([])
lat = np.array([])
K = np.array([]).reshape(0, int(sv.max() + 4))
so = np.array([])
count = np.array([])

i = 0
# Iterate through
for f in files:
    # Load TROPOMI/GEOS-Chem data and Jacobian matrix
    data = load_obj(f)

    # Skip if there are no observations
    if data['obs_GC'].shape[0] == 0:
        continue

    # Get data
    obs_GC = data['obs_GC']

    # Get an index corresponding to only the data within the lat/lon
    # bounds
    obs_idx = np.where(
        (obs_GC[:, 2] > xlim[0])
        & (obs_GC[:, 2] < xlim[1])
        & (obs_GC[:, 3] > ylim[0])
        & (obs_GC[:, 3] < ylim[1])
    )[0]

    # Skip if no data is in bounds
    if len(obs_idx) == 0:
        continue

    # Get TROPOMI and GC data within the bounds
    obs_GC_sub = dc(obs_GC)[obs_idx, :]
    y_f = obs_GC_sub[:, 0]
    Fxa_f = obs_GC_sub[:, 1]
    lon_f = obs_GC_sub[:, 2]
    lat_f = obs_GC_sub[:, 3]
    count_f = obs_GC_sub[:, 4]

    # Calculate the observing system errors
    s_super0_p = np.array(
        [s_super(15, p) if p >= 1 else s_super0_1 for p in count_f]
    )
    so_f = np.power(15, 2)
    gP = s_super0_p**2 / s_super0_1**2
    so_f = gP * so_f
    so_f = [obs if obs > 0 else 1 for obs in so_f]

    # Load Jacobian and subset
    K_f = data['K'][obs_idx, :]
    # K_f = K_f[:, sv_idx]

    # Append
    y = np.append(y, y_f)
    Fxa = np.append(Fxa, Fxa_f)
    lon = np.append(lon, lon_f)
    lat = np.append(lat, lat_f)
    so = np.append(so, so_f)
    K = np.concatenate((K, K_f))
    count = np.concatenate((count, count_f))

# Get a list of the visualization files (for data counts)
lat_full = np.array([])
lon_full = np.array([])
files = glob.glob(f'{root_dir}/inversion/data_visualization/*.pkl')
files.sort()
for f in files:
    # Load TROPOMI/GEOS-Chem data and Jacobian matrix
    data = load_obj(f)

    # Skip if there are no observations
    if data['obs_GC'].shape[0] == 0:
        continue

    # Get data
    obs_GC = data['obs_GC']

    # Get an index corresponding to only the data within the lat/lon
    # bounds
    obs_idx = np.where(
        (obs_GC[:, 2] > xlim[0])
        & (obs_GC[:, 2] < xlim[1])
        & (obs_GC[:, 3] > ylim[0])
        & (obs_GC[:, 3] < ylim[1])
    )[0]

    lon_full = np.append(lon_full, obs_GC[obs_idx, 2])
    lat_full = np.append(lat_full, obs_GC[obs_idx, 3])

np.save(f'{root_dir}/data/y.npy', y)
np.save(f'{root_dir}/data/Fxa.npy', Fxa)
np.save(f'{root_dir}/data/lon.npy', lon)
np.save(f'{root_dir}/data/lat.npy', lat)
np.save(f'{root_dir}/data/lon_full.npy', lon_full)
np.save(f'{root_dir}/data/lat_full.npy', lat_full)
np.save(f'{root_dir}/data/so.npy', so)
np.save(f'{root_dir}/data/K.npy', K)
np.save(f'{root_dir}/data/count.npy', count)