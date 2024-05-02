## Run from inversion directory in IMI
root_dir = '/n/holyscratch01/jacob_lab/hnesser/permian_OSSE_dev'

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

# ## Update state vector
# # Load prior simulation domain
# prior_emis_dir = f'{root_dir}/Permian_OSSE/jacobian_runs/Permian_OSSE_0000/OutputDir'
# prior_emis = xr.open_dataset(f'{prior_emis_dir}/HEMCO_diagnostics.202005010000.nc')

# # Isolate one variable and rename
# sv = dc(prior_emis['EmisCH4_Total']).squeeze(drop=True)
# sv = sv.rename('Clusters')
# sv.attrs = {'long_name' : 'Clusters generated for analytical inversion', 
#             'units' : 'none'}

# # Fill in the cluster values
# sv.values = np.arange(1, sv.shape[0]*sv.shape[1] + 1).reshape(sv.shape)
# print(sv.values.max(), 'state vector elements')

# # Save out
# sv.to_netcdf(f'{root_dir}/OSSE/clusters_permian_subset.nc')

# ## Now get a Boolean vector that maps between the original state vector
# ## and the updated/subsetted state vector
# # Load the original
# sv_orig = xr.open_dataset(f'{root_dir}/clusters_permian_long.nc')
# sv_orig = sv_orig['StateVector'].squeeze(drop=True)

# # Copy it to make the map
# sv_idx = dc(sv_orig)
# sv_idx = sv_idx.where(sv_idx < 0, True).astype(bool)
# sv_idx = sv_idx.where(sv_orig.lon.isin(sv.lon) & sv_orig.lat.isin(sv.lat), 
#                       False)
# sv_idx = sv_idx.values.flatten()

## Now construct the Jacobian
# First get the lats and lons that exclude the buffer grid cells
lat_min = float(sv.lat.min().values)
lat_max = float(sv.lat.max().values)
lon_min = float(sv.lon.min().values)
lon_max = float(sv.lon.max().values)

# Adjust to account for GEOS-Chem buffer grid cells
degx = 4 * 0.3125
degy = 4 * 0.25

xlim = [lon_min + degx, lon_max - degx]
ylim = [lat_min + degy, lat_max - degy]

# Get a list of the Jacobian files
files = glob.glob(f'{root_dir}/Permian_OSSE/inversion/data_visualization/*.pkl')
files.sort()

# Initialize vectors for the observations and prior model output,
# and initialize a matrix for the Jacobian
y = np.array([])
Fxa = np.array([])
lon = np.array([])
lat = np.array([])
K = np.array([]).reshape(0, int(sv.max()))

y_full = np.array([])
Fxa_full = np.array([])
lon_full = np.array([])
lat_full = np.array([])

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
        (obs_GC[:, 2] >= xlim[0])
        & (obs_GC[:, 2] <= xlim[1])
        & (obs_GC[:, 3] >= ylim[0])
        & (obs_GC[:, 3] <= ylim[1])
    )[0]

    # Get an index corresponding to only the data within the
    # larger lat/lon bounds
    outside = ((obs_GC[:, 2] >= lon_min)
               & (obs_GC[:, 2] <= lon_max)
               & (obs_GC[:, 3] >= lat_min)
               & (obs_GC[:, 3] <= lat_max))
    inside = ((obs_GC[:, 2] >= xlim[0])
              & (obs_GC[:, 2] <= xlim[1])
              & (obs_GC[:, 3] >= ylim[0])
              & (obs_GC[:, 3] <= ylim[1])) 
    obs_idx_full = outside & ~inside

    # Skip if no data is in bounds
    if len(obs_idx) == 0:
        continue

    # Get TROPOMI and GC data within the bounds
    obs_GC_sub = dc(obs_GC)[obs_idx, :]
    y_f = obs_GC_sub[:, 0]
    Fxa_f = obs_GC_sub[:, 1]
    lon_f = obs_GC_sub[:, 2]
    lat_f = obs_GC_sub[:, 3]

    # Load Jacobian and subset
    K_f = data['K'][obs_idx, :]
    # K_f = K_f[:, sv_idx]

    # Get TROPOMI and GC data within the larger domain
    obs_GC_full = dc(obs_GC)[obs_idx_full, :]
    y_full_f = obs_GC_full[:, 0]
    Fxa_full_f = obs_GC_full[:, 1]
    lon_full_f = obs_GC_full[:, 2]
    lat_full_f = obs_GC_full[:, 3]

    # Append
    y = np.append(y, y_f)
    Fxa = np.append(Fxa, Fxa_f)
    lon = np.append(lon, lon_f)
    lat = np.append(lat, lat_f)
    K = np.concatenate((K, K_f))

    y_full = np.append(y_full, y_full_f)
    Fxa_full = np.append(Fxa_full, Fxa_full_f)
    lon_full = np.append(lon_full, lon_full_f)
    lat_full = np.append(lat_full, lat_full_f)

np.save(f'{root_dir}/data_OSSE/y.npy', y)
np.save(f'{root_dir}/data_OSSE/Fxa.npy', Fxa)
np.save(f'{root_dir}/data_OSSE/lon.npy', lon)
np.save(f'{root_dir}/data_OSSE/lat.npy', lat)
np.save(f'{root_dir}/data_OSSE/K.npy', K)

np.save(f'{root_dir}/data_OSSE/y_full.npy', y_full)
np.save(f'{root_dir}/data_OSSE/Fxa_full.npy', Fxa_full)
np.save(f'{root_dir}/data_OSSE/lon_full.npy', lon_full)
np.save(f'{root_dir}/data_OSSE/lat_full.npy', lat_full)