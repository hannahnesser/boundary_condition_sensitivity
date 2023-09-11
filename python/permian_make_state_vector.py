import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import sys
sys.path.append('.')
import gcpy as gc

data_dir = '../data/'

sv = xr.open_dataset(f'{data_dir}clusters_permian.nc')['Clusters']

# Get the sub-domain
sv_ux, sv_cnts = np.unique(sv.values, return_counts=True)
sv_sub = sv.where(sv <= sv_ux[sv_cnts == 1].max(), drop=True)

# Subset it for six buffer grid cells around the domain
buffer = 7
lat_delta = 0.25
lon_delta = 0.3125
lats, lons = gc.create_gc_grid(sv_sub.lat.min() - buffer*lat_delta,
                               sv_sub.lat.max() + buffer*lat_delta,
                               lat_delta,
                               sv_sub.lon.min() - buffer*lon_delta,
                               sv_sub.lon.max() + buffer*lon_delta,
                               lon_delta,
                               return_xarray=False)
sv = sv.sel(lon=lons, lat=lats)

# set the outer grid cells to -1
sub_buffer = 4
idx = ((sv.lat < sv.lat.min() + sub_buffer*lat_delta) |
       (sv.lat > sv.lat.max() - sub_buffer*lat_delta) |
       (sv.lon < sv.lon.min() + sub_buffer*lon_delta) |
       (sv.lon > sv.lon.max() - sub_buffer*lon_delta))
sub_shape = sv.where(~idx, drop=True).shape
print(sub_shape)

sv = sv.where(~idx, 0)

# Set all values to Null before filling in the state vector values
sv = sv.where(sv == 0)

# Fill in the cluster values
count = int(1)
for i in range(len(sv.lat)):
    for j in range(len(sv.lon)):
        if sv[i, j].isnull():
            sv[i, j] = count
            count += int(1)

tmp = sv.where(sv > 0, drop=True)
sv = sv.astype(int)

import format_plots as fp
fig, ax = fp.get_figax(maps=True, lats=sv.lat, lons=sv.lon)
sv.plot(ax=ax)
ax = fp.format_map(ax, lats=sv.lat, lons=sv.lon)
[ax.axhline(lat, color='grey') for lat in [tmp.lat.min(), tmp.lat.max()]]
[ax.axvline(lon, color='grey') for lon in [tmp.lon.min(), tmp.lon.max()]]
plt.show()

# Save out
sv = gc.define_HEMCO_std_attributes(sv, name='StateVector')
sv = gc.define_HEMCO_var_attributes(sv, 'StateVector',
                                    long_name='Clusters generated for analytical inversion',
                                    units='none')
sv.attrs = {'Title' : 'Clusters generated for analytical inversion'}

sv = sv.squeeze(drop=True)
print(sv['StateVector'].shape)
print(sv['lat'])
print(sv['lon'])

# Save out clusters
encoding = {'_FillValue' : None, 'dtype' : 'float64'}
var = {k : encoding for k in sv.keys()}

encoding = {'_FillValue' : None, 'dtype' : 'float64'}
coord = {k : encoding for k in sv.coords}
var.update(coord)
sv.to_netcdf(join(data_dir, 'clusters_permian_long_padded.nc'), encoding=var)
             # unlimited_dims=['time'])


# gc.save_HEMCO_netcdf(sv, data_dir, 'clusters_permian_long_padded.nc',
#                      dtype='float64')