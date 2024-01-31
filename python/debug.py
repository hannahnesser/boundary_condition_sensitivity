import xarray as xr
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
import format_plots as fp

# Define lat and lon limits
imi_lons = [-107.5, -97.8125]
imi_lats = [27.75, 35.75]

permian_lons = [-111, -95]
permian_lats = [24, 39]

# Get data
file_name = '../data/S5P_RPRO_L2__CH4____20200531T223349_20200601T001519_13642_03_020400_20221126T164255.nc'

trop = xr.open_dataset(file_name, group='PRODUCT')

data = {}

data['methane'] = trop['methane_mixing_ratio_bias_corrected'].values[0, :, :]
data['qa_value'] = trop['qa_value'].values[0, :, :]
data['longitude'] = trop['longitude'].values[0, :, :]
data['latitude'] = trop['latitude'].values[0, :, :]

# Plot
fig, ax = fp.get_figax(maps=True, lats=permian_lats, lons=permian_lons)
ax.scatter(data['longitude'], data['latitude'], c=data['methane'])
[ax.axhline(lat, color='grey', ls='--') for lat in imi_lats]
[ax.axvline(lon, color='grey', ls='--') for lon in imi_lons]
ax = fp.format_map(ax, lats=permian_lats, lons=permian_lons)

plt.show()

print(data)