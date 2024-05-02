import xarray as xr
from os.path import join

## -------------------------------------------------------------------------##
## HEMCO input functions
## -------------------------------------------------------------------------##
def define_HEMCO_std_attributes(data, name=None):
    '''
    This function defines the attributes for time, lat, and lon,
    the standard GEOS-Chem dimensions. It currently doesn't have the
    capacity to define level attributes.
    '''
    print('Remember to define the following attributes for non-standard')
    print('variables:')
    print(' - title (global)')
    print(' - long_name')
    print(' - units')

    # Check if time is in the dataset and, if not, add it
    if 'time' not in data.coords:
        data = data.assign_coords(time=0)
        data = data.expand_dims('time')

    # Convert to dataset
    if type(data) != xr.core.dataset.Dataset:
        assert name is not None, 'Name is not provided for dataset.'
        data = data.to_dataset(name=name)

    # Set time, lat, and lon attributes
    data.time.attrs = {'long_name' : 'Time',
                       'units' : 'hours since 2009-01-01 00:00:00',
                       'calendar' : 'standard'}
    data.lat.attrs = {'long_name': 'latitude', 'units': 'degrees_north'}
    data.lon.attrs = {'long_name': 'longitude', 'units': 'degrees_east'}
    return data

def define_HEMCO_var_attributes(data, var, long_name, units):
    data[var].attrs = {'long_name' : long_name, 'units' : units}
    return data

def save_HEMCO_netcdf(data, data_dir, file_name, dtype='float32', **kwargs):
    encoding = {'_FillValue' : None, 'dtype' : dtype}
    var = {k : encoding for k in data.keys()}
    coord = {k : encoding for k in data.coords}
    var.update(coord)
    data.to_netcdf(join(data_dir, file_name), encoding=var,
                   unlimited_dims=['time'], **kwargs)
