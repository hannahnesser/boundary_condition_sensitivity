'''
This is a package designed to deal with common GEOS-Chem
processing needs. It also contains several generic functions that
are used in the course of working with GEOS-Chem data.
'''
import numpy as np
import pandas as pd
import xarray as xr
import pickle
from scipy.stats import linregress

from os.path import join
from os import listdir
import os
import warnings

# Plotting
from matplotlib import rcParams

# Import information for plotting in a consistent fashion
from utilities import plot_settings as ps

# Other font details
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = ps.LABEL_FONTSIZE*ps.SCALE
rcParams['text.usetex'] = True
# rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = 0

## -------------------------------------------------------------------------##
## Loading functions
## -------------------------------------------------------------------------##
def file_exists(file_name):
    '''
    Check for the existence of a file
    '''
    data_dir = file_name.rpartition('/')[0]
    if file_name.split('/')[-1] in listdir(data_dir):
        return True
    else:
        print(f'{file_name} is not in the data directory.')
        return False

def save_obj(obj, name):
        with open(name , 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    # Open a generic file using pickle
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def read_file(*file_names, **kwargs):
    file_suffix = file_names[0].split('.')[-1]
    # Require that the file exists
    for f in file_names:
        # Require that the files exist and that all files are the
        # same type
        assert file_exists(f), f'{f} does not exist.'
        assert f.split('.')[-1] == file_suffix, \
               'Variable file types provided.'

        # If multiple files are provided, require that they are netcdfs
        if len(file_names) > 1:
            assert file_suffix[:2] == 'nc', \
                   'Multiple files are provided that are not netcdfs.'

    # If a netcdf, read it using xarray
    if file_suffix[:2] == 'nc':
        file = read_netcdf_file(*file_names, **kwargs)
    # Else, read it using a generic function
    else:
        if 'chunks' in kwargs:
            warnings.warn('NOTE: Chunk sizes were provided, but the file is not a netcdf. Chunk size is ignored.', stacklevel=2)
        file = read_generic_file(*file_names, **kwargs)

    return file

def read_generic_file(file_name):
    # Open a generic file using pickle
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def read_netcdf_file(*file_names, **kwargs):
    # Open a dataset
    if len(file_names) > 1:
        # Currently assumes that we are stacking the files
        # vertically
        if 'concat_dim' in kwargs:
            if len(kwargs['concat_dim']) > 1:
                file_names = [[f] for f in file_names]
        data = xr.open_mfdataset(file_names, **kwargs)
    else:
        if 'dims' in kwargs:
            del kwargs['dims']
        data = xr.open_dataset(file_names[0], **kwargs)

    # If there is only one variable, convert to a dataarray
    variables = list(data.keys())
    if len(variables) == 1:
        data = data[variables[0]]

    # Return the file
    return data

def calculate_chunk_size(available_memory_GB, n_threads=None,
                         dtype='float32'):
    '''
    This function returns a number that gives the total number of
    elements that should be held in a chunk. It does not specify the exact
    chunks for specific dimensions.
    '''

    # Get the number of active threads
    if n_threads is None:
        n_threads = int(os.environ['OMP_NUM_THREADS'])

    # Approximate the number of chunks that are held in memory simultaneously
    # by dask (reference: https://docs.dask.org/en/latest/array-best-practices.html#:~:text=Orient%20your%20chunks,-When%20reading%20data&text=If%20your%20Dask%20array%20chunks,closer%20to%201MB%20than%20100MB.)
    chunks_in_memory = 20*n_threads

    # Calculate the memory that is available per chunk (in GB)
    mem_per_chunk = available_memory_GB/chunks_in_memory

    # Define the number of bytes required for each element
    if dtype == 'float32':
        bytes_per_element = 4
    elif dtype == 'float64':
        bytes_per_element = 8
    else:
        print('Data type is not recognized. Defaulting to reserving 8 bytes')
        print('per element.')
        bytes_per_element = 8

    # Calculate the number of elements that can be held in the available
    # memory for each chunk
    number_of_elements = mem_per_chunk*1e9/bytes_per_element

    # Scale the number of elements down by 10% to allow for wiggle room.
    return int(0.9*number_of_elements)

## -------------------------------------------------------------------------##
## Statistics functions
## -------------------------------------------------------------------------##
def rmse(diff):
    return np.sqrt(np.mean(diff**2))

def add_quad(data):
    return np.sqrt((data**2).sum())

def group_data(data, groupby, quantity='DIFF',
                stats=['count', 'mean', 'std', rmse]):
    return data.groupby(groupby).agg(stats)[quantity].reset_index()

def comparison_stats(xdata, ydata):
    m, b, r, p, err = linregress(xdata.flatten(), ydata.flatten())
    bias = (ydata - xdata).mean()
    return m, b, r, bias

def rma_modified(xdata, ydata):
    m, _, _, _ = np.linalg.lstsq(xdata.reshape((-1, 1)),
                                 ydata.reshape((-1, 1)), rcond=None)
    slope = np.sign(m[0][0])*ydata.std()/xdata.std()
    return slope

def rma(xdata, ydata):
    _, _, r, _ = comparison_stats(xdata, ydata)
    slope = np.sign(r)*ydata.std()/xdata.std()
    return slope


    # if relative:
    #     y = ydata/y.max()
    #     x = xdata/x.max()
    # else:
    #     y = (ydata - ydata.min())/(ydata.max() - ydata.min())
    #     x = (xdata - xdata.min())/(xdata.max() - xdata.min())

def rel_err(data, truth):
    return (data - truth)/truth

## -------------------------------------------------------------------------##
## Grid functions
## -------------------------------------------------------------------------##
def adjust_grid_bounds(lat_min, lat_max, lat_delta,
                       lon_min, lon_max, lon_delta,
                       buffer=[0, 0, 0, 0]):
    '''
    This function adjusts the default GEOS-Chem grid bounds,
    which are given as grid box centers, to grid box edges.
    It also allows for the option to remove buffer grid cells,
    although the default is to assume 0 buffer grid cells.
    (Buffer grid cells should be expressed in the standard
    GEOS-Chem convention of [N S E W].)
    '''
    lat_min = lat_min + lat_delta*buffer[1] - lat_delta/2
    lat_max = lat_max - lat_delta*buffer[0] + lat_delta/2
    lon_min = lon_min + lon_delta*buffer[3] - lon_delta/2
    lon_max = lon_max - lon_delta*buffer[2] + lon_delta/2
    return [lat_min, lat_max], [lon_min, lon_max]

def subset_data_latlon(data, lat_min, lat_max, lon_min, lon_max):
    '''
    This function subsets a given dataset (in xarray form, with
    latitude and longitude variables lat and lon, respectively)
    to a given lat lon grid.
    '''
    data = data.where((data.lat > lat_min) & (data.lat < lat_max) &
                      (data.lon > lon_min) & (data.lon < lon_max),
                      drop=True)
    return data

def create_gc_grid(lat_min, lat_max, lat_delta,
                   lon_min, lon_max, lon_delta,
                   centers=True, return_xarray=True):
    '''
    This function creates a grid with values corresponding to the
    centers of each grid cell. The latitude and longitude limits
    provided correspond to grid cell centers if centers=True and
    edges otherwise.
    '''
    if not centers:
        lat_min += lat_delta/2
        lat_max -= lat_delta/2
        lon_min += lon_delta/2
        lon_max -= lon_delta/2

    lats = np.arange(lat_min, lat_max + lat_delta, lat_delta)
    lons = np.arange(lon_min, lon_max + lon_delta, lon_delta)

    if return_xarray:
        data = xr.DataArray(np.zeros((len(lats), len(lons))),
                            coords=[lats, lons],
                            dims=['lats', 'lons'])
    else:
        data = [lats, lons]

    return data

def nearest_loc(data, compare_data):
    indices = np.abs(compare_data.reshape(-1, 1) -
                     data.reshape(1, -1)).argmin(axis=0)
    return indices

## -------------------------------------------------------------------------##
## GEOS-Chem correction functions
## -------------------------------------------------------------------------##
def fill_GC_first_hour(data):
    '''
    This function fills in the first hour of GEOS-Chem output data
    because of GEOS-Chem's failure to output data during the first time
    segment of every simulation. What an odd bug.
    '''
    data0 = data.where(data.time == data.time[0], drop=True)

    # Change the time of that data to the 0th hour
    t = pd.Timestamp(data0.time.values[0]).replace(hour=0).to_datetime64()
    data0.time.values[0] = t

    # Merge the datasets
    data = xr.concat([data0, data], dim='time')

    return data

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
