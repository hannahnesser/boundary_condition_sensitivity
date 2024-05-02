'''
This is a package designed to deal with common GEOS-Chem
processing needs. It also contains several generic functions that
are used in the course of working with GEOS-Chem data.
'''
import pandas as pd
import xarray as xr
import pickle
from os import listdir
import warnings


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