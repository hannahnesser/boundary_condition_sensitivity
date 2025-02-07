import numpy as np
import xarray as xr
from copy import deepcopy as dc

## -------------------------------------------------------------------------##
## Mapping between vector and grid space
## -------------------------------------------------------------------------##
def clusters_1d_to_2d(data, clusters, default_value=0):
    '''
    Matches inversion data to a cluster file.
    Parameters:
        data (np.array)        : Inversion data. Must have the same length
                                 as the number of clusters, and must be
                                 sorted in ascending order of cluster number
                                 - i.e. [datapoint for cluster 1, datapoint for
                                 cluster 2, datapoint for cluster 3...]
        clusters (xr.Datarray) : 2d array of cluster values for each gridcell.
                                 You can get this directly from a cluster file
                                 used in an analytical inversion.
                                 Dimensions: ('lat','lon')
        default_value (numeric): The fill value for the array returned.
    Returns:
        result (xr.Datarray)   : A 2d array on the GEOS-Chem grid, with
                                 inversion data assigned to each gridcell based
                                 on the cluster file.
                                 Missing data default to the default_value.
                                 Dimensions: same as clusters ('lat','lon').
    '''
    # check that length of data is the same as number of clusters
    clust_list = np.unique(clusters)[np.unique(clusters)!=0] # unique, nonzero clusters
    assert len(data)==len(clust_list), (f'Data length ({len(data)}) is not the same as '
                                        f'the number of clusters ({len(clust_list)}).')

    # build a lookup table from data.
    #    data_lookup[0] = default_value (value for cluster 0),
    #    data_lookup[1] = value for cluster 1, and so forth
    if ~np.any(np.isnan(clusters.values)) or ~np.any(clusters.values == 0):  
        data_lookup = np.append(default_value, data)
    else:
        print('Not using default value.')
        data_lookup = dc(data)

    # use fancy indexing to map data to 2d cluster array
    cluster_index = clusters.squeeze().data.astype(int).tolist()
    result = clusters.copy().squeeze()         # has same shape/dims as clusters
    result.values = data_lookup[cluster_index] # map data to clusters

    return result

def clusters_2d_to_1d(data, clusters, default_value=0):
    '''
    Flattens data on the GEOS-Chem grid, and ensures the resulting order is
    ascending with respect to cluster number.
    Parameters:
        clusters (xr.Datarray) : 2d array of cluster values for each gridcell.
                                 You can get this directly from a cluster file
                                 used in an analytical inversion.
                                 Dimensions: ('lat','lon')
        data (xr.DataArray)    : Data on a 2d GEOS-Chem grid.
                                 Dimensions: ('lat','lon')
   '''
    # Data must be a dataarray
    assert type(data) == xr.core.dataarray.DataArray, \
           "Input data must be a dataarray."

    # Combine clusters and data into one dataarray
    data = data.to_dataset(name='data')
    data['clusters'] = clusters

    # Convert to a dataframe and reset index to remove lat/lon/time
    # dimensions
    data = data.to_dataframe().reset_index()[['data', 'clusters']]

    # Remove non-cluster datapoints
    data = data[data['clusters'] > 0]

    # Fill nans that may result from data and clusters being different
    # shapes
    data = data.fillna(default_value)

    # Sort
    data = data.sort_values(by='clusters')

    return data['data'].values.reshape((-1, 1))

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

def distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    
    All args must be of equal length.    
    
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6378.137 * c
    return km