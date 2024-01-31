import numpy as np
import xarray as xr
from copy import deepcopy as dc
import sys
sys.path.append('.')
import settings as s
import gcpy as gc
import format_plots as fp

def solve_inversion(xa, sa, y, so, k, c):
    # Solve the inversion
    kso_inv = k/so
    sa_inv = np.diag(1/sa.reshape(-1,))

    shat = np.linalg.inv(sa_inv + kso_inv.T @ k)
    g = shat @ kso_inv.T
    a = np.identity(len(xa)) - shat @ sa_inv
    xhat = (xa + g @ (y - k @ xa - c))
    gsum = g.sum(axis=1)

    bc_contrib = (g @ c)
    xa_contrib = (a @ xa)
    tot_correct = bc_contrib + xa_contrib
    zeta = bc_contrib/xhat#/tot_correct

    return xhat, np.diag(a), zeta, gsum

def get_gamma(xa, sa, y, so, k, c, tol=1e-1):
    print('Finding gamma...')
    gamma = 10
    gamma_not_found = True
    while gamma_not_found:
        so_gamma = so/gamma
        xhat, _, _, _ = solve_inversion(xa, sa, y, so_gamma, k, c)
        cost = (((xhat - xa)**2)/sa).sum()/len(xa)
        print(f'{gamma:.2f}: {cost:.3f}')
        if np.abs(cost - 1) <= tol:
            gamma_not_found = False
        elif cost > 1:
            gamma /= 2
        elif cost < 1:
            gamma *= 1.5
    print('Gamma found!')
    print('-'*70)

    return gamma


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
    if default_value in clusters.values:  
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

    return data['data'].values

def plot_state(data, clusters_plot, default_value=0, cbar=True, **kw):
    # protect inputs from modification
    data_to_plot = dc(data)

    # Put the state vector layer back on the 2d GEOS-Chem grid
    # matches the data to lat/lon data using a cluster file
    data_to_plot = clusters_1d_to_2d(data_to_plot, clusters_plot, default_value)

    # Plot
    fig, ax, c = plot_state_format(data_to_plot, default_value, cbar, **kw)
    return fig, ax, c

def plot_state_format(data, default_value=0, cbar=True, **kw):
    '''
    Format and plot one layer of the state vector.
    Parameters:
        data (xr.DataArray)     : One layer of the state vector, mapped onto a
                                  2d GEOS-Chem grid using a cluster file. If
                                  your state vector has only one layer,
                                  this may contain your entire state vector.
                                  Dimensions: ('lat','lon')
        default_value (numeric) : The fill value for the array returned.
        cbar (bool)             : Should the function plot a colorbar?
    '''
    # Get kw
    title = kw.pop('title', '')
    kw['cmap'] = kw.get('cmap', 'viridis')
    if 'norm' not in kw:
        kw['vmin'] = kw.get('vmin', data.min())
        kw['vmax'] = kw.get('vmax', data.max())
    kw['add_colorbar'] = False
    cbar_kwargs = kw.pop('cbar_kwargs', {})
    label_kwargs = kw.pop('label_kwargs', {})
    title_kwargs = kw.pop('title_kwargs', {})
    map_kwargs = kw.pop('map_kwargs', {})
    fig_kwargs = kw.pop('fig_kwargs', {})

    # Get figure
    lat_step = np.median(np.diff(data.lat))
    lat_range = [data.lat.min().values - lat_step/2,
                 data.lat.max().values + lat_step/2]
    lon_step = np.median(np.diff(data.lon))
    lon_range = [data.lon.min().values - lon_step/2,
                 data.lon.max().values + lon_step/2]
    fig, ax  = fp.get_figax(maps=True, lats=lat_range, lons=lon_range,
                            **fig_kwargs)

    # Plot data
    # fig, ax = fig_kwargs['figax']
    c = data.plot(ax=ax, snap=True, **kw)

    # Add title and format map
    ax = fp.add_title(ax, title, **title_kwargs)
    # ax = fp.format_map(ax, lat_range, lon_range, **map_kwargs)

    if cbar:
        cbar_title = cbar_kwargs.pop('title', '')
        horiz = cbar_kwargs.pop('horizontal', False)
        cpi = cbar_kwargs.pop('cbar_pad_inches', 0.25)
        if horiz:
            orient = 'horizontal'
            cbar_t_kwargs = {'y' : cbar_kwargs.pop('y', -4)}
        else:
            orient = 'vertical'
            cbar_t_kwargs = {'x' : cbar_kwargs.pop('x', 5)}

        cax = fp.add_cax(fig, ax, horizontal=horiz, cbar_pad_inches=cpi)
        cb = fig.colorbar(c, ax=ax, cax=cax, orientation=orient, **cbar_kwargs)
        cb = fp.format_cbar(cb, cbar_title, horizontal=horiz, **cbar_t_kwargs)
        return fig, ax, cb
    else:
        return fig, ax, c