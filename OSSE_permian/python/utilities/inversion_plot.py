from copy import deepcopy as dc
import numpy as np
from matplotlib import rcParams
from utilities import format_plots as fp
from utilities import grid, utils

_, config = utils.setup()

def plot_state(data, clusters_plot, default_value=0, cbar=True, **kw):
    # protect inputs from modification
    data_to_plot = dc(data)

    # Put the state vector layer back on the 2d GEOS-Chem grid
    # matches the data to lat/lon data using a cluster file
    data_to_plot = grid.clusters_1d_to_2d(data_to_plot, clusters_plot, default_value)

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