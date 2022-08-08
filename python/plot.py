import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np

import sys
sys.path.append('.')
import format_plots as fp
import config
import inversion as inv

## -------------------------------------------------------------------------##
# Define plotting functions
## -------------------------------------------------------------------------##
def plot_inversion(x_a, x_hat, x_true, x_hat_true=None, s_a=None, a=None,
                   optimize_BC=False, figax=None,
                   add_text=True, add_legend=True):
    # Set up plots
    nstate = x_a.shape[0]
    xp = np.arange(1, nstate+1)
    if figax is None:
        fig, ax = format_plot(nstate)
    else:
        fig, ax = figax

    # Subset prior error
    if optimize_BC and (s_a is not None):
        s_a = s_a[:-1, :-1]

    # Plot "true " emissions
    ax = plot_emis(ax, x_true, c=fp.color(2), ls='--', label='Truth')

    # Plot prior
    ax = plot_emis(ax, x_a, s_a, c=fp.color(4), marker='.', markersize=10,
                   label='Prior')

    # Plot posterior
    if x_hat_true is not None:
        ax = plot_emis(ax, x_hat*x_a, c=fp.color(8), marker='.', markersize=5,
                       lw=1, label='Posterior', zorder=10)
        ncol = 2
    else: # if x_hat_true is none
        x_hat_true = x_hat
        ncol = 3

    ax = plot_emis(ax, x_hat_true*x_a, c=fp.color(6), marker='*',
               markersize=10, label='True BC Posterior')

    if add_text:
        # Add text on whether the boundary condition is otpimized or not
        add_text_label(ax, optimize_BC)

    if add_legend:
        ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.45),
                           loc='upper center', ncol=ncol)
    ax = fp.add_labels(ax, 'State vector element', 'Emissions\n(ppb/day)')

    return fig, ax

def plot_emis(ax, emis, err=None, **kwargs):
    nstate = emis.shape[0]
    xp = np.arange(1, nstate+1)
    if err is None:
        ax.plot(xp, 3600*24*emis, **kwargs)
    else:
        ax.errorbar(xp, 3600*24*emis, yerr=3600*24*np.diag(err)**0.5*emis,
                    **kwargs)
    ax.set_ylim(0, 200)
    return ax


def plot_avker(ax, a, **kwargs):
    nstate = a.shape[0]
    xp = np.arange(1, nstate+1)

    # Plot avker sensitivities
    kwargs['c'] = kwargs.pop('c', 'grey')
    kwargs['lw'] = kwargs.pop('lw', 1)
    kwargs['ls'] = kwargs.pop('ls', '-')
    ax.plot(xp, np.diag(a), **kwargs)
    # for i, row in enumerate(a):
    #     ax.plot(xp, row, c=fp.color(i*2, lut=a.shape[0]*2))

    # Set limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    return ax

def plot_obs(nstate, y, y_a, y_ss, obs_t, optimize_BC):
    # Plot observations
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    nt = len(obs_t)
    ax.plot(xp, y_ss, c='black', lw=2, label='Steady State', zorder=10)
    ax.plot(xp, y, c='grey', lw=0.5, ls=':', markersize=10,
            label='Observed', zorder=9)
    for i, y_a_column in enumerate(y_a.T):
        if (i == 0) or (i == (nt - 1)):
            ax.plot(xp, y_a_column,
                    c=fp.color(i+2, cmap='plasma_r', lut=nt + 2),
                    lw=0.5, ls='-',
                    label=f'Modeled (t={(obs_t[i]/3600):.1f} hrs)')
        else:
            ax.plot(xp, y_a_column,
                    c=fp.color(i+2, cmap='plasma_r', lut=nt + 2),
                    lw=0.5, ls='-')

    # Aesthetics
    add_text_label(ax, optimize_BC)
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.45), loc='upper center',
                       ncol=2)
    ax = fp.add_labels(ax, 'State Vector Element', 'XCH4 (ppb)')
    ax.set_ylim(y_ss.min()-50, y_ss.max()+50)

    return fig, ax

def plot_obs_diff(nstate, y, y_hat, y_a, obs_t, optimize_BC):
    # Plot observations
    fig, ax = format_plot(nstate)

    xp = np.arange(1, nstate+1)
    nt = len(obs_t)
    ax.axhline(0, ls='--', lw=1, color='grey')
    y_diff = y - y_a
    for i, y_column in enumerate(y_diff.T):
        if (i == 0) or (i == (nt - 1)):
            ax.plot(xp, y_column,
                    c=fp.color(i+2, cmap='plasma', lut=nt + 2),
                    lw=0.5, ls='-',
                    label=f'Difference (t={(obs_t[i]/3600):.1f} hrs)')
        else:
            ax.plot(xp, y_column,
                    c=fp.color(i+2, cmap='plasma', lut=nt + 2),
                    lw=0.5, ls='-')

    y_diff = y - y_hat
    for i, y_column in enumerate(y_diff.T):
        ax.plot(xp, y_column,
                c=fp.color(i+2, cmap='viridis', lut=nt + 2),
                lw=0.5, ls='--')

    # Aesthetics
    add_text_label(ax, optimize_BC)
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.45), loc='upper center',
                       ncol=2)
    ax = fp.add_labels(ax, 'State Vector Element', 'XCH4 (ppb)')
    ax.set_ylim(-200, 200)

    return fig, ax

def plot_cost_func(x_hat, x_a, s_a_vec, y_hat, y, s_o_vec, obs_t,
                   optimize_BC):
    # Plot observations
    nstate = len(x_a)
    nobs = len(s_o_vec)
    fig, ax = format_plot(nstate)
    # ax2 = ax.twinx()

    xp = np.arange(1, nstate+1)
    nt = len(obs_t)

    # Plot x component
    x_diff = (x_hat - np.ones(nstate))**2/s_a_vec
    ax.plot(xp, x_diff, c=fp.color(5), lw=2, ls='-', label=r'$J_A(\hat{x})$')
    #, color=fp.color(5))
    # ax.tick_params(axis='y', labelcolor=fp.color(5))
    # ax.set_ylim(0, 50) # x_diff.max()*1.1)

    # Plot y component
    y_diff = ((y - y_hat)**2/s_o_vec.reshape(y.shape))#.sum(axis=1)
    ax.plot(xp, y_diff.sum(axis=1), c=fp.color(2), lw=2, ls='-',
            label=r'$J_O(\hat{x})$')
    # for i, y_column in enumerate(y_diff.T):
    #     ax.plot(xp, y_column, c=fp.color(2), lw=0.5, ls='--')


    # y_diff = (y - y_a)**2/s_o_vec.reshape(y.shape)/nobs
    # ax2.plot(xp, y_diff.sum(axis=1), c=fp.color(2), lw=2, ls='--')

    # ax2 = fp.add_labels(ax2, '', r'$J_O(\hat{x})$', color=fp.color(2))
    # ax2.tick_params(axis='y', labelcolor=fp.color(2))
    # ax2.set_ylim(0, y_diff.sum(axis=1).max()*1.1)

    # Aesthetics
    add_text_label(ax, optimize_BC)
    ax = fp.add_labels(ax, '', r'$J(\hat{x})$')
    ax.set_xlabel('State Vector Element',
                  fontsize=config.LABEL_FONTSIZE*config.SCALE,
                  labelpad=config.LABEL_PAD, color='black')
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.45), loc='upper center',
                       ncol=2)
    ax.set_ylim(0, 30)
    # ax.set_ylim(0, 5)

    return fig, ax

def format_plot(nstate, nplots=1, **fig_kwargs):
    fig, ax = fp.get_figax(aspect=4 - 0.65*(nplots-1), cols=1, rows=nplots,
                           **fig_kwargs)
    if nplots == 1:
        ax = [ax]
    for axis in ax:
        for i in range(nstate+2):
            axis.axvline(i-0.5, c=fp.color(1), alpha=0.2, ls=':')
        axis.set_xticks(np.arange(0, nstate+1, 2))
        axis.set_xlim(0.5, nstate+0.5)
        axis.set_facecolor('white')
    if nplots == 1:
        return fig, ax[0]
    else:
        return fig, ax

def add_text_label(ax, optimize_BC):
    if optimize_BC:
        txt = 'BC optimized'
    else:
        txt = 'BC not optimized'
    # txt = txt + f'\nn = {nstate}\nm = {nobs}\nU = {(U*3600)}'
    ax.text(0.98, 0.95, txt, ha='right', va='top',
                 fontsize=config.LABEL_FONTSIZE*config.SCALE,
                 transform=ax.transAxes)

## -------------------------------------------------------------------------##
# Define Permian plotting functions
## -------------------------------------------------------------------------##
## -------------------------------------------------------------------------##
def plot_state(data, clusters_plot, default_value=0, cbar=True,
               category=None, time=None, category_list=None,
               time_list=None, cluster_list=None, **kw):
    '''
    Plots a state vector element.
    Parameters:
        data (np.array)         : Inversion data.
                                  Dimensions: nstate
        clusters (xr.Datarray)  : 2d array of cluster values for each gridcell.
                                  You can get this directly from a cluster file
                                  used in an analytical inversion.
                                  Dimensions: ('lat','lon')
    Optional Parameters:
        default_value (numeric) : The fill value for the array returned.
        cbar (bool)             : Should the function a colorbar?
        category (string)       : The category you would like to extract. Must match with
                                  element(s) in in category_list.
        time (string)           : The time you would like to extract. Must match with
                                  element(s) in time_list. If there is no time, use None.
        category_list (list)    : The category labels for each element of the state vector.
                                  Dimensions: nstate
        time_list (list)        : The time labels for each element of the state vector.
                                  Dimensions: nstate
        cluster_list (list)     : Cluster numbers for each element of the state vector.
                                  If this option is not included, the data must be
                                  in ascending order of cluster number.
                                  Dimensions: nstate.

    Returns:
        fig, ax, c: Figure, axis, and colorbar for an mpl plot.
    '''
    # protect inputs from modification
    data_to_plot = np.copy(data)

    # Select one "layer" at a time
    # each "layer" corresponds to one "2d cluster file"
    # if you only have one layer in your dataset, you can skip this.
    if ((category is not None) or (time is not None)
        or (category_list is not None) or (time_list is not None)
        or (cluster_list is not None) ):
        data_to_plot = get_one_statevec_layer(data_to_plot, category=category, time=time,
                                              category_list=category_list,
                                              time_list=time_list,
                                              cluster_list=cluster_list)

    # Put the state vector layer back on the 2d GEOS-Chem grid
    # matches the data to lat/lon data using a cluster file
    data_to_plot = inv.match_data_to_clusters(data_to_plot, clusters_plot, default_value)

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
    kw['vmin'] = kw.get('vmin', data.min())
    kw['vmax'] = kw.get('vmax', data.max())
    kw['add_colorbar'] = False
    cbar_kwargs = kw.pop('cbar_kwargs', {})
    label_kwargs = kw.pop('label_kwargs', {})
    title_kwargs = kw.pop('title_kwargs', {})
    map_kwargs = kw.pop('map_kwargs', {})
    fig_kwargs = kw.pop('fig_kwargs', {})

    # Get figure
    lat_range = [data.lat.min(), data.lat.max()]
    lon_range = [data.lon.min(), data.lon.max()]
    fig, ax  = fp.get_figax(maps=True, lats=lat_range, lons=lon_range,
                            **fig_kwargs)

    # Plot data
    c = data.plot(ax=ax, snap=True, **kw)

    # Set limits
    ax.set_xlim(lon_range)
    ax.set_ylim(lat_range)

    # Add title and format map
    ax = fp.add_title(ax, title, **title_kwargs)
    ax = fp.format_map(ax, data.lat, data.lon, **map_kwargs)

    if cbar:
        cbar_title = cbar_kwargs.pop('title', '')
        cax = fp.add_cax(fig, ax)
        cb = fig.colorbar(c, ax=ax, cax=cax, **cbar_kwargs)
        cb = fp.format_cbar(cb, cbar_title)
        return fig, ax, cb
    else:
        return fig, ax, c

