import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np
import math

import sys
sys.path.append('.')
import format_plots as fp
import plot_settings as ps
import inversion as inv

## -------------------------------------------------------------------------##
# Define plotting functions
## -------------------------------------------------------------------------##
def plot_inversion(xa, x_hat, x_true, xhat_true=None, sa=None, a=None,
                   optimize_BC=False, figax=None,
                   add_text=True, add_legend=True):
    # Set up plots
    nstate = xa.shape[0]
    xp = np.arange(1, nstate+1)
    if figax is None:
        fig, ax = format_plot(nstate)
    else:
        fig, ax = figax

    # Subset prior error
    if optimize_BC and (sa is not None):
        sa = sa[:-1, :-1]

    # Plot "true " emissions
    ax = plot_emis(ax, x_true, c=fp.color(2), ls='--', label='Truth')

    # Plot prior
    ax = plot_emis(ax, xa, sa, c=fp.color(4), marker='.', markersize=10,
                   label='Prior')

    # Plot posterior
    if xhat_true is not None:
        ax = plot_emis(ax, x_hat*xa, c=fp.color(8), marker='.', markersize=5,
                       lw=1, label='Posterior', zorder=10)
        ncol = 2
    else: # if xhat_true is none
        xhat_true = x_hat
        ncol = 3

    ax = plot_emis(ax, xhat_true*xa, c=fp.color(6), marker='*',
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
        ax.plot(xp, emis, **kwargs)
    else:
        ax.errorbar(xp, emis, yerr=np.diag(err)**0.5*emis,
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

def plot_obs_diff(nstate, y, yhat, y_a, obs_t, optimize_BC):
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

    y_diff = y - yhat
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

def plot_cost_func(x_hat, xa, sa_vec, yhat, y, s_o_vec, obs_t,
                   optimize_BC):
    # Plot observations
    nstate = len(xa)
    nobs = len(s_o_vec)
    fig, ax = format_plot(nstate)
    # ax2 = ax.twinx()

    xp = np.arange(1, nstate+1)
    nt = len(obs_t)

    # Plot x component
    x_diff = (x_hat - np.ones(nstate))**2/sa_vec
    ax.plot(xp, x_diff, c=fp.color(5), lw=2, ls='-', label=r'$J_A(\hat{x})$')
    #, color=fp.color(5))
    # ax.tick_params(axis='y', labelcolor=fp.color(5))
    # ax.set_ylim(0, 50) # x_diff.max()*1.1)

    # Plot y component
    y_diff = ((y - yhat)**2/s_o_vec.reshape(y.shape))#.sum(axis=1)
    ax.plot(xp, y_diff.sum(axis=1), c=fp.color(2), lw=2, ls='-',
            label=r'$J_O(\hat{x})$')

    # Aesthetics
    add_text_label(ax, optimize_BC)
    ax = fp.add_labels(ax, '', r'$J(\hat{x})$')
    ax.set_xlabel('State Vector Element',
                  fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
                  labelpad=ps.LABEL_PAD, color='black')
    ax = fp.add_legend(ax, bbox_to_anchor=(0.5, -0.45), loc='upper center',
                       ncol=2)
    ax.set_ylim(0, 30)
    # ax.set_ylim(0, 5)

    return fig, ax

def format_plot(fig, ax, nstate, **fig_kwargs):
    # Deal with difference in axis shapes
    if type(ax) == np.ndarray:
        if len(ax.shape) == 1:
            ncols = 1
        else:
            ncols = ax.shape[1]
    else:
        ncols = 1
        ax = [ax]

    # Formatting
    for axis in ax:
        for i in range(nstate+2):
            # ncols = 1 --> lw = 1      1 - 0*0.25
            # ncols = 2 --> lw = 0.75   1 - math.log2(2)*0.25
            # ncols = 4 --> lw = 0.5    1 - math.log2(4)*0.25
            # ncols = 8 --> lw = 0.25   1 - math.log2(8)*0.25
            axis.axvline(i-0.5, c=fp.color(1), alpha=0.2, ls=':', 
                         lw=1-math.log2(ncols)*0.25)
        axis.set_xticks(np.arange(0, nstate+1, 2))
        axis.set_xlim(0.5, nstate+0.5)
        axis.set_facecolor('white')

        # Adjust aspect
        xs = axis.get_xlim()
        ys = axis.get_ylim()
        axis.set_aspect(0.25*(xs[1]-xs[0])/(ys[1]-ys[0]), adjustable='box')
    if len(ax) == 1:
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
                 fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
                 transform=ax.transAxes)

## -------------------------------------------------------------------------##
# Define Permian plotting functions
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


## -------------------------------------------------------------------------##
## Plotting functions : comparison
## -------------------------------------------------------------------------##
def add_stats_text(ax, r, bias):
    if r**2 <= 0.99:
        ax.text(0.05, 0.9, r'R = %.2f' % r,
                fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
                transform=ax.transAxes)
    else:
        ax.text(0.05, 0.9, r'R $>$ 0.99',
                fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
                transform=ax.transAxes)
    ax.text(0.05, 0.875, 'Bias = %.2f' % bias,
            fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
            transform=ax.transAxes,
            va='top')
    return ax

def plot_comparison_hexbin(xdata, ydata, cbar, stats, **kw):
    cbar_kwargs = kw.pop('cbar_kwargs', {})
    fig_kwargs = kw.pop('fig_kwargs', {})
    lims = kw.pop('lims', None)
    fig, ax = fp.get_figax(**fig_kwargs)
    ax.set_aspect('equal')

    # Get data limits
    xlim, ylim, xy, dmin, dmax = fp.get_square_limits(xdata, ydata, lims=lims)

    # Set bins and gridsize for hexbin
    if ('vmin' not in kw) or ('vmax' not in kw):
        bin_max = len(xdata)*0.1
        round_by = len(str(len(xdata)/10).split('.')[0]) - 1
        bin_max = 1+int(round(bin_max, -round_by))
        kw['bins'] = np.arange(0, bin_max)
    kw['gridsize'] = math.floor((dmax - dmin)/(xy[1] - xy[0])*40)

    # Plot hexbin
    c = ax.hexbin(xdata, ydata, cmap=fp.cmap_trans('plasma_r'), **kw)

    # Aesthetics
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Print information about R2 on the plot
    if stats:
        _, _, r, bias = comparison_stats(xdata, ydata)
        ax = add_stats_text(ax, r, bias)

    if cbar:
        cbar_title = cbar_kwargs.pop('title', '')
        cax = fp.add_cax(fig, ax)
        cbar = fig.colorbar(c, cax=cax, **cbar_kwargs)
        cbar = fp.format_cbar(cbar, cbar_title)
        return fig, ax, cbar
    else:
        return fig, ax, c

def plot_comparison_scatter(xdata, ydata, stats, **kw):
    fig_kwargs = kw.pop('fig_kwargs', {})
    lims = kw.pop('lims', None)

    fig, ax = fp.get_figax(**fig_kwargs)
    ax.set_aspect('equal')

    # Get data limits
    xlim, ylim, xy, dmin, dmax = fp.get_square_limits(xdata, ydata, lims=lims)

    # Plot hexbin
    kw['color'] = kw.pop('color', fp.color(4))
    kw['s'] = kw.pop('s', 3)
    c = ax.scatter(xdata, ydata, **kw)

    # Aesthetics
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Print information about R2 on the plot
    if stats:
        _, _, r, bias = comparison_stats(xdata, ydata)
        ax = add_stats_text(ax, r, bias)

    return fig, ax, c

def plot_comparison_dict(xdata, ydata, **kw):
    fig_kwargs = kw.pop('fig_kwargs', {})
    fig, ax = fp.get_figax(**fig_kwargs)
    ax.set_aspect('equal')

    # We need to know how many data sets were passed
    n = len(ydata)
    cmap = kw.pop('cmap', 'inferno')

    # Plot data
    count = 0
    for k, ydata in ydata.items():
        ax.scatter(xdata, ydata, alpha=0.5, s=5*ps.SCALE,
                   c=color(count, cmap=cmap, lut=n))
        count += 1

    # Color bar (always True)
    cax = fp.add_cax(fig, ax)
    cbar_ticklabels = kw.pop('cbar_ticklabels', list(ydata.keys()))
    norm = colors.Normalize(vmin=0, vmax=n)
    cbar = colorbar.ColorbarBase(cax, cmap=plt.cm.get_cmap(cmap, lut=n),
                                 norm=norm)
    cbar.set_ticks(0.5 + np.arange(0,n+1))
    cbar.set_ticklabels(cbar_ticklabels)
    cbar = format_cbar(cbar)

    return fig, ax, cbar

def plot_comparison(xdata, ydata, cbar=True, stats=True, hexbin=True, **kw):
    # Get other plot labels
    xlabel = kw.pop('xlabel', '')
    ylabel = kw.pop('ylabel', '')
    label_kwargs = kw.pop('label_kwargs', {})
    title = kw.pop('title', '')
    title_kwargs = kw.pop('title_kwargs', {})

    if type(ydata) == dict:
        fig, ax, c = plot_comparison_dict(xdata, ydata, **kw)
    elif hexbin:
        fig, ax, c = plot_comparison_hexbin(xdata, ydata, cbar, stats, **kw)
    else:
        fig, ax, c = plot_comparison_scatter(xdata, ydata, stats, **kw)

    # Aesthetics
    ax = plot_one_to_one(ax)
    ax = fp.add_labels(ax, xlabel, ylabel, **label_kwargs)
    ax = fp.add_title(ax, title, **title_kwargs)

    # Make sure we have the same ticks
    # ax.set_yticks(ax.get_xticks(minor=False), minor=False)
    ax.set_xticks(ax.get_yticks(minor=False), minor=False)
    ax.set_xlim(ax.get_ylim())

    return fig, ax, c


def plot_one_to_one(ax):
    xlim, ylim, _, _, _ = get_square_limits(ax.get_xlim(),
                                            ax.get_ylim())
    ax.plot(xlim, xlim, c='0.1', lw=2, ls=':',
            alpha=0.5, zorder=0)
    return ax

