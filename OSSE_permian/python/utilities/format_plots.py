import numpy as np
import math
from scipy.ndimage import zoom
from collections import OrderedDict
from os.path import join

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cf

from utilities import utils

_, config = utils.setup()

# Other font details
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = config['label_size']*config['scale']
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = config['title_pad']

from matplotlib.font_manager import findfont, FontProperties
font = findfont(FontProperties(family=['sans-serif']))

def color(k, cmap='CMRmap', lut=10):
    c = plt.cm.get_cmap(cmap, lut=lut)
    return colors.to_hex(c(k))

def cmap_from_color(color_high, color_low=(1, 1, 1), N=100):
    rgb_map = [color_low, colors.to_rgb(color_high)]
    cmap = colors.LinearSegmentedColormap.from_list('cmap', rgb_map, N=N)
    return cmap

def cmap_trans(cmap, ncolors=300, nalpha=20, set_bad=None, reverse=False):
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.append(np.linspace(0.0, 1.0, nalpha),
                                  np.ones(ncolors-nalpha))
    
    if reverse:
        color_array = color_array[::-1, :]

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(
        name=str(cmap) + '_trans', colors=color_array)

    if set_bad is not None:
        map_object.set_bad(color=set_bad)

    return map_object

def cmap_trans_center(cmap, ncolors=300, nalpha=20):
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    half_l = math.floor((ncolors - nalpha)/2)
    half_r = math.ceil((ncolors - nalpha)/2)
    color_array[:,-1] = np.concatenate((np.ones(half_l),
                                        np.linspace(1, 0, int(nalpha/2)),
                                        np.linspace(0, 1, int(nalpha/2)),
                                        np.ones(half_r)))

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=str(cmap) + '_ctrans' ,
                                                   colors=color_array)

    return map_object

def get_figsize(aspect, rows, cols, **fig_kwargs):
    # Set default kwarg values
    max_width = fig_kwargs.get('max_width', config['width']*config['scale'])*cols
    max_height = fig_kwargs.get('max_height', config['height']*config['scale'])*rows

    # Get figsize
    if aspect > 1: # width > height
        figsize = (max_width,
                   max_width/aspect)
    else: # width < height
        figsize = (max_height*aspect,
                   max_height)
    return figsize

def get_aspect(rows, cols, aspect=None,
               maps=False, lats=None, lons=None):
    if maps:
        aspect = np.cos(np.mean([np.min(lats), np.max(lats)])*np.pi/180)
        xsize = np.ptp([np.max(lons), np.min(lons)])*aspect
        ysize = np.ptp([np.max(lats), np.min(lats)])
        aspect = xsize/ysize
    return aspect*cols/rows

def make_axes(rows=1, cols=1, aspect=None,
              maps=False, lats=None, lons=None,
              **fig_kwargs):
    aspect = get_aspect(rows, cols, aspect, maps, lats, lons)
    figsize = get_figsize(aspect, rows, cols, **fig_kwargs)
    kw = {}
    if maps:
        kw['subplot_kw'] = {'projection' : ccrs.PlateCarree()}
    kw['sharex'] = fig_kwargs.pop('sharex', True)
    kw['sharey'] = fig_kwargs.pop('sharey', False)
    # kw['width_ratios'] = fig_kwargs.pop('width_ratios', None)
    # if (rows + cols) > 2:
    #     kw['constrained_layout'] = True
        # figsize = tuple(f*1.5 for f in figsize)
    fig, ax = plt.subplots(rows, cols, figsize=figsize, **kw)
    # plt.subplots_adjust(right=1)
    return fig, ax

def add_cax(fig, ax, cbar_pad_inches=0.25, horizontal=False):
    # should be updated to infer cbar width and cbar_pad_inches
    if not horizontal:
        try:
            axis = ax[-1, -1]
            height = ax[0, -1].get_position().y1 - ax[-1, -1].get_position().y0
            ax_width = ax[0, -1].get_position().x1 - ax[0, 0].get_position().x0
        except IndexError:
            axis = ax[-1]
            # height = ax[-1].get_position().height
            height = ax[0].get_position().y1 - ax[-1].get_position().y0
            ax_width = ax[-1].get_position().x1 - ax[0].get_position().x0
        except TypeError:
            axis = ax
            height = ax.get_position().height
            ax_width = ax.get_position().width

        # x0
        fig_width = fig.get_size_inches()[0]
        x0_init = axis.get_position().x1
        x0 = (fig_width*x0_init + cbar_pad_inches*config['scale'])/fig_width

        # y0
        y0 = axis.get_position().y0

        # Width
        width = 0.1*config['scale']/fig_width
    else:
        try:
            axis = ax[-1, 0]
            width = ax[-1, -1].get_position().x1 - ax[-1, 0].get_position().x0
        except IndexError:
            axis = ax[0]
            width = ax[-1].get_position().x1 - ax[0].get_position().x0
        except TypeError:
            axis = ax
            width = ax.get_position().width

        # x0
        x0 = axis.get_position().x0

        # y0
        fig_height = fig.get_size_inches()[1]
        y0_init = axis.get_position().y0
        y0 = (fig_height*y0_init - cbar_pad_inches*config['scale'])/fig_height

        # Height
        height = 0.1*config['scale']/fig_height

    # Make axis
    cax = fig.add_axes([x0, y0, width, height])

    # if return_coords:
    #     return cax, x0, y0, width, height
    # else:
    return cax

def get_figax(rows=1, cols=1, aspect=4,
              maps=False, lats=None, lons=None,
              figax=None, **fig_kwargs):
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = make_axes(rows, cols, aspect, maps, lats, lons, **fig_kwargs)

        if (rows > 1) or (cols > 1):
            for axis in ax.flatten():
                axis.set_facecolor('0.98')
                if maps:
                    axis = format_map(axis, lats=lats, lons=lons)
            # plt.subplots_adjust(hspace=0.1, wspace=0.4)
        else:
            ax.set_facecolor('0.98')
            if maps:
                ax = format_map(ax, lats=lats, lons=lons)

    return fig, ax

def add_labels(ax, xlabel, ylabel, **label_kwargs):
    # Set default values
    label_kwargs['fontsize'] = label_kwargs.get('fontsize',
                                                config['label_size']*config['scale'])
    label_kwargs['labelpad'] = label_kwargs.get('labelpad',
                                                config['label_pad'])
    labelsize = label_kwargs.pop('labelsize',
                                 config['tick_size']*config['scale'])

    # Set labels
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel(ylabel, **label_kwargs)
    ax.tick_params(axis='both', which='both', labelsize=labelsize)
    return ax

def add_legend(ax, **legend_kwargs):
    legend_kwargs['frameon'] = legend_kwargs.get('frameon', False)
    legend_kwargs['fontsize'] = legend_kwargs.get('fontsize',
                                                  (config['label_size']*
                                                   config['scale']))

    # Remove duplicates from legend
    try:
        handles = legend_kwargs.pop('handles')
        labels = legend_kwargs.pop('labels')
    except:
        handles, labels = ax.get_legend_handles_labels()
    labels = OrderedDict(zip(labels, handles))
    handles = labels.values()
    labels = labels.keys()

    ax.legend(handles=handles, labels=labels, **legend_kwargs)
    return ax

def add_title(ax, title, **title_kwargs):
    title_kwargs['y'] = title_kwargs.get('y', config['title_loc'])
    title_kwargs['pad'] = title_kwargs.get('pad', config['title_pad'])
    title_kwargs['fontsize'] = title_kwargs.get('fontsize',
                                                config['title_size']*config['scale'])
    title_kwargs['va'] = title_kwargs.get('va', 'bottom')
    ax.set_title(title, **title_kwargs)
    return ax

def add_subtitle(ax, subtitle, **kwargs):
    y = kwargs.pop('y', config['title_loc'])
    kwargs['ha'] = kwargs.get('ha', 'center')
    kwargs['va'] = kwargs.get('va', 'bottom')
    kwargs['xytext'] = kwargs.get('xytext', (0, config['title_pad']))
    kwargs['textcoords'] = kwargs.get('textcoords', 'offset points')
    kwargs['fontsize'] = kwargs.get('fontsize',
                                    config['subtitle_size']*config['scale'])
    subtitle = ax.annotate(s=subtitle, xy=(0.5, y), xycoords='axes fraction',
                           **kwargs)
    return ax

def get_square_limits(xdata, ydata, **kw):
    # Get data limits
    dmin = min(np.min(xdata), np.min(ydata))
    dmax = max(np.max(xdata), np.max(ydata))
    pad = (dmax - dmin)*0.05
    dmin -= pad
    dmax += pad

    try:
        # get lims
        ylim = kw.pop('lims')
        xlim = ylim
        xy = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    except:
        # set lims
        xlim = ylim = xy = (dmin, dmax)

    return xlim, ylim, xy, dmin, dmax

def format_map(ax, lats, lons, lat_delta=0.25, lon_delta=0.3125,
               fontsize=config['tick_size']*config['scale'],
               **gridline_kwargs):
    # Get kwargs
    gridline_kwargs['draw_labels'] = gridline_kwargs.get('draw_labels', True)
    gridline_kwargs['color'] = gridline_kwargs.get('color', 'grey')
    gridline_kwargs['linestyle'] = gridline_kwargs.get('linestyle', ':')

    # Format
    ax.set_ylim(min(lats) - lat_delta/2, max(lats) + lat_delta/2)
    ax.set_xlim(min(lons) - lon_delta/2, max(lons) + lon_delta/2)
    ax.add_feature(cf.OCEAN.with_scale('50m'), facecolor='0.98', linewidth=0.5)
    ax.add_feature(cf.LAND.with_scale('50m'), facecolor='0.98', linewidth=0.5)
    ax.add_feature(cf.STATES.with_scale('50m'), edgecolor='0.3', linewidth=0.2,
                   zorder=10)
    ax.add_feature(cf.LAKES.with_scale('50m'), facecolor='none',
                   edgecolor='0.3 ', linewidth=0.2)
    ax.coastlines(resolution='50m', color='0.2', linewidth=0.5)

    # gl = ax.gridlines(**gridline_kwargs)
    # gl.xlabel_style = {'fontsize' : fontsize}
    # gl.ylabel_style = {'fontsize' : fontsize}
    return ax

def format_cbar(cbar, cbar_title='', horizontal=False, **cbar_kwargs):
    # cbar.set_label(cbar_title, fontsize=BASEFONT*config['scale'],
    #                labelpad=CBAR_config['label_pad'])
            # x0
    if horizontal:
        x = 0.5
        y = cbar_kwargs.pop('y', -4)
        rotation = 'horizontal'
        va = 'top'
        ha = 'center'
    else:
        x = cbar_kwargs.pop('x', 5)
        y = 0.5
        rotation = 'vertical'
        va = 'center'
        ha = 'left'

    cbar.ax.tick_params(axis='both', which='both',
                        labelsize=config['tick_size']*config['scale'])
    cbar.ax.text(x, y, cbar_title, ha=ha, va=va, rotation=rotation,
                 fontsize=config['label_size']*config['scale'],
                 transform=cbar.ax.transAxes)

    return cbar

def save_fig(fig, loc, name, **kwargs):
    fig.savefig(join(loc, name + '.png'),
                bbox_inches='tight', 
                dpi=500,
                transparent=True, **kwargs)
    print('Saved %s' % name + '.png')


def plot_clusters(sv_data, ax=None, **kw):
    # Plot outlines
    lat_min, lat_max = sv_data['lat'].min().values, sv_data['lat'].max().values
    lon_min, lon_max = sv_data['lon'].min().values, sv_data['lon'].max().values
    sv = sv_data.values
    sv = np.concatenate([sv[:, 0][:, None], sv, sv[:, -1][:, None]], axis=1)
    sv = np.concatenate([sv[0, :][None, :], sv, sv[-1, :][None, :]], axis=0)
    sv = zoom(sv, 50, order=0, mode='nearest')
    sv_max = np.nanmax(sv) + 1
    colors = kw.pop('colors', 'black')
    linestyles = kw.pop('linestyles', None)
    linewidths = kw.pop('linewidths', 1.5)
    ax.contour(sv, levels=np.arange(0, sv_max + 1, 1),
               extent=[lon_min - 0.3125/2 - 0.1, 
                       lon_max + 0.3125/2 + 0.1, 
                       lat_min - 0.25/2 - 0.1, 
                       lat_max + 0.25/2 + 0.1],
               colors=colors, linewidths=linewidths, linestyles=linestyles, 
               zorder=20)
    
    # # Plot zeros
    # sv_z = sv_data.where(sv_data == 0)
    # x, y = np.meshgrid(sv_z['lon'], sv_z['lat'])
    # ax.pcolor(x, y, sv_z.values, hatch='/////', alpha=0, zorder=20,
    #           linewidths=0.5)
    
    return ax