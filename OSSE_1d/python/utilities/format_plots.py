import numpy as np
import math
from collections import OrderedDict
from os.path import join

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cf

from utilities import plot_settings as ps

# Other font details
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = 'AppleGothic'
rcParams['font.size'] = ps.LABEL_FONTSIZE*ps.SCALE
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
rcParams['axes.titlepad'] = ps.TITLE_PAD

from matplotlib.font_manager import findfont, FontProperties
font = findfont(FontProperties(family=['sans-serif']))

def color(k, cmap='CMRmap', lut=10):
    c = plt.cm.get_cmap(cmap, lut=lut)
    return colors.to_hex(c(k))

def cmap_from_color(color_high, color_low=(1, 1, 1), N=100):
    rgb_map = [color_low, colors.to_rgb(color_high)]
    cmap = colors.LinearSegmentedColormap.from_list('cmap', rgb_map, N=N)
    return cmap

def cmap_trans(cmap, ncolors=300, nalpha=20):
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.append(np.linspace(0.0, 1.0, nalpha),
                                  np.ones(ncolors-nalpha))

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=str(cmap) + '_trans',colors=color_array)

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
    max_width = fig_kwargs.get('max_width', ps.BASE_WIDTH*ps.SCALE)*cols
    max_height = fig_kwargs.get('max_height', ps.BASE_HEIGHT*ps.SCALE)*rows

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
    kw['width_ratios'] = fig_kwargs.pop('width_ratios', None)
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
        x0 = (fig_width*x0_init + cbar_pad_inches*ps.SCALE)/fig_width

        # y0
        y0 = axis.get_position().y0

        # Width
        width = 0.1*ps.SCALE/fig_width
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
        y0 = (fig_height*y0_init - cbar_pad_inches*ps.SCALE)/fig_height

        # Height
        height = 0.1*ps.SCALE/fig_height

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
                                                ps.LABEL_FONTSIZE*ps.SCALE)
    label_kwargs['labelpad'] = label_kwargs.get('labelpad',
                                                ps.LABEL_PAD)
    labelsize = label_kwargs.pop('labelsize',
                                 ps.TICK_FONTSIZE*ps.SCALE)

    # Set labels
    ax.set_xlabel(xlabel, **label_kwargs)
    ax.set_ylabel(ylabel, **label_kwargs)
    ax.tick_params(axis='both', which='both', labelsize=labelsize)
    return ax

def add_legend(ax, **legend_kwargs):
    legend_kwargs['frameon'] = legend_kwargs.get('frameon', False)
    legend_kwargs['fontsize'] = legend_kwargs.get('fontsize',
                                                  (ps.LABEL_FONTSIZE*
                                                   ps.SCALE))

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
    title_kwargs['y'] = title_kwargs.get('y', ps.TITLE_LOC)
    title_kwargs['pad'] = title_kwargs.get('pad', ps.TITLE_PAD)
    title_kwargs['fontsize'] = title_kwargs.get('fontsize',
                                                ps.TITLE_FONTSIZE*ps.SCALE)
    title_kwargs['va'] = title_kwargs.get('va', 'bottom')
    ax.set_title(title, **title_kwargs)
    return ax

def add_subtitle(ax, subtitle, **kwargs):
    y = kwargs.pop('y', ps.TITLE_LOC)
    kwargs['ha'] = kwargs.get('ha', 'center')
    kwargs['va'] = kwargs.get('va', 'bottom')
    kwargs['xytext'] = kwargs.get('xytext', (0, ps.TITLE_PAD))
    kwargs['textcoords'] = kwargs.get('textcoords', 'offset points')
    kwargs['fontsize'] = kwargs.get('fontsize',
                                    ps.SUBTITLE_FONTSIZE*ps.SCALE)
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

def format_map(ax, lats, lons,
               fontsize=ps.TICK_FONTSIZE*ps.SCALE,
               **gridline_kwargs):
    # Get kwargs
    gridline_kwargs['draw_labels'] = gridline_kwargs.get('draw_labels', True)
    gridline_kwargs['color'] = gridline_kwargs.get('color', 'grey')
    gridline_kwargs['linestyle'] = gridline_kwargs.get('linestyle', ':')

    # Format
    ax.set_ylim(min(lats), max(lats))
    ax.set_xlim(min(lons), max(lons))
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
    # cbar.set_label(cbar_title, fontsize=BASEFONT*ps.SCALE,
    #                labelpad=CBAR_ps.LABEL_PAD)
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
                        labelsize=ps.TICK_FONTSIZE*ps.SCALE)
    cbar.ax.text(x, y, cbar_title, ha=ha, va=va, rotation=rotation,
                 fontsize=ps.LABEL_FONTSIZE*ps.SCALE,
                 transform=cbar.ax.transAxes)

    return cbar

def save_fig(fig, loc, name, **kwargs):
    fig.savefig(join(loc, name + '.png'),
                bbox_inches='tight', dpi=500,
                transparent=True, **kwargs)
    print('Saved %s' % name + '.png')
