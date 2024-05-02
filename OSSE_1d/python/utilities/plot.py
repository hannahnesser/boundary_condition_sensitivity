import numpy as np
import math

import sys
sys.path.append('.')
from utilities import format_plots as fp
from utilities import plot_settings as ps

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

def plot_one_to_one(ax):
    xlim, _, _, _, _ = fp.get_square_limits(ax.get_xlim(),
                                               ax.get_ylim())
    ax.plot(xlim, xlim, c='0.1', lw=2, ls=':',
            alpha=0.5, zorder=0)
    return ax

